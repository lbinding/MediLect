import gc
import inspect
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from types import MethodType

class PaddleVLTranscriber:
    """
    Loads PaddleOCR-VL (0.9B) once and reuses it for every crop/page.

    NOTE: only supports ELEMENT-level recognition via transformers (OCR /
    table / chart / formula on a single cropped region) — not full-page
    layout analysis. Feed it pre-cropped regions, not raw full pages.
    """

    _PROMPTS = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "chart": "Chart Recognition:",
        "formula": "Formula Recognition:",
    }

    def __init__(
        self,
        model_name: str = "PaddlePaddle/PaddleOCR-VL",
        use_gpu: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        task: str = "ocr",
        max_new_tokens: int = 2048,
        attn_implementation: str = "flash_attention_2",
    ):
        if task not in self._PROMPTS:
            raise ValueError(f"task must be one of {list(self._PROMPTS)}")

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if use_gpu and self.device == "cpu":
            print("⚠️  GPU requested but CUDA not available — falling back to CPU.")

        self.task = task
        self.max_new_tokens = max_new_tokens

        print(f"⏳ Loading PaddleOCR-VL model '{model_name}' ({self.device})...")

        load_kwargs = dict(trust_remote_code=True, dtype=dtype)
        
        # ------------------------------------------------------------------ #
        # CONFIGURATION & MEMORY PATCH
        # ------------------------------------------------------------------ #
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        load_kwargs["config"] = config

        def _patch_cache_position(model):
            """
            Wraps prepare_inputs_for_generation to safely handle cache_position 
            without altering the modern DynamicCache object type.
            """
            if getattr(model, "_cache_position_patched", False):
                return

            original_prepare = model.prepare_inputs_for_generation 

            def patched_prepare_inputs_for_generation(
                self, input_ids, cache_position=None, past_key_values=None, **kwargs
            ):
                if cache_position is None:
                    past_length = 0
                    if past_key_values is not None:
                        if hasattr(past_key_values, "get_seq_length"):
                            past_length = past_key_values.get_seq_length()
                        elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                            past_length = past_key_values[0][0].shape[-2]
                            
                    cache_position = torch.arange(
                        past_length, past_length + input_ids.shape[1], device=input_ids.device
                    )

                model_inputs = original_prepare(
                    input_ids, 
                    past_key_values=past_key_values, 
                    cache_position=cache_position,
                    **kwargs
                )
                
                if "cache_position" not in model_inputs:
                    model_inputs["cache_position"] = cache_position
                    
                return model_inputs

            model.prepare_inputs_for_generation = MethodType(patched_prepare_inputs_for_generation, model)
            model._cache_position_patched = True
            
            model.config.use_cache = True
            if hasattr(model.config, "text_config"):
                model.config.text_config.use_cache = True

        def attempt_model_load(attn):
            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_name, attn_implementation=attn, **load_kwargs
                ).to(self.device).eval()
            except KeyError as e:
                if 'default' in str(e):
                    patched = False
                    for mod_name, mod in list(sys.modules.items()):
                        if "modeling_paddleocr_vl" in mod_name and hasattr(mod, "ROPE_INIT_FUNCTIONS"):
                            def paddle_default_rope(cfg, dev):
                                dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
                                base = getattr(cfg, "rope_theta", 10000.0)
                                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(dev) / dim))
                                return inv_freq, 1.0

                            if "default" not in mod.ROPE_INIT_FUNCTIONS:
                                mod.ROPE_INIT_FUNCTIONS["default"] = paddle_default_rope
                                patched = True

                            if hasattr(mod, "RotaryEmbedding") and not hasattr(
                                mod.RotaryEmbedding, "compute_default_rope_parameters"
                            ):
                                def compute_default_rope_parameters(self, config=None, device=None, **kwargs):
                                    cfg = config if config is not None else self.config
                                    dev = device if device is not None else getattr(self, "inv_freq", torch.tensor(0)).device
                                    return paddle_default_rope(cfg, dev)

                                mod.RotaryEmbedding.compute_default_rope_parameters = compute_default_rope_parameters
                                patched = True
                            
                            if patch_create_causal_mask(mod):
                                patched = True
    
                    if patched:
                        print("🔧 Successfully hot-patched PaddleVL's RoPE dispatch! Retrying load...")
                        return AutoModelForCausalLM.from_pretrained(
                            model_name, attn_implementation=attn, **load_kwargs
                        ).to(self.device).eval()
                raise

        def patch_create_causal_mask(mod):
            """
            transformers' create_causal_mask() has drifted from the signature
            PaddleOCR-VL's vendored forward() was written against.
            """
            if not hasattr(mod, "create_causal_mask"):
                return False
            original_fn = mod.create_causal_mask
            if getattr(original_fn, "_cache_position_filtered", False):
                return False 

            sig = inspect.signature(original_fn)
            accepted_params = set(sig.parameters.keys())
            required_params = {
                name for name, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            }

            def filtered_create_causal_mask(*args, **kwargs):
                # Dynamically align "embeds" variable names to the current transformers signature
                if "inputs_embeds" in accepted_params and "input_embeds" in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                elif "input_embeds" in accepted_params and "inputs_embeds" in kwargs:
                    kwargs["input_embeds"] = kwargs.pop("inputs_embeds")

                filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}

                missing = required_params - filtered_kwargs.keys()
                if missing:
                    raise TypeError(
                        f"patched create_causal_mask: still missing required args {missing} "
                        f"after rename/filter — inspect the caller's kwargs: {list(kwargs.keys())}"
                    )

                return original_fn(*args, **filtered_kwargs)

            filtered_create_causal_mask._cache_position_filtered = True
            mod.create_causal_mask = filtered_create_causal_mask
            return True
        
        try:
            self.model = attempt_model_load(attn_implementation)
        except Exception as e:
            print(f"⚠️  '{attn_implementation}' unavailable ({e}); falling back to 'sdpa'.")
            self.model = attempt_model_load("sdpa")

        _patch_cache_position(self.model)  

        for mod_name, mod in list(sys.modules.items()):
            if "modeling_paddleocr_vl" in mod_name:
                patch_create_causal_mask(mod)

        # ------------------------------------------------------------------ #

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"

        print("✅ PaddleOCR-VL model loaded.")

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def run(
        self,
        crops: List[Union[np.ndarray, Image.Image]],
        task: Optional[str] = None,
    ) -> List[str]:
        if not crops:
            return []

        prompt = self._PROMPTS[task or self.task]
        results = []
        
        for crop in crops:
            pil_img = self._to_pil(crop)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ]
                }
            ]
            
            try:
                # 1. Format the text string with the <image> placeholder
                text_prompt = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )

                # 2. Explicitly pass BOTH the text and the image to the processor
                # This guarantees `pixel_values` (the actual image data) is generated!
                inputs = self.processor(
                    text=text_prompt, 
                    images=pil_img, 
                    return_tensors="pt"
                ).to(self.device)

                print("pixel_values" in inputs, inputs.get("pixel_values", torch.tensor([])).shape,
                    inputs.get("pixel_values", torch.tensor([])).float().mean().item())

                with torch.inference_mode():
                    generated = self.model.generate(
                        **inputs, 
                        max_new_tokens=self.max_new_tokens, 
                        do_sample=False, 
                        use_cache=True
                    )
                    print(torch.isnan(generated).any().item(), torch.isinf(generated).any().item())


                # Strip the echoed prompt tokens, keep only newly generated content
                input_len = inputs["input_ids"].shape[1]
                new_tokens = generated[:, input_len:]
                decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
                results.append(decoded.strip())
                
            except Exception as e:
                print(f" ⚠️ Failed during generation: {e}")
                results.append("")

        return results

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_pil(crop: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Accepts a cv2-style BGR numpy array or a PIL Image."""
        if isinstance(crop, Image.Image):
            return crop.convert("RGB")
        if crop.ndim == 3 and crop.shape[2] == 3:
            crop = crop[:, :, ::-1]  # BGR -> RGB
        return Image.fromarray(crop)

    def unload(self):
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()