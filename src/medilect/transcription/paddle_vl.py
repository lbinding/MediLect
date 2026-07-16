import gc
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from types import MethodType
import inspect
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
        import sys
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        load_kwargs["config"] = config
        def _patch_cache_position(model):
            """
            Wraps prepare_inputs_for_generation to safely handle cache_position 
            without altering the modern DynamicCache object type.
            """
            if getattr(model, "_cache_position_patched", False):
                return

            original_prepare = model.prepare_inputs_for_generation  # already bound

            def patched_prepare_inputs_for_generation(
                self, input_ids, cache_position=None, past_key_values=None, **kwargs
            ):
                # 1. Fallback calculator if cache_position was dropped by a prior function
                if cache_position is None:
                    past_length = 0
                    if past_key_values is not None:
                        # Safely check for the modern Cache object method
                        if hasattr(past_key_values, "get_seq_length"):
                            past_length = past_key_values.get_seq_length()
                        # Fallback just in case it ever IS a tuple
                        elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                            past_length = past_key_values[0][0].shape[-2]
                            
                    cache_position = torch.arange(
                        past_length, past_length + input_ids.shape[1], device=input_ids.device
                    )

                # 2. Call original function AND PASS cache_position ALONG!
                model_inputs = original_prepare(
                    input_ids, 
                    past_key_values=past_key_values, 
                    cache_position=cache_position,
                    **kwargs
                )
                
                # 3. Re-inject cache_position if the older vendored code forgot to return it
                if "cache_position" not in model_inputs:
                    model_inputs["cache_position"] = cache_position
                    
                return model_inputs

            model.prepare_inputs_for_generation = MethodType(patched_prepare_inputs_for_generation, model)
            model._cache_position_patched = True
            
            # Force cache ON in the configuration
            model.config.use_cache = True
            if hasattr(model.config, "text_config"):
                model.config.text_config.use_cache = True

        def attempt_model_load(attn):
            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_name, attn_implementation=attn, **load_kwargs
                ).to(self.device).eval()
            except KeyError as e:
                # 2. If it crashes, the Paddle script has already been loaded into memory.
                # We find it, inject the missing "default" function they forgot, and retry.
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

                            # --- NEW: also patch the RotaryEmbedding class itself ---
                            # Newer transformers' _init_weights calls
                            # module.compute_default_rope_parameters(config) directly
                            # for rope_type == "default", bypassing ROPE_INIT_FUNCTIONS
                            # entirely. The vendored class doesn't define this method.
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
    
                            # ----------------------------------------------------------

                    if patched:
                        print("🔧 Successfully hot-patched PaddleVL's RoPE dispatch (dict + class method)! Retrying load...")
                        return AutoModelForCausalLM.from_pretrained(
                            model_name, attn_implementation=attn, **load_kwargs
                        ).to(self.device).eval()
                # If it's a different KeyError, re-raise it
                raise

        def patch_create_causal_mask(mod):
            """
            transformers' create_causal_mask() has drifted from the signature
            PaddleOCR-VL's vendored forward() was written against:
            - some kwargs were dropped (e.g. cache_position in some versions)
            - some kwargs were renamed (e.g. inputs_embeds -> input_embeds)
            We rename known-renamed kwargs, then filter anything still unrecognized,
            so the call matches whatever signature is actually installed.
            """
            if not hasattr(mod, "create_causal_mask"):
                return False
            original_fn = mod.create_causal_mask
            if getattr(original_fn, "_cache_position_filtered", False):
                return False  # already patched

            sig = inspect.signature(original_fn)
            accepted_params = set(sig.parameters.keys())
            required_params = {
                name for name, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            }

            # Known renames between older vendored code and current transformers.
            # Left side = what PaddleOCR-VL's code passes; right = what's currently expected.
            RENAME_MAP = {
                "inputs_embeds": "input_embeds",
            }

            def filtered_create_causal_mask(*args, **kwargs):
                # Apply renames first, without clobbering an already-correct kwarg.
                for old_name, new_name in RENAME_MAP.items():
                    if old_name in kwargs and new_name not in kwargs:
                        kwargs[new_name] = kwargs.pop(old_name)

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
            # flash-attn not installed, or GPU too old to support it...
            print(f"⚠️  '{attn_implementation}' unavailable ({e}); falling back to 'sdpa'.")
            self.model = attempt_model_load("sdpa")

        
        _patch_cache_position(self.model)  

        for mod_name, mod in list(sys.modules.items()):
            if "modeling_paddleocr_vl" in mod_name:
                patch_create_causal_mask(mod)

        # ------------------------------------------------------------------ #

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        # Left-padding is required for correct batched generation with a
        # decoder-only model (otherwise generated text gets misaligned).
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"

        print("✅ PaddleOCR-VL model loaded.")
        print(self.model.generation_config.eos_token_id)
        print(self.processor.tokenizer.eos_token_id)

    # ------------------------------------------------------------------ #
    # Public interface — matches your existing `transcriber.run(crops=...)`
    # call signature.
    # ------------------------------------------------------------------ #
    def run(
        self,
        crops: List[Union[np.ndarray, Image.Image]],
        task: Optional[str] = None,
    ) -> List[str]:
        if not crops:
            return []

        prompt = self._PROMPTS[task or self.task]
        
        # 1. FIX: The chat template MUST include the image placeholder dict, 
        # otherwise the VLM won't know where to insert the visual features.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        results = []
        
        # 2. FIX: Process sequentially to prevent massive CUDA OOM crashes.
        # (45 crops * 3550 image tokens = ~160,000 tokens in a single batch!)
        for crop in crops:
            pil_img = self._to_pil(crop)
            
            inputs = self.processor(
                text=text, images=pil_img, return_tensors="pt"
            )
            inputs = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True
                )

            # Strip the echoed prompt tokens, keep only newly generated content
            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated[:, input_len:]
            decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
            results.append(decoded.strip())

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