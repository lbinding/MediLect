# packagename/transcription/paddle_vl.py
from __future__ import annotations

import gc
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


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
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation=attn_implementation, **load_kwargs
            ).to(self.device).eval()
        except Exception as e:
            # flash-attn not installed, or GPU too old to support it (needs
            # Ampere or newer + `pip install flash-attn`) -> fall back safely
            print(f"⚠️  '{attn_implementation}' unavailable ({e}); falling back to 'sdpa'.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation="sdpa", **load_kwargs
            ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        # Left-padding is required for correct batched generation with a
        # decoder-only model (otherwise generated text gets misaligned).
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"

        print("✅ PaddleOCR-VL model loaded.")

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
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
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