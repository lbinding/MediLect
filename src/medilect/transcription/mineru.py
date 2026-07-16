from __future__ import annotations
import gc
from typing import List, Union
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from mineru_vl_utils import MinerUClient


class MinerUTranscriber:
    """
    Loads the MinerU2.5 VLM once and reuses it for every page/document.
    Uses the `transformers` backend of mineru-vl-utils (simplest to install,
    slower than vLLM — fine for local/dev testing).
    """

    def __init__(
        self,
        model_name: str = "opendatalab/MinerU2.5-2509-1.2B",
        use_gpu: bool = True,
        dtype: str = "auto",
        image_analysis: bool = False,
    ):
        device_map = "auto" if use_gpu and torch.cuda.is_available() else "cpu"
        if use_gpu and device_map == "cpu":
            print("⚠️  GPU requested but CUDA not available — falling back to CPU.")

        print(f"⏳ Loading MinerU model '{model_name}' ({device_map})...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        if not hasattr(self.model.config, "max_position_embeddings"):
            self.model.config.max_position_embeddings = 32768

        self.client = MinerUClient(
            backend="transformers",
            model=self.model,
            processor=self.processor,
            image_analysis=image_analysis,
        )
        print("✅ MinerU model loaded.")

    # ------------------------------------------------------------------ #
    # Public interface — kept identical to your existing pipeline's
    # `transcriber.run(crops=...)` call signature.
    # ------------------------------------------------------------------ #
    def run(self, crops: List[Union[np.ndarray, Image.Image]]) -> List[str]:
        if not crops:
            return []

        pil_crops = [self._to_pil(c) for c in crops]

        with torch.inference_mode():
            # Batches layout-detect + content-extract across all crops at once
            batched_blocks = self.client.batch_two_step_extract(pil_crops)

        return [self._blocks_to_text(blocks) for blocks in batched_blocks]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_pil(crop: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Accepts a cv2-style BGR numpy array or a PIL Image."""
        if isinstance(crop, Image.Image):
            return crop.convert("RGB")
        if crop.ndim == 3 and crop.shape[2] == 3:
            crop = crop[:, :, ::-1]  # BGR -> RGB
        return Image.fromarray(crop)

    @staticmethod
    def _blocks_to_text(blocks) -> str:
        """
        Joins a page's ContentBlock list into a single markdown-ish string,
        in the order MinerU returns them (already reading-order sorted).
        Each ContentBlock generally exposes a `.content` attribute; this
        falls back gracefully if the field name differs in your installed
        version — run `print(blocks[0])` once to confirm the schema.
        """
        parts = []
        for block in blocks:
            text = getattr(block, "content", None)
            if text is None and isinstance(block, dict):
                text = block.get("content") or block.get("text")
            if text:
                parts.append(str(text).strip())
        return "\n\n".join(parts)

    def unload(self):
        """Free GPU memory explicitly, e.g. between test runs."""
        del self.model
        del self.processor
        del self.client
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()