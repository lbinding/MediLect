import os
import cv2
import time
import numpy as np
from typing import List, Optional
from .base import BaseTranscriber
import torch
from doctr.models import ocr_predictor

class DocTRTranscriber(BaseTranscriber):
    """
    Transcription engine using Mindee's docTR.
    Fast, lightweight, and highly accurate for printed text.
    """
    
    def __init__(self, use_gpu: bool = True):

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"⚙️ Booting docTR OCR Engine on {self.device}...")
        
        # Load the end-to-end predictor (Detection + Recognition)
        self.predictor = ocr_predictor(pretrained=True)
        
        # Safely map to GPU if requested and available
        if self.device == "cuda":
            if hasattr(self.predictor, "to"):
                self.predictor = self.predictor.to(self.device)
            else:
                # Fallback mapping for some internal docTR model architectures
                self.predictor.det_predictor.model.to(self.device)
                self.predictor.reco_predictor.model.to(self.device)

    def run(self, crops: List[np.ndarray], full_page_image: Optional[np.ndarray] = None) -> List[str]:
        transcriptions = []
        total_crops = len(crops)
        
        for idx, crop in enumerate(crops):
            print(f"         -> [docTR] Reading crop {idx+1}/{total_crops}...", end="", flush=True)
            start_time = time.time()
            
            if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                transcriptions.append("")
                print(" skipped (empty)")
                continue

            try:
                # docTR expects standard RGB OpenCV images
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                # docTR can digest multiple images at once, but we run sequentially 
                # to keep your telemetry accurate and memory rock-solid
                result = self.predictor([rgb_crop])
                
                # docTR returns a heavily nested object: Pages -> Blocks -> Lines -> Words
                extracted_text = ""
                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            line_text = " ".join([word.value for word in line.words])
                            extracted_text += line_text + "\n"
                
                clean_text = extracted_text.strip()
                transcriptions.append(clean_text)
                
                elapsed = time.time() - start_time
                print(f" done in {elapsed:.2f}s")
                
            except Exception as e:
                print(f" ⚠️ Failed: {e}")
                transcriptions.append("")

        return transcriptions