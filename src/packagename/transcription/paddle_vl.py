import cv2
import torch
from PIL import Image
from typing import List
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM

class PaddleVLTranscriber:
    """
    Transcription engine using Baidu's PaddleOCR-VL.
    Implements official chat templates and bfloat16 precision for stability.
    """
    def __init__(self, model_id: str = "PaddlePaddle/PaddleOCR-VL", task: str = "ocr"): 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        
        self.prompts = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
        }
        self.prompt_text = self.prompts[task]   # <-- this was missing

        print(f"⚙️ Booting PaddleOCR-VL ({model_id}) on {self.device}")
        KNOWN_GOOD_REVISION = "f54aa90d389e98361cf295b7f4544bfb7452996d"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=KNOWN_GOOD_REVISION,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            #revision=KNOWN_GOOD_REVISION,
            trust_remote_code=True,
            use_fast=True
        )

    def run(self, crops: List[np.ndarray]) -> List[str]:
        """
        Transcribes a batch of OpenCV images into text using the VLM chat template.
        """
        transcriptions = []
        
        for idx, crop in enumerate(crops):
            # Guard against empty arrays
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                transcriptions.append("")
                continue

            # 1. Convert OpenCV BGR to PIL RGB
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_crop)
            
            # 2. Build the official Chat Template payload
            messages = [
                {
                    "role": "user",         
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": self.prompt_text},
                    ]
                }
            ]
            
            try:
                # 3. Apply the template to inject invisible control tokens
                inputs = self.processor.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    add_generation_prompt=True,     
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=1024)
                
                # 4. Decode and clean
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                
                # Strip out the prompt if the model echoed it back
                clean_text = generated_text
                if self.prompt_text in clean_text:
                    clean_text = clean_text.split(self.prompt_text)[-1]
                
                transcriptions.append(clean_text.strip())
                
            except Exception as e:
                print(f"⚠️ Failed to transcribe crop {idx}: {e}")
                transcriptions.append("")
            
        return transcriptions