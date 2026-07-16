import os
import cv2
import re
import random
import pytesseract
import numpy as np
from typing import List, Tuple, Iterator
from ollama import chat
from .base import BasePreprocessor
from ..datamodels import DocumentRotation 
from medilect.config.settings import configure_tesseract

class AutoOrientPreprocessor(BasePreprocessor):
    """Corrects 90, 180, and 270 degree page rotations."""
    
    def __init__(self, min_confidence: float = 4.0, vlm_fallback_model: str = "qwen3-vl:8b"):
        self.min_confidence = min_confidence
        self.vlm_model = vlm_fallback_model
        
        configure_tesseract()

    def run(self, images: List[np.ndarray]) -> List[np.ndarray]:
        corrected_images = []
        for img in images:
            angle = self._detect_angle(img)
            self.last_detected_angle = angle
            corrected_images.append(self._rotate_image(img, angle))
        return corrected_images

    def _rotate_image(self, img: np.ndarray, angle: int) -> np.ndarray:
        if angle == 0:
            return img
        
        rotation_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        return cv2.rotate(img, rotation_map[angle])

    def random_patch_generator(self, img: np.ndarray, num_patches: int = 100, patch_scale: float = 0.5) -> Iterator[Tuple[str, np.ndarray]]:
        """Yields the full image, followed by random patches of the image."""
        h, w = img.shape[:2] 
        yield "Full Image", img
        
        patch_w = int(w * patch_scale)
        patch_h = int(h * patch_scale)
        
        for i in range(num_patches):
            x = random.randint(0, max(0, w - patch_w))
            y = random.randint(0, max(0, h - patch_h))
            crop_img = img[y : y + patch_h, x : x + patch_w]
            yield f"Patch_{i+1}", crop_img

    def _detect_angle(self, img: np.ndarray) -> int:
        # --- Step 1: Try the full image ---
        try:
            osd = pytesseract.image_to_osd(img)
            angle = int(re.search(r'(?<=Rotate: )\d+', osd).group(0))
            conf = float(re.search(r'(?<=Orientation confidence: )[\d\.]+', osd).group(0))
            
            if conf > self.min_confidence:
                print(f"\n Global Rotation detected: {angle}° (Confidence: {conf})")
                return angle
        except Exception:
            pass

        # --- Step 2: Try random patches ---
        print(" Full image confidence too low. Trying random patches...")
        generator = self.random_patch_generator(img, num_patches=100, patch_scale=0.5)
        next(generator) # Skip the full image
        
        for name, crop_img in generator:
            try:
                osd = pytesseract.image_to_osd(crop_img)
                angle = int(re.search(r'(?<=Rotate: )\d+', osd).group(0))
                conf = float(re.search(r'(?<=Orientation confidence: )[\d\.]+', osd).group(0))
        
                #print(f"Patch Conf: {conf}")
                if conf > self.min_confidence:
                    print(f" Tesseract found rotation via '{name}': {angle}°")
                    return angle
            except Exception:
                pass

        # --- Step 3: VLM Fallback ---
        print(f"\n Exhausted all Tesseract patches. Falling back to VLM ({self.vlm_model})...")
        
        try:
            max_dim = 896
            h, w = img.shape[:2]
            img_scaled = img 
            
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img_scaled = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            success, buffer = cv2.imencode('.jpg', img_scaled)
            if not success:
                raise ValueError("Failed to encode image to bytes.")
            img_bytes = buffer.tobytes()

            # Call local VLM using dynamically configured model
            response = chat(
                model=self.vlm_model,
                format='json',
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            "You are an expert document layout analyzer specialized in optical orientation correction. "
                            "Analyze the text alignment, header positions, and reading order in the image to determine "
                            "how it must be rotated to be perfectly upright and readable from left-to-right, top-to-bottom."
                        )
                    },
                    {
                        'role': 'user',
                        'content': (
                            "Analyze this document image and determine the exact CLOCKWISE rotation angle in degrees "
                            "required to make the text perfectly upright. \n\n"
                            "Use this strict guide:\n"
                            "- If the document is already upright, return 0.\n"
                            "- If the document is sideways (rotated counter-clockwise), return 90.\n"
                            "- If the document is completely upside down, return 180.\n"
                            "- If the document is sideways (rotated clockwise), return 270.\n\n"
                            "You MUST return a valid JSON object matching this exact schema layout:\n"
                            "{\n"
                            '  "thinking": "Brief explanation of text orientation clues noticed.",\n'
                            '  "rotation_angle": 0\n'
                            "}\n"
                            "Output only the JSON object."
                        ),
                        'images': [img_bytes] 
                    }
                ],
                options={
                    'temperature': 0.0
                }
            )
            # --- Step 4: Parse and Validate Safely ---
            raw_content = response.message.content
            
            # Locate outermost braces (Guarantees clean extraction even if LLM adds preamble text)
            start_idx = raw_content.find('{')
            end_idx = raw_content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_string = raw_content[start_idx : end_idx + 1]
            else:
                print(f"VLM failed to output JSON! Raw output:\n{raw_content}")
                return 0
                    
            # SANITIZE: Scrub raw control characters inside string properties
            json_string = re.sub(r'[\x00-\x1F\x7F]', ' ', json_string)
                
            cleaned_data = DocumentRotation.model_validate_json(json_string)
            vlm_angle = cleaned_data.rotation_angle
            
            if vlm_angle in [0, 90, 180, 270]:
                print(f"VLM successfully detected rotation: {vlm_angle}°")
                return vlm_angle
            
            print(f"VLM returned invalid angle ({vlm_angle}°). Defaulting to 0°.")
            return 0

        except Exception as e:
            print(f"VLM Fallback failed ({type(e).__name__}: {e}). Defaulting to 0°.")
            return 0