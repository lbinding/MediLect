import cv2
import numpy as np
import re
from typing import List, Tuple
from ollama import chat
from .base import BasePreprocessor
from ..datamodels import DocumentComposition

class SpreadSplitterPreprocessor(BasePreprocessor):
    """Detects composite 2-page scans and slices them down the optimal gutter."""
    
    def __init__(self, vlm_model: str = "qwen3-vl:8b", canny_thresholds: Tuple[int, int] = (50, 150)):
        self.vlm_model = vlm_model
        self.canny_low, self.canny_high = canny_thresholds

    def run(self, images: List[np.ndarray]) -> List[np.ndarray]:
        processed_stream = []
        
        for img in images:
            if self._is_composite_spread(img):
                left, right = self._execute_smart_split(img)
                processed_stream.extend([left, right])
            else:
                processed_stream.append(img)
                
        return processed_stream

    def _is_composite_spread(self, img: np.ndarray) -> bool:
        # Add a quick check: if the image is taller than it is wide, it's likely a single page
        h, w = img.shape[:2]
        if h >= w:
            return False
        
        # --- Step 1: Resize & Encode to Bytes ---
        try:
            # Prevent Context Overflow: Resize image so the longest side is max 1024px
            max_dim = 1024
            h, w = img.shape[:2]
            img_scaled = img
            
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img_scaled = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            success, buffer = cv2.imencode('.jpg', img_scaled)
            if not success:
                raise ValueError("Failed to encode image to bytes.")
            img_bytes = buffer.tobytes()

            # --- Step 2: The VLM Call ---
            print(f"👁️ Asking VLM ({self.vlm_model}) to check document composition...")
            response = chat(
                model=self.vlm_model,
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            "You are an expert document layout analyzer. Your task is to determine if a scanned image "
                            "is a 'composite' (two separate physical pages scanned simultaneously, like an open book) "
                            "or a single physical page. Do not be fooled by single pages with multi-column text layouts. "
                            "Look for physical boundaries like a central binding gutter. "
                            "You MUST return a valid JSON object. Do NOT output the schema itself. "
                            'Your output must exactly match this structure: '
                            '{"thinking": "...", "is_composite_spread": "..."}'
                        )
                    },
                    {
                        'role': 'user',
                        'content': "Analyze this document's physical layout and output the JSON.",
                        'images': [img_bytes]
                    }
                ],
                options={
                    'temperature': 0.0, 
                    'num_ctx': 5120,     
                    'num_predict': 1536   
                }
            )

            # --- Step 3: Parse and Validate ---
            raw_content = response['message']['content'] if isinstance(response, dict) else response.message.content
            
            # Safely locate outermost braces
            start_idx = raw_content.find('{')
            end_idx = raw_content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_string = raw_content[start_idx : end_idx + 1]
                json_string = re.sub(r'[\x00-\x1F\x7F]', ' ', json_string)
                
                cleaned_data = DocumentComposition.model_validate_json(json_string)
                
                if cleaned_data.is_composite_spread:
                    print("VLM Analysis: Multiple physical pages detected!")
                    return True
                
                print("VLM Analysis: Single physical page detected.")
                return False
            
            print(f" VLM returned invalid format. Raw output: '{raw_content}'")
            return False

        except Exception as e:
            print(f"VLM Composition Check failed ({type(e).__name__}: {e}). Defaulting to False.")
            return False
        
    def _execute_smart_split(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = img.shape[:2]

        # --- Define the search area ---
        # We only want to look for the cut in the middle 20% of the document
        search_start = int(width * 0.40)
        search_end = int(width * 0.60)
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract just the center strip
        center_strip = gray[:, search_start:search_end]

        # --- Find the "Path of Least Resistance" ---
        blurred = cv2.GaussianBlur(center_strip, (5, 5), 0)
        # Uses dynamically configured Canny thresholds
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Sum the edges vertically for every column in our search strip
        vertical_projection = np.sum(edges, axis=0)

        # Find the local x-coordinate with the absolute minimum text/edges
        best_cut_local = np.argmin(vertical_projection)
        
        # Translate that back to the global x-coordinate of the full image
        best_cut_global = search_start + best_cut_local

        # --- Step 3: Perform the Split ---
        left_page = img[:, :best_cut_global]
        right_page = img[:, best_cut_global:]

        print(f" Smart split executed at pixel column {best_cut_global} (out of {width})")
        
        return left_page, right_page