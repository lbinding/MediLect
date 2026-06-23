import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
from .base import BasePreprocessor

class MacroRegionExtractor(BasePreprocessor):
    """
    Decomposes full document pages into distinct semantic layout crops (Paragraphs, Headers, Tables).
    Features an automatic fallback from Microsoft Florence-2 to OpenCV Morphological Smearing.
    """
    def __init__(self, use_florence_vlm: bool = False, pad_pixels: int = 6):
        self.use_florence = use_florence_vlm
        self.pad = pad_pixels
        self.page_audit_map: Dict[int, Dict[str, Any]] = {}
        
        if self.use_florence:
            self._init_florence_engine()

    def _init_florence_engine(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            from transformers.configuration_utils import PreTrainedConfig
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            from transformers.modeling_utils import PreTrainedModel # <--- NEW GUARD

            # =================================================================
            # THE HUGGINGFACE v4.40+ TOTAL RIOT SHIELD
            # =================================================================

            # BOUNCER 1: Config 
            orig_cfg = getattr(PreTrainedConfig, "__getattr__", None)
            def _cfg_getattr(obj, key):
                if key == "forced_bos_token_id": return None
                if orig_cfg: return orig_cfg(obj, key)
                raise AttributeError(f"'{obj.__class__.__name__}' has no '{key}'")
            PreTrainedConfig.__getattr__ = _cfg_getattr

            # BOUNCER 2: Tokenizer
            orig_tok = getattr(PreTrainedTokenizerBase, "__getattr__", None)
            def _tok_getattr(obj, key):
                if key == "additional_special_tokens": return []
                if orig_tok: return orig_tok(obj, key)
                raise AttributeError(f"'{obj.__class__.__name__}' has no '{key}'")
            PreTrainedTokenizerBase.__getattr__ = _tok_getattr

            # BOUNCER 3: Scaled Dot-Product Attention (SDPA) runtime patch
            if not hasattr(PreTrainedModel, "_supports_sdpa"):
                PreTrainedModel._supports_sdpa = False

            # =================================================================

            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            print(f"⚙️ Booting Microsoft Florence-2-large-ft in {self.dtype} mode...")
            model_id = "microsoft/Florence-2-large-ft" 
            
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                dtype=self.dtype,
                trust_remote_code=True
            ).to(self.device).eval()
            
        except Exception as e:
            print(f"⚠️ Initialization failed ({e}). Falling back to OpenCV Morphological Engine.")
            self.use_florence = False
            
    def _detect_regions_florence(self, img: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str]]:
        img_h, img_w = img.shape[:2]
        task = "<OCR_WITH_REGION>" 
        
        # =================================================================
        # 1. THE EDGE-BLINDNESS CURE
        # Wrap the entire image in a 60px white border so top/bottom text 
        # is pushed into the model's optical "safe zone".
        # =================================================================
        border = 60
        padded_img = cv2.copyMakeBorder(
            img, border, border, border, border, 
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        pad_h, pad_w = padded_img.shape[:2]

        # =================================================================
        # 2. THE DYNAMIC STRIP MINER
        # Instead of 2 halves, we drop an 800px window down the page with 
        # a massive 200px overlap. This guarantees 1:1 native optical resolution.
        # =================================================================
        window_height = 800
        overlap = 200
        step = window_height - overlap
        
        all_regions = []
        y_start = 0
        
        print(f"      ↳ Strip-mining page in {window_height}px chunks to force 1:1 resolution...")
        
        while y_start < pad_h:
            y_end = min(y_start + window_height, pad_h)
            crop = padded_img[y_start:y_end, :]
            
            # Run inference on this specific horizontal strip
            crop_regions = self._run_inference_pass(crop, offset_y=y_start, task=task)
            all_regions.extend(crop_regions)
            
            if y_end == pad_h:
                break
            y_start += step

        # =================================================================
        # 3. COORDINATE RESTORATION
        # We must subtract the 60px border from the generated bounding boxes 
        # so they map perfectly back to your original, un-padded PDF page.
        # =================================================================
        final_regions = []
        for (bbox, label) in all_regions:
            x0, y0, x1, y1 = bbox
            
            x0, y0 = x0 - border, y0 - border
            x1, y1 = x1 - border, y1 - border
            
            # Clip the boxes safely inside the true image bounds
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(img_w, x1), min(img_h, y1)
            
            # Only keep the box if it didn't collapse into a 0-pixel singularity
            if x1 > x0 and y1 > y0:
                final_regions.append(((x0, y0, x1, y1), label))

        # Deduplicate the heavy overlap zones
        return self._kill_swallowed_boxes(final_regions)

    def _run_inference_pass(self, crop: np.ndarray, offset_y: int, task: str) -> List[Tuple[Tuple[int, int, int, int], str]]:
        import re
        
        # 1. CLAHE Contrast Filter (Turns grey text pitch black)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        boosted_crop = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        # =================================================================
        # THE SQUARE CANVAS TRICK
        # We pad the rectangular strip with white space until it is a perfect square.
        # This prevents the Hugging Face ViT Processor from warping the aspect ratio
        # of the letters when it downsamples the image.
        # =================================================================
        h, w = boosted_crop.shape[:2]
        max_dim = max(h, w)
        
        # Create a pure white square canvas
        square_canvas = np.full((max_dim, max_dim, 3), 255, dtype=np.uint8)
        # Paste our actual crop into the top-left corner
        square_canvas[0:h, 0:w] = boosted_crop

        pil_img = Image.fromarray(cv2.cvtColor(square_canvas, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=task, images=pil_img, return_tensors="pt").to(self.device, self.dtype)
        
        # 2. Strict Deterministic Generation
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=8192,      
            num_beams=1               
        )
        
        txt = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = self.processor.post_process_generation(txt, task=task, image_size=pil_img.size)
        
        regions = []
        data = result.get(task, {})
        
        for box, label in zip(data.get('quad_boxes', []), data.get('labels', [])):
            xs, ys = box[0::2], box[1::2]
            
            # Because we pasted the image into the top-left corner of the canvas (0,0),
            # Florence's coordinates for the text are natively perfectly aligned!
            x0, y0 = int(min(xs)), int(min(ys)) + offset_y
            x1, y1 = int(max(xs)), int(max(ys)) + offset_y
            
            # Clip the boxes to ensure they didn't bleed into the artificial white space
            x0, x1 = max(0, x0), min(w, x1)
            y0_local, y1_local = max(0, int(min(ys))), min(h, int(max(ys)))
            
            # If the box is valid, sanitize the label and save
            if x1 > x0 and y1_local > y0_local:
                safe_label = re.sub(r'[\\/*?:"<>|\n\t]', "", str(label))
                clean_label = safe_label.replace(" ", "_").strip()
                
                if not clean_label:
                    clean_label = "unnamed_text"
                clean_label = clean_label[:40]
                
                regions.append(((x0, y0, x1, y1), clean_label))

        return regions

    def _kill_swallowed_boxes(self, regions: List[Tuple[Tuple[int,int,int,int], str]]) -> List[Tuple[Tuple[int,int,int,int], str]]:
        """Removes any bounding box that is >85% physically contained inside another larger box."""
        if not regions:
            return []

        # 1. Sort boxes by pixel area, largest to smallest
        def box_area(item):
            b = item[0]
            return (b[2] - b[0]) * (b[3] - b[1])

        sorted_regions = sorted(regions, key=box_area, reverse=True)
        survivors = []

        for curr_item in sorted_regions:
            curr_box = curr_item[0]
            curr_area = box_area(curr_item)
            if curr_area <= 0:
                continue

            is_swallowed = False
            for parent_item in survivors:
                p_box = parent_item[0]

                # Calculate the pixel area of the intersection between Curr and Parent
                ix0 = max(curr_box[0], p_box[0])
                iy0 = max(curr_box[1], p_box[1])
                ix1 = min(curr_box[2], p_box[2])
                iy1 = min(curr_box[3], p_box[3])

                if ix1 > ix0 and iy1 > iy0:
                    intersection_area = (ix1 - ix0) * (iy1 - iy0)
                    overlap_ratio = intersection_area / curr_area
                    
                    # If the larger box covers more than 85% of this box, it's redundant. Kill it.
                    if overlap_ratio > 0.85:
                        is_swallowed = True
                        break

            if not is_swallowed:
                survivors.append(curr_item)

        return survivors
    def run(self, images: List[np.ndarray]) -> List[np.ndarray]:
        flat_region_stream = []
        self.page_audit_map = {}

        for page_idx, page_img in enumerate(images):
            # 1. Get raw bounding boxes from either AI or Classical Computer Vision
            if self.use_florence:
                raw_boxes = self._detect_regions_florence(page_img)
            else:
                raw_boxes = self._detect_regions_opencv_smear(page_img)

            # 2. Sort boxes top-to-bottom (reading order)
            raw_boxes = sorted(raw_boxes, key=lambda b: b[0][1])

            page_records = []
            img_h, img_w = page_img.shape[:2]

            for box_idx, (bbox, label) in enumerate(raw_boxes):
                # Apply our "Buffer of Forgiveness" inflation
                padded_box = self._inflate_box(bbox, img_w, img_h)
                x0, y0, x1, y1 = padded_box
                
                crop_img = page_img[y0:y1, x0:x1].copy()
                flat_region_stream.append(crop_img)

                page_records.append({
                    "region_id": f"region_{box_idx+1:02d}_{label}",
                    "bounding_box": padded_box,
                    "label": label,
                    "crop_array": crop_img
                })

            # Save the full visual state for the test script to interrogate
            self.page_audit_map[page_idx] = {
                "source_page_image": page_img,
                "extracted_regions": page_records
            }

        return flat_region_stream

    def _inflate_box(self, bbox: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox
        return (
            max(0, x0 - self.pad),
            max(0, y0 - self.pad),
            min(img_w, x1 + self.pad),
            min(img_h, y1 + self.pad)
        )

    def _detect_regions_opencv_smear(self, img: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """Classical fallback: Dilates ink until lines merge into distinct paragraph blobs."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Otsu binarization (Inverted: ink becomes white 255, paper becomes black 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Horizontally smear the white letters into solid lines, then merge close lines vertically
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 12))
        smeared = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(smeared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        min_area = (img.shape[0] * img.shape[1]) * 0.005 # Ignore specs smaller than 0.5% of the page

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append(((x, y, x + w, y + h), "layout_block"))

        return regions


import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from .base import BasePreprocessor

class PaddleBoxExtractor(BasePreprocessor):
    """
    Uses PaddleOCR's DBNet++ to locate sub-millimeter text boundaries.
    Runs on the highly stable v2.8.1 API.
    """
    def __init__(self, pad_pixels: int = 4):
        from paddleocr import PaddleOCR
        
        # In v2.8.1, rec=False works perfectly and disables the language model.
        self.det_engine = PaddleOCR(use_angle_cls=False, rec=False, show_log=False, use_mkldnn=False, enable_mkldnn=False)
        self.pad = pad_pixels
        self.page_audit_map: Dict[int, Dict[str, Any]] = {}

    def run(self, images: List[np.ndarray]) -> List[np.ndarray]:
        flat_region_stream = []
        self.page_audit_map = {}

        for page_idx, page_img in enumerate(images):
            img_h, img_w = page_img.shape[:2]
            
            polygons = self.det_engine.ocr(page_img, rec=False)
            
            page_records = []
            
            if polygons and polygons[0] is not None:
                sorted_polys = sorted(polygons[0], key=lambda box: box[0][1])

                for poly_idx, poly in enumerate(sorted_polys):
                    xs = [pt[0] for pt in poly]
                    ys = [pt[1] for pt in poly]
                    
                    x0, y0 = int(min(xs)), int(min(ys))
                    x1, y1 = int(max(xs)), int(max(ys))

                    px0 = max(0, x0 - self.pad)
                    py0 = max(0, y0 - self.pad)
                    px1 = min(img_w, x1 + self.pad)
                    py1 = min(img_h, y1 + self.pad)

                    box_tuple = (px0, py0, px1, py1)
                    crop_arr = page_img[py0:py1, px0:px1].copy()

                    flat_region_stream.append(crop_arr)
                    page_records.append({
                        "region_id": f"line_{poly_idx+1:03d}",
                        "bounding_box": box_tuple,
                        "label": "detected_line",
                        "crop_array": crop_arr
                    })

            self.page_audit_map[page_idx] = {
                "source_page_image": page_img,
                "extracted_regions": page_records
            }

        return flat_region_stream
    

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
from .base import BasePreprocessor

class SuryaBoxExtractor(BasePreprocessor):
    """
    Uses the Surya SegFormer for line-level text boundary detection.
    Requires: pip install surya-ocr==0.6.0
    """
    def __init__(self, pad_pixels: int = 4):
        import torch
        from surya.model.detection.model import load_model, load_processor        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️ Booting Surya Detection Engine on {self.device}...")

        self.det_model = load_model()
        self.det_processor = load_processor()
        self.det_model.to(self.device)

        self.pad = pad_pixels
        self.page_audit_map: Dict[int, Dict[str, Any]] = {}

    def run(self, images: List[np.ndarray]) -> List[np.ndarray]:
        from surya.detection import batch_text_detection

        flat_region_stream = []
        self.page_audit_map = {}

        for page_idx, page_img in enumerate(images):
            img_h, img_w = page_img.shape[:2]
            pil_img = Image.fromarray(cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB))

            predictions = batch_text_detection(
                [pil_img], self.det_model, self.det_processor
            )

            page_records = []

            if predictions and predictions[0].bboxes:
                sorted_boxes = sorted(predictions[0].bboxes, key=lambda b: b.bbox[1])

                raw_boxes = [(int(b.bbox[0]), int(b.bbox[1]), int(b.bbox[2]), int(b.bbox[3])) for b in sorted_boxes]

                # ← add this line
                merged_boxes = self._merge_lines_into_paragraphs(raw_boxes, gap_multiplier=0.5)

                for box_idx, box_obj in enumerate(merged_boxes):
                    # FIX: Unpack the tuple directly instead of calling box_obj.bbox
                    x0, y0, x1, y1 = box_obj  

                    px0 = max(0, int(x0) - self.pad)
                    py0 = max(0, int(y0) - self.pad)
                    px1 = min(img_w, int(x1) + self.pad)
                    py1 = min(img_h, int(y1) + self.pad)

                    box_tuple = (px0, py0, px1, py1)
                    crop_arr = page_img[py0:py1, px0:px1].copy()

                    flat_region_stream.append(crop_arr)
                    page_records.append({
                        "region_id": f"line_{box_idx+1:03d}",
                        "bounding_box": box_tuple,
                        "label": "detected_line",
                        "crop_array": crop_arr
                    })

            self.page_audit_map[page_idx] = {
                "source_page_image": page_img,
                "extracted_regions": page_records
            }

        return flat_region_stream
    
    def _merge_lines_into_paragraphs(self, boxes: List[Tuple[int,int,int,int]], gap_multiplier: float = 0.8) -> List[Tuple[int,int,int,int]]:
        """
        Merges line-level bounding boxes into paragraph blocks using 2D spatial proximity.
        Prevents horizontal merging across columns by verifying horizontal alignment.
        """
        if not boxes:
            return []

        import statistics

        # Calculate dynamic thresholds based on text scale
        line_heights = [y1 - y0 for (x0, y0, x1, y1) in boxes]
        median_height = statistics.median(line_heights)
        
        v_threshold = gap_multiplier * median_height
        # Max horizontal gap allowed for indented lines or slight misalignments
        h_threshold = median_height * 1.0 

        paragraphs = []

        for box in boxes:
            x0, y0, x1, y1 = box
            merged = False

            # Look through existing paragraphs to find a spatial match
            for idx, para in enumerate(paragraphs):
                px0, py0, px1, py1 = para

                # 1. Check vertical proximity to the bottom of the paragraph
                v_gap = y0 - py1

                # 2. Check horizontal alignment (Do they belong to the same column?)
                # Calculate horizontal overlap length
                overlap = min(x1, px1) - max(x0, px0)
                has_horizontal_overlap = overlap > 0

                # If they don't overlap, check if the horizontal distance between them is tiny
                h_gap = 0 if has_horizontal_overlap else (x0 - px1 if x0 > px1 else px0 - x1)

                # 3. Decision Matrix: Merge only if vertically close AND structurally in the same column
                if (0 <= v_gap <= v_threshold) and (has_horizontal_overlap or h_gap <= h_threshold):
                    # Expand the paragraph's bounding box boundaries
                    new_px0 = min(px0, x0)
                    new_py0 = min(py0, y0)
                    new_px1 = max(px1, x1)
                    new_py1 = max(py1, y1)
                    
                    paragraphs[idx] = (new_px0, new_py0, new_px1, new_py1)
                    merged = True
                    break  # Found its home, stop looking

            # If the line doesn't fit into any existing paragraph, it seeds a new one
            if not merged:
                paragraphs.append((x0, y0, x1, y1))

        return paragraphs
    

class SuryaLayoutBlockExtractor(BasePreprocessor):
    """
    Uses the Surya SegFormer Layout weights to detect full block-level structures.
    Requires: pip install surya-ocr==0.6.0
    """
    def __init__(self, pad_pixels: int = 4):
        import torch
        # FIX: Both engines use the base detection loading utilities
        from surya.model.detection.model import load_model, load_processor       
        from surya.settings import settings
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️ Booting Surya Layout Analysis Engine on {self.device}...")

        # Load the base text line model (required to feed the layout analyzer)
        self.det_model = load_model()
        self.det_processor = load_processor()
        self.det_model.to(self.device)

        # Load the specific layout block model via settings checkpoint
        self.layout_model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        self.layout_processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        self.layout_model.to(self.device)

        self.pad = pad_pixels
        self.page_audit_map: Dict[int, Dict[str, Any]] = {}

    def run(self, images: List[np.ndarray]) -> List[np.ndarray]:
        from surya.detection import batch_text_detection
        from surya.layout import batch_layout_detection
        from PIL import Image
        import cv2

        flat_region_stream = []
        self.page_audit_map = {}

        for page_idx, page_img in enumerate(images):
            img_h, img_w = page_img.shape[:2]
            pil_img = Image.fromarray(cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB))

            line_predictions = batch_text_detection(
                [pil_img], self.det_model, self.det_processor
            )
            predictions = batch_layout_detection(
                [pil_img], self.layout_model, self.layout_processor, line_predictions
            )

            page_records = []

            if predictions and predictions[0].bboxes:
                layout_blocks = predictions[0].bboxes

                # ─── ADDED: GEOMETRIC COORDINATE SORTING ───
                # Adjust Y_ROW_SNAP based on text density (e.g., 30-50 pixels). 
                # This groups blocks sitting on the same horizontal plane.
                Y_ROW_SNAP = 40  
                
                layout_blocks = sorted(
                    layout_blocks, 
                    key=lambda b: (int(b.bbox[1]) // Y_ROW_SNAP, b.bbox[0])
                )
                # ───────────────────────────────────────────

                for box_idx, block_obj in enumerate(layout_blocks):
                    if block_obj.label in ["PageHeader", "PageFooter", "BlankPage"]:
                        continue

                    x0, y0, x1, y1 = block_obj.bbox

                    px0 = max(0, int(x0) - self.pad)
                    py0 = max(0, int(y0) - self.pad)
                    px1 = min(img_w, int(x1) + self.pad)
                    py1 = min(img_h, int(y1) + self.pad)

                    box_tuple = (px0, py0, px1, py1)
                    crop_arr = page_img[py0:py1, px0:px1].copy()

                    flat_region_stream.append(crop_arr)
                    page_records.append({
                        # These indices will now perfectly match your top-left -> bottom-right sequence
                        "region_id": f"block_{box_idx+1:03d}",
                        "bounding_box": box_tuple,
                        "label": block_obj.label, 
                        "crop_array": crop_arr
                    })

            self.page_audit_map[page_idx] = {
                "source_page_image": page_img,
                "extracted_regions": page_records
            }

        return flat_region_stream