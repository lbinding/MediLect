# Code Reference

This document describes the repository's Python modules and the role of each component in the overall document-processing pipeline.

## Package entry points

### src/packagename/datamodels.py

Defines the Pydantic models used by the VLM-based preprocessors:

- DocumentRotation
  - Stores the result of orientation analysis.
  - Includes a reasoning field plus a validated rotation angle of 0, 90, 180, or 270.

- DocumentComposition
  - Stores the result of composite-spread analysis.
  - Includes a reasoning field and a boolean flag indicating whether the image appears to contain two physical pages.

## Preprocessing modules

### src/packagename/preprocessing/base.py

Contains BasePreprocessor, the abstract base class for all preprocessing steps.

- run(images) must accept a list of NumPy image arrays and return a new list of NumPy image arrays.
- This gives each preprocessor a consistent interface.

### src/packagename/preprocessing/rotation.py

Implements AutoOrientPreprocessor.

Responsibilities:
- Detect page rotation using Tesseract's OCR orientation detection.
- Fall back to a VLM-based orientation check when OCR confidence is low.
- Rotate images by 0, 90, 180, or 270 degrees.

Important behavior:
- The class stores the last detected angle in self.last_detected_angle.
- It uses a deterministic rotation map for image transformation.

### src/packagename/preprocessing/splitting.py

Implements SpreadSplitterPreprocessor.

Responsibilities:
- Detect whether a scanned image is a composite two-page spread.
- If so, split the image into left and right pages using edge analysis along the center strip.

Implementation notes:
- The process uses a VLM prompt plus a fallback computer-vision split algorithm.
- The split is chosen by finding the minimum vertical edge response in the center of the image.

### src/packagename/preprocessing/layout.py

Contains several layout extraction strategies.

#### MacroRegionExtractor
- Decomposes pages into semantic regions using either Florence-2 or OpenCV morphology.
- Returns cropped image regions rather than text.
- Maintains a page_audit_map that records each page's source image and extracted regions.

#### PaddleBoxExtractor
- Uses PaddleOCR's detector to find text boxes.
- Produces line-like crops with padding.

#### SuryaBoxExtractor
- Uses the Surya detection model to locate text lines.
- Produces line-level crops and stores them in page_audit_map.

#### SuryaLayoutBlockExtractor
- Uses Surya's layout model to detect larger structural regions such as paragraphs or blocks.
- Filters out page headers, footers, and blank pages.

## Transcription modules

### src/packagename/transcription/base.py

Defines BaseTranscriber, the abstract interface for transcription systems.

### src/packagename/transcription/mineru.py

Implements MinerUTranscriber.

Responsibilities:
- Accepts a list of full-page image arrays.
- Converts them into a temporary PDF.
- Invokes the MinerU CLI to produce Markdown output.
- Returns the generated Markdown as a list of strings.

### src/packagename/transcription/paddle_vl.py

Implements PaddleVLTranscriber.

Responsibilities:
- Loads the PaddleOCR-VL model and processor.
- Transcribes each crop using a chat-template-style prompt.
- Returns a list of transcription strings aligned with the input crop order.

## Postprocessing modules

### src/packagename/postprocessing/base.py

Defines BasePostprocessor.

### src/packagename/postprocessing/deid.py

Implements HybridDeidentifier.

Responsibilities:
- Remove sensitive information from OCR text.
- First run a RoBERTa NER model to redact named entities that are not dates.
- Then run a second pass with an Ollama-based LLM to catch additional PII patterns.

The method outputs a dictionary keyed by page id containing:
- clean_ocr
- roberta_only
- final_llm_scrubbed

## Utility modules

### src/packagename/utils/image.py

Provides helpers for turning common image-like inputs into OpenCV BGR arrays.

Supported inputs:
- file paths
- Pillow images
- NumPy arrays

## Tests

### tests/test_preprocessors.py
- Demonstrates rotation and splitting preprocessing on image files.
- Outputs a batch of processed images and a JSON audit log.

### tests/test_macro_boxing.py
- Demonstrates layout extraction from PDF pages.
- Saves full-page bounding-box overlays and per-region crops.

### tests/test_transcription.py
- Runs a transcription pipeline over PDF files.
- Writes one markdown output file per page.

### tests/test_deid.py
- Exercises the de-identification postprocessor on OCR-style text.
