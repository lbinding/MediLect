# Medical Record Transcription Pipeline

This repository implements a document-processing pipeline for extracting and cleaning text from medical record scans and PDFs. The workflow is designed around four major stages:

1. Preprocessing: rotate pages, detect composite spreads, and isolate meaningful layout regions.
2. Transcription: run OCR/transcription over the prepared image crops.
3. Postprocessing: remove sensitive information and normalize extracted text.
4. Validation: exercise the pipeline with a set of test scripts and output folders.

## Repository layout

- src/packagename/datamodels.py: Pydantic schemas used by the VLM-based preprocessors.
- src/packagename/preprocessing/: geometry and layout pre-processing components.
- src/packagename/transcription/: OCR/transcription adapters for different engines.
- src/packagename/postprocessing/: de-identification and text cleaning logic.
- src/packagename/utils/: shared utility helpers.
- tests/: end-to-end and regression-style test scripts.

## Pipeline overview

A typical processing flow looks like this:

- Load a PDF or image.
- Normalize orientation with AutoOrientPreprocessor.
- Detect and split composite two-page spreads with SpreadSplitterPreprocessor.
- Extract regions or crops using one of the layout extractors.
- Transcribe those crops with a selected engine.
- Optionally postprocess the results for de-identification.

## Core modules

### Preprocessing

The preprocessing package contains components that prepare document images before OCR.

- AutoOrientPreprocessor
  - Corrects page rotation using Tesseract OCR first, then falls back to an Ollama-based VLM prompt when needed.
  - Returns rotated images in the same order as the input.

- SpreadSplitterPreprocessor
  - Detects composite page spreads using a VLM prompt.
  - If a spread is detected, it splits the image into two physical pages based on a center-strip edge analysis.

- BasePreprocessor
  - Defines the common interface used by all preprocessing components.

- Layout-based extractors
  - MacroRegionExtractor: uses either Florence-2 or classical OpenCV morphology to isolate layout regions.
  - PaddleBoxExtractor: uses PaddleOCR detection to extract line-level boxes.
  - SuryaBoxExtractor: uses Surya OCR detection for line-level boxes.
  - SuryaLayoutBlockExtractor: uses Surya layout detection to produce block-level image regions.

### Transcription

The transcription package hosts adapters to external OCR or VLM transcription systems.

- BaseTranscriber
  - Common abstract interface for transcribers.

- MinerUTranscriber
  - Wraps the MinerU CLI and converts images into a temporary PDF for OCR processing.

- PaddleVLTranscriber
  - Uses PaddleOCR-VL with a chat template to transcribe image crops.

### Postprocessing

The postprocessing package is intended for downstream cleanup and privacy protection.

- BasePostprocessor
  - Shared interface for postprocessing modules.

- HybridDeidentifier
  - Removes PII and sensitive identifiers from OCR text.
  - Combines a deterministic RoBERTa-based NER pass with an LLM-based audit pass.

### Utilities

- to_numpy_bgr
  - Normalizes common input types such as image paths, PIL images, and NumPy arrays into OpenCV BGR arrays.

## Tests

The test scripts provide example workflows for running the pipeline on local data.

- tests/test_preprocessors.py: exercises the rotation and spread-splitting pipeline.
- tests/test_macro_boxing.py: demonstrates layout-region extraction on PDF pages.
- tests/test_transcription.py: batches transcription over PDFs and writes markdown output files.
- tests/test_deid.py: exercises the de-identification postprocessor.

## Dependencies

This project relies on a mix of computer vision, OCR, and language-model tooling. Typical dependencies include:

- OpenCV
- NumPy
- Pillow
- PyTorch
- Transformers
- Pydantic
- Ollama
- PaddleOCR / PaddleOCR-VL
- MinerU
- Tesseract
- pypdfium2

## Notes

Several components are implemented as adapters around heavyweight local models and external services. In a production setting, you may want to wrap these behind configuration files or environment variables rather than hard-coding paths and model names.
