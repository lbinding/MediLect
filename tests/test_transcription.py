import os
import sys
import shutil
import numpy as np
from pathlib import Path


# Mount local package AND the new config folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))  # Allows us to import from the root config folder

from config.settings import RAW_DATA_DIR, TRANSCRIPTION_OUT_DIR, configure_tesseract
from medilect.transcription.mineru import MinerUTranscriber
from medilect.transcription.paddle_vl import PaddleVLTranscriber
from medilect.transcription.docTR import DocTRTranscriber
from medilect.preprocessing.layout import SuryaBoxExtractor
from medilect.utils.data_loader import UniversalDataLoader
from medilect.preprocessing.rotation import AutoOrientPreprocessor
from medilect.preprocessing.splitting import SpreadSplitterPreprocessor

# Instantly apply Tesseract configuration to the environment
configure_tesseract()

preprocessors = [
    AutoOrientPreprocessor(),
    SpreadSplitterPreprocessor(),
    # SuryaBoxExtractor(pad_pixels=4)
        ]

transcribers = [
    #MinerUTranscriber(use_gpu=True),   # loaded once, stays loaded for the whole batch
    #PaddleVLTranscriber(use_gpu=True, task="ocr"),
    DocTRTranscriber(use_gpu=True)

]
transcriber = transcribers[0]

def test_full_page_transcription():
    # Clean the output directory for a fresh run
    if TRANSCRIPTION_OUT_DIR.exists():
        shutil.rmtree(TRANSCRIPTION_OUT_DIR)
    TRANSCRIPTION_OUT_DIR.mkdir(parents=True, exist_ok=True)


    # ---------------------------------------------------------
    # DATA INGESTION
    # ---------------------------------------------------------
    data_loader = UniversalDataLoader(render_dpi=200)
    
    # Pull the input directory dynamically from our config file
    data_stream = data_loader.load(RAW_DATA_DIR)

    print("\n" + "="*60)
    print(f"📂 Starting batch processing from: {RAW_DATA_DIR.name}")

    for document in data_stream:
        # We only want to run OCR on image data
        if document["data_type"] != "image":
            continue
            
        stem = document["stem"]
        page_num = document["page_num"]
        total_pages = document["total_pages"]
        page_img_bgr = document["data"]

        print(f"\n📖 Processing {document['filename']} (Page {page_num}/{total_pages})...")
        
        try:
            # Create a dedicated folder for this document's text files
            doc_folder = TRANSCRIPTION_OUT_DIR / stem
            doc_folder.mkdir(parents=True, exist_ok=True)

            # --- EXECUTE PIPELINE ---
            current_crops = [page_img_bgr]
            
            # 1. Run the image through all configured preprocessors sequentially
            for processor in preprocessors:
                print(f"       ↳ Running {processor.__class__.__name__}...")
                current_crops = processor.run(current_crops)
                
            print(f"       ↳ Transcribing {len(current_crops)} image segment(s) with {transcriber.__class__.__name__}...")

            # 2. Final Transcription Stage (Converts Image Crops -> Text Strings)
            results = transcriber.run(crops=current_crops)

            # Combine list of strings into a single markdown string
            final_text = "\n\n".join(results)
            
            # Save to a Markdown file
            out_file = doc_folder / f"Page_{page_num:02d}.md"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(final_text)

            print(f"   └── ✅ Saved transcription to {doc_folder.name}/Page_{page_num:02d}.md")

        except Exception as e:
            print(f"❌ Failed processing {document['filename']} page {page_num}: {e}")

    print(f"\n✅ Batch finished! Verify your transcriptions here: {TRANSCRIPTION_OUT_DIR.resolve()}")

if __name__ == "__main__":
    test_full_page_transcription()