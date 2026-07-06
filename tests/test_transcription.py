import os
import sys
import cv2
import shutil
import numpy as np
from pathlib import Path
import pypdfium2 as pdfium 

# Mount local package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from packagename.transcription.mineru import MinerUTranscriber
from packagename.transcription.paddle_vl import PaddleVLTranscriber
from packagename.preprocessing.layout import SuryaBoxExtractor

def test_full_page_transcription():
    PDF_DATA_DIR = Path(r"C:\Users\lawrence\Desktop\RWL\data")
    ROOT_OUT_DIR = Path(r"C:\Users\lawrence\Desktop\RWL\Github\RWL_medical_record_transcription\tests\Transcription_Outputs")

    # Clean the output directory for a fresh run
    if ROOT_OUT_DIR.exists():
        shutil.rmtree(ROOT_OUT_DIR)
    ROOT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 Initializing Transcription Engine...")
    
    # ---------------------------------------------------------
    # TRANSCRIBER TOGGLE
    # True  -> Uses MinerU (Full page parsing)
    # False -> Uses PaddleVL (Requires Surya crop extraction)
    # ---------------------------------------------------------
    USE_MINERU = False 
    
    if USE_MINERU:
        transcriber = MinerUTranscriber(use_gpu=True)
        layout_engine = None
    else:
        transcriber = PaddleVLTranscriber()
        layout_engine = None  # SuryaBoxExtractor(pad_pixels=4)
    
    for file_path in PDF_DATA_DIR.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != '.pdf':
            continue

        print(f"\n" + "="*60)
        print(f"📖 Ingesting PDF Document: {file_path.name}")
        
        try:
            # Open PDF instantly with PDFium
            pdf = pdfium.PdfDocument(file_path)
            print(f"   ↳ Extracted {len(pdf)} pages via PDFium.")

            # Create a dedicated folder for this document's text files
            doc_folder = ROOT_OUT_DIR / file_path.stem
            doc_folder.mkdir(parents=True, exist_ok=True)

            for page_idx, page in enumerate(pdf):
                # Render at 200 DPI for high-quality text extraction
                pil_img = page.render().to_pil()
                page_img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                print(f"     -> Processing Page {page_idx+1}...")
                
                # --- SMART ROUTING ---
                if layout_engine is None:
                    # MinerU natively digests the entire page
                    processing_batch = [page_img_bgr]
                else:
                    # PaddleVL needs Surya to isolate the text lines first
                    print(f"       ↳ Extracting layout crops with Surya...")
                    processing_batch = layout_engine.run([page_img_bgr])
                
                print(f"       ↳ Transcribing {len(processing_batch)} image segment(s)...")
                
                # Both adapters now use the exact same unified signature
                results = transcriber.run(crops=processing_batch)

                # Combine list of strings into a single markdown string
                final_text = "\n\n".join(results)
                
                # Save to a Markdown file
                out_file = doc_folder / f"Page_{page_idx+1:02d}.md"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(final_text)

            print(f"   └── ✅ Saved all transcriptions to {doc_folder.name}/")

        except Exception as e:
            print(f"❌ Failed processing {file_path.name}: {e}")

    print(f"\n✅ Batch finished! Verify your transcriptions here: {ROOT_OUT_DIR.resolve()}")

if __name__ == "__main__":
    test_full_page_transcription()