import sys
import cv2
import shutil
import numpy as np
from pathlib import Path
import pypdfium2 as pdfium 

# --- THE "LOCAL PACKAGE" PYTHON HACK ---
#   TODO: CHANGE FOR FINAL VERSION - ITS NOT NEEDED
# This forces Python to look in your the src folder 
#PROJECT_ROOT = Path(__file__).resolve().parent.parent
#sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medilect.preprocessing import AutoOrientPreprocessor, SpreadSplitterPreprocessor
from medilect.preprocessing.layout import MacroRegionExtractor, PaddleBoxExtractor, SuryaBoxExtractor, SuryaLayoutBlockExtractor

def test_pdf_deconstruction():
    PDF_DATA_DIR = Path(r"C:\Users\lawrence\Desktop\RWL\data")
    ROOT_OUT_DIR = Path(r"C:\Users\lawrence\Desktop\RWL\Github\RWL_medical_record_transcription\tests\Segmented_Layouts")

    if ROOT_OUT_DIR.exists():
        shutil.rmtree(ROOT_OUT_DIR)
    ROOT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Setup preprocessors 
    preprocessors = [
        #AutoOrientPreprocessor(),
        #SpreadSplitterPreprocessor(),
        #MacroRegionExtractor(use_florence_vlm=True, pad_pixels=8),
        #PaddleBoxExtractor(pad_pixels=4),
        #SuryaBoxExtractor(pad_pixels=4),
        SuryaLayoutBlockExtractor(pad_pixels=4)
    ]
    extractor = preprocessors[-1]

    for file_path in PDF_DATA_DIR.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != '.pdf':
            continue

        print(f"\n" + "="*60)
        print(f"📖 Ingesting PDF Document: {file_path.name}")
        
        try:
            # Open PDF instantly with PDFium
            pdf = pdfium.PdfDocument(file_path)
            print(f"   ↳ Extracted {len(pdf)} pages via PDFium. Rendering to BGR arrays...")

            image_stream = []
            for page in pdf:
                # Render at 200 DPI (200 / 72 PDF points = scale of 2.777)
                pil_img = page.render(scale=1.0).to_pil()
                image_stream.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

            # Push the document through the assembly line
            for step in preprocessors:
                image_stream = step.run(image_stream)

            # Harvest the semantic crops generated on the back end
            for page_idx, audit_data in extractor.page_audit_map.items():
                
                doc_folder = ROOT_OUT_DIR / file_path.stem / f"Page_{page_idx+1:02d}"
                doc_folder.mkdir(parents=True, exist_ok=True)

                source_img = audit_data["source_page_image"]
                regions = audit_data["extracted_regions"]

                # Master visual validation overview
                canvas = source_img.copy()
                for reg in regions:
                    x0, y0, x1, y1 = reg["bounding_box"]
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(canvas, reg["label"], (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                cv2.imwrite(str(doc_folder / "FULL_PAGE_BOUNDED.jpg"), canvas)

                # Micro-Crops
                for reg in regions:
                    cv2.imwrite(str(doc_folder / f"{reg['region_id']}.jpg"), reg["crop_array"])

                print(f"   └── [Page {page_idx+1}] Saved overview + {len(regions)} semantic region crops.")

        except Exception as e:
            print(f"❌ Failed processing {file_path.name}: {e}")

    print(f"\n✅ Batch finished! Verify your crops here: {ROOT_OUT_DIR.resolve()}")

if __name__ == "__main__":
    test_pdf_deconstruction()