import os
import sys
import cv2
import json
from pathlib import Path

# --- THE "LOCAL PACKAGE" PYTHON HACK ---
#   TODO: CHANGE FOR FINAL VERSION - ITS NOT NEEDED
# This forces Python to look in your the src folder 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from packagename.utils.image import to_numpy_bgr
from packagename.preprocessing import AutoOrientPreprocessor, SpreadSplitterPreprocessor

def run_pipeline_test():
    # 1. Define Paths (Preserving your exact Windows paths)
    COMPOSITE_DIR = Path(r"C:\Users\lawrence\Desktop\RWL\Composite_Pages\pages")
    OUT_DIR = Path(r"C:\Users\lawrence\Desktop\RWL\Github\RWL_medical_record_transcription\tests\rotation_pipeline_output")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    execution_log = []

    # 2. Instantiate the Assembly Line ONCE outside the loop
    print("⚙️ Spinning up VLM instances and loading Preprocessors...")
    orienter = AutoOrientPreprocessor(min_confidence=4.0, vlm_fallback_model="qwen3-vl:8b")
    splitter = SpreadSplitterPreprocessor(vlm_model="qwen3-vl:8b", canny_thresholds=(50, 150))
    
    pipeline = [orienter]#, splitter]

    # 3. Run the batch
    for file_path in COMPOSITE_DIR.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
            
        # Preserving your 'LS' test filter!
        if 'LS' not in file_path.name:
            continue

        print(f"\n" + "="*50)
        print(f"📸 Processing asset: {file_path.name}")
        
        try:
            # Step A: Universal Cast
            raw_img = to_numpy_bgr(file_path)
            
            # Step B: Put it on the conveyor belt as a stream of 1
            image_stream = [raw_img]

            # Step C: Push through the modules
            for step in pipeline:
                image_stream = step.run(image_stream)

            # Step D: Inspect the stream results
            was_split = len(image_stream) > 1
            print(f"✅ Stream complete. Total physical pages extracted: {len(image_stream)}")

            # Step E: Save the dynamic stream output
            saved_files = []
            for idx, page_img in enumerate(image_stream):
                # If it split into a spread, name them _page_1, _page_2. If standard, leave stem alone.
                page_suffix = f"_page_{idx+1}" if was_split else ""
                out_name = f"{file_path.stem}{page_suffix}.jpg"
                out_path = OUT_DIR / out_name
                
                cv2.imwrite(str(out_path), page_img)
                saved_files.append(out_name)

            # Step F: Log the metadata
            execution_log.append({
                'original_file': file_path.name,
                'detected_rotation': getattr(orienter, 'last_detected_angle', 'Unknown'),
                'was_composite_spread': was_split,
                'pages_output': len(image_stream),
                'generated_files': saved_files,
                'status': 'SUCCESS'
            })

        except Exception as e:
            print(f"❌ FAILED on {file_path.name}: {str(e)}")
            execution_log.append({
                'original_file': file_path.name,
                'status': 'FAILED',
                'error': str(e)
            })

    # 4. Dump the final audit log
    log_file = OUT_DIR / "preprocessing_audit_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(execution_log, f, indent=4)

    print(f"\n🎉 Batch processing finished! Audit log saved to: {log_file}")

if __name__ == "__main__":
    run_pipeline_test()