import cv2
import json
from pathlib import Path
from medilect.preprocessing import AutoOrientPreprocessor, SpreadSplitterPreprocessor
from medilect.config.settings import COMPOSITE_DATA_DIR, COMPOSITE_OUT_DIR
from medilect.utils.data_loader import UniversalDataLoader

def run_pipeline_test():
    COMPOSITE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    execution_log = []

    # 2. Instantiate the Assembly Line & DataLoader ONCE outside the loop
    print("Spinning up VLM instances and loading Preprocessors...")
    orienter = AutoOrientPreprocessor(min_confidence=4.0, vlm_fallback_model="qwen3-vl:8b")
    splitter = SpreadSplitterPreprocessor(vlm_model="qwen3-vl:8b", canny_thresholds=(50, 150))
    
    pipeline = [orienter, splitter]

    data_loader = UniversalDataLoader(render_dpi=200)
    data_stream = data_loader.load(COMPOSITE_DATA_DIR)

    print("\n" + "="*60)
    print(f"Starting batch processing from: {COMPOSITE_DATA_DIR}")

    # 3. Run the batch
    for document in data_stream:
        # We only want to process images (the dataloader handles extensions)
        if document["data_type"] != "image":
            continue
            
        filename = document["filename"]
        stem = document["stem"]
        page_img_bgr = document["data"]

        # Preserving your 'LS' test filter!
        if 'LS' not in filename:
            continue

        print(f"\n" + "="*50)
        print(f"📸 Processing asset: {filename}")
        
        try:
            # Step A & B: The dataloader already parsed the BGR image, put it on the conveyor belt
            image_stream = [page_img_bgr]

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
                out_name = f"{stem}{page_suffix}.jpg"
                out_path = COMPOSITE_OUT_DIR / out_name
                
                cv2.imwrite(str(out_path), page_img)
                saved_files.append(out_name)

            # Step F: Log the metadata
            execution_log.append({
                'original_file': filename,
                'detected_rotation': getattr(orienter, 'last_detected_angle', 'Unknown'),
                'was_composite_spread': was_split,
                'pages_output': len(image_stream),
                'generated_files': saved_files,
                'status': 'SUCCESS'
            })

        except Exception as e:
            print(f"FAILED on {filename}: {str(e)}")
            execution_log.append({
                'original_file': filename,
                'status': 'FAILED',
                'error': str(e)
            })

    # 4. Dump the final audit log
    log_file = COMPOSITE_OUT_DIR / "preprocessing_audit_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(execution_log, f, indent=4)

    print(f"\nBatch processing finished! Audit log saved to: {log_file}")

if __name__ == "__main__":
    run_pipeline_test()