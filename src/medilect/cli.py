import os
import sys
import argparse
import numpy as np
from pathlib import Path
from medilect.utils.data_loader import UniversalDataLoader
from medilect.utils.debug_logger import DebugLogger
from medilect.transcription.mineru import MinerUTranscriber
from medilect.transcription.docTR import DocTRTranscriber
from medilect.transcription.paddle_vl import PaddleVLTranscriber
from medilect.postprocessing.deid import HybridDeidentifier
from medilect.preprocessing.rotation import AutoOrientPreprocessor
from medilect.preprocessing.splitting import SpreadSplitterPreprocessor

def execute_pipeline(input_path_str: str, out_dir_str: str, pages_str: str, merge_flag: bool, skip_deid_flag: bool, transcriber_type: str, debug_flag: bool):
    """
    Core business logic for the OCR and De-Identification pipeline.
    """
    input_path = Path(input_path_str)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Debug Logger if requested
    debug_logger = DebugLogger(out_dir_str) if debug_flag else None
    page_metadata = {} # Tracks state for the debug CSV
    
    # Parse the target pages if specified
    target_pages = None
    if pages_str:
        try:
            target_pages = set(int(p.strip()) for p in pages_str.split(","))
            print(f"🎯 Targeting specific pages: {target_pages}")
        except ValueError:
            print("❌ Invalid format for --pages. Please use a comma-separated list of numbers (e.g., '1,3,5').")
            sys.exit(1)

    print("\n🚀 Initializing Pipeline Components...")
    data_loader = UniversalDataLoader(render_dpi=200)
    preprocessors = [
        AutoOrientPreprocessor(),
        SpreadSplitterPreprocessor()
    ]
    
    # Dynamically boot the requested transcriber
    if transcriber_type == 'mineru':
        transcriber = MinerUTranscriber(use_gpu=True)
    elif transcriber_type == 'doctr':
        transcriber = DocTRTranscriber(use_gpu=True)
    elif transcriber_type == 'paddleVL':
        transcriber = PaddleVLTranscriber(use_gpu=True, task="ocr")
    else:
        print(f"❌ Unknown transcriber engine requested: {transcriber_type}")
        sys.exit(1)
    
    if not skip_deid_flag:
        deidentifier = HybridDeidentifier()

    corpus_dict = {}

    print(f"\n📂 Ingesting data from: {input_path}")
    data_stream = data_loader.load(input_path)

    for document in data_stream:
        if document["data_type"] != "image":
            continue
            
        page_num = document["page_num"]
        
        if target_pages and page_num not in target_pages:
            continue
            
        stem = document["stem"]
        total_pages = document["total_pages"]
        page_img_bgr = document["data"]
        page_key = f"{stem}_page_{page_num:02d}"
        
        # Initialize debug tracking state for this page
        page_metadata[page_key] = {
            "stem": stem,
            "page": page_num,
            "rotated": False,
            "composite": False,
            "removed_info": []
        }

        print(f"\n📖 Processing {document['filename']} (Page {page_num}/{total_pages})...")
        
        try:
            # 1. Preprocessing Stage (Rotation & Composite Splitting)
            current_crops = [page_img_bgr]
            
            if debug_logger:
                debug_logger.save_step_image(page_img_bgr, stem, page_num, "00_original")

            for processor in preprocessors:
                print(f"       ↳ Running {processor.__class__.__name__}...")
                
                before_img = current_crops[0] if current_crops else None
                current_crops = processor.run(current_crops)
                
                # Metadata checks
                if processor.__class__.__name__ == "AutoOrientPreprocessor":
                    # If the numpy array changed, a rotation occurred
                    if before_img is not None and not np.array_equal(before_img, current_crops[0]):
                        page_metadata[page_key]["rotated"] = True
                        
                elif processor.__class__.__name__ == "SpreadSplitterPreprocessor":
                    if len(current_crops) > 1:
                        page_metadata[page_key]["composite"] = True
                
                if debug_logger:
                    for i, crop in enumerate(current_crops):
                        debug_logger.save_step_image(crop, stem, page_num, f"{processor.__class__.__name__}_{i}")
            
            # 2. Transcription Stage
            print(f"       ↳ Transcribing {len(current_crops)} segment(s) with {transcriber.__class__.__name__}...")
            page_raw_text = ""
            
            for crop in current_crops:
                results = transcriber.run(crops=[crop])                    
                page_raw_text += "\n\n".join(results) + "\n\n"
                
            corpus_dict[page_key] = page_raw_text.strip()
            
        except Exception as e:
            print(f"❌ Failed processing {page_key}: {e}")

    if not corpus_dict:
        print("\n⚠️ No pages were processed. Check your input path and --pages argument.")
        sys.exit(0)

    final_output_dict = {}

    if not skip_deid_flag:
        print("\n🛡️ Running Hybrid De-identification Sweep...")
        audit_results = deidentifier.run(corpus_dict)
        for pk, versions in audit_results.items():
            final_output_dict[pk] = versions["final_llm_scrubbed"]
            
            # Extract removed info for the debug CSV
            if debug_logger and pk in page_metadata:
                # ⚠️ Adjust "redacted_entities" to match whatever key your Deidentifier outputs
                removed = versions.get("redacted_entities", [])
                page_metadata[pk]["removed_info"] = removed
    else:
        print("\n⚠️ Skipping De-identification as requested.")
        final_output_dict = corpus_dict

    # Write all tracked metadata to the debug CSV
    if debug_logger:
        print(f"\n📊 Saving debug log to {debug_logger.csv_path}")
        for pk, meta in page_metadata.items():
            debug_logger.log_metadata(
                filename=meta["stem"],
                page=meta["page"],
                rotated=meta["rotated"],
                composite=meta["composite"],
                removed_info=meta["removed_info"]
            )

    print(f"\n💾 Saving Outputs to {out_dir.resolve()}...")
    
    # Export based on the merge flag
    if merge_flag:
        grouped_docs = {}
        for pk, text in final_output_dict.items():
            doc_stem = pk.rsplit('_page_', 1)[0]
            if doc_stem not in grouped_docs:
                grouped_docs[doc_stem] = []
            grouped_docs[doc_stem].append((pk, text))
            
        for doc_stem, pages in grouped_docs.items():
            merged_filepath = out_dir / f"{doc_stem}_complete.md"
            with open(merged_filepath, "w", encoding="utf-8") as f:
                f.write(f"# Document: {doc_stem}\n\n")
                for pk, text in sorted(pages, key=lambda x: x[0]):
                    f.write(f"## {pk}\n\n{text}\n\n{'='*50}\n\n")
            print(f"  └── ✅ Saved merged document: {merged_filepath.name}")
            
    else:
        for pk, text in final_output_dict.items():
            doc_stem = pk.rsplit('_page_', 1)[0]
            doc_folder = out_dir / doc_stem
            doc_folder.mkdir(exist_ok=True)
            
            page_filepath = doc_folder / f"{pk}.md"
            with open(page_filepath, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  └── ✅ Saved individual page: {doc_folder.name}/{page_filepath.name}")

    print("\n🎉 Pipeline Execution Complete!")

def main():
    parser = argparse.ArgumentParser(description="RWL Medical Record Processing Pipeline")
    parser.add_argument('--in', dest='input_path', help='Path to the target document (PDF/Image) or a directory of files.', required=True)
    parser.add_argument('--out', dest='out_dir', help='Directory where the transcribed and de-identified files will be saved.', default='./output')
    parser.add_argument('--pages', dest='pages', help="Comma-separated list of pages to process (e.g., '1,3,5').", default=None)
    parser.add_argument('--transcriber', dest='transcriber_type', choices=['mineru', 'doctr', 'paddleVL'], default='mineru', help="Transcription model to use (mineru, doctr, or paddleVL).")
    parser.add_argument('--merge', dest='merge', action='store_true', help="Merge all processed pages into a single text file.")
    parser.add_argument('--skip-deid', dest='skip_deid', action='store_true', help="Skip the LLM/RoBERTa de-identification stage.")
    parser.add_argument('--debug', dest='debug', action='store_true', help="Enable debug mode: saves intermediate images and outputs a metadata CSV.")
    args = parser.parse_args()

    execute_pipeline(
        input_path_str=args.input_path,
        out_dir_str=args.out_dir,
        pages_str=args.pages,
        merge_flag=args.merge,
        skip_deid_flag=args.skip_deid,
        transcriber_type=args.transcriber_type,
        debug_flag=args.debug
    )

if __name__ == "__main__":
    main()