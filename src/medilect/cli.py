import os
import sys
import argparse
from pathlib import Path
from medilect.utils.data_loader import UniversalDataLoader
from medilect.transcription.mineru import MinerUTranscriber
from medilect.transcription.docTR import DocTRTranscriber
from medilect.transcription.paddle_vl import PaddleVLTranscriber
from medilect.postprocessing.deid import HybridDeidentifier
from medilect.preprocessing.rotation import AutoOrientPreprocessor
from medilect.preprocessing.splitting import SpreadSplitterPreprocessor

def execute_pipeline(input_path_str: str, out_dir_str: str, pages_str: str, merge_flag: bool, skip_deid_flag: bool, transcriber_type: str):
    """
    Core business logic for the OCR and De-Identification pipeline.
    """
    input_path = Path(input_path_str)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    
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

    # We collect all raw OCR texts in a dictionary to pass to the De-identifier later
    # Format: {"filename_page_01": "raw text...", ...}
    corpus_dict = {}

    print(f"\n📂 Ingesting data from: {input_path}")
    data_stream = data_loader.load(input_path)

    for document in data_stream:
        if document["data_type"] != "image":
            continue
            
        page_num = document["page_num"]
        
        # Check against user's requested pages
        if target_pages and page_num not in target_pages:
            continue
            
        stem = document["stem"]
        total_pages = document["total_pages"]
        page_img_bgr = document["data"]
        
        page_key = f"{stem}_page_{page_num:02d}"
        print(f"\n📖 Processing {document['filename']} (Page {page_num}/{total_pages})...")
        
        try:
            # 1. Preprocessing Stage (Rotation & Composite Splitting)
            current_crops = [page_img_bgr]
            for processor in preprocessors:
                print(f"       ↳ Running {processor.__class__.__name__}...")
                current_crops = processor.run(current_crops)
            
            # 2. Transcription Stage
            print(f"       ↳ Transcribing {len(current_crops)} segment(s)...")
            
            was_split = len(current_crops) > 1
            
            for idx, crop in enumerate(current_crops):
                # Dynamically route arguments based on the engine (MinerU vs Paddle)
                if transcriber.__class__.__name__ == "MinerUTranscriber":
                    results = transcriber.run(crops=[crop], full_page_image=crop)
                else:
                    results = transcriber.run(crops=[crop])
                
                segment_text = "\n\n".join(results).strip()
                
                # If the image was a composite spread, save each half as an independent page file!
                if was_split:
                    split_key = f"{page_key}_split_{idx+1}"
                    corpus_dict[split_key] = segment_text
                else:
                    corpus_dict[page_key] = segment_text
            
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
    else:
        print("\n⚠️ Skipping De-identification as requested.")
        final_output_dict = corpus_dict

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
    # inputs
    parser = argparse.ArgumentParser(description="RWL Medical Record Processing Pipeline")
    parser.add_argument('--in', dest='input_path', help='Path to the target document (PDF/Image) or a directory of files.', required=True)
    parser.add_argument('--out', dest='out_dir', help='Directory where the transcribed and de-identified files will be saved.', default='./output')
    parser.add_argument('--pages', dest='pages', help="Comma-separated list of pages to process (e.g., '1,3,5').", default=None)
    parser.add_argument('--transcriber', dest='transcriber_type', choices=['mineru', 'doctr', 'paddleVL'], default='mineru', help="Transcription model to use (mineru, doctr, or paddleVL).")
    parser.add_argument('--merge', dest='merge', action='store_true', help="Merge all processed pages into a single text file.")
    parser.add_argument('--skip-deid', dest='skip_deid', action='store_true', help="Skip the LLM/RoBERTa de-identification stage.")
    args = parser.parse_args()

    # call function to do correction
    execute_pipeline(
        input_path_str=args.input_path,
        out_dir_str=args.out_dir,
        pages_str=args.pages,
        merge_flag=args.merge,
        skip_deid_flag=args.skip_deid,
        transcriber_type=args.transcriber_type
    )

if __name__ == "__main__":
    main()