import os
import glob
import json
import re
import difflib
import sys
from pathlib import Path
import pandas as pd
from medilect.config.settings import OCR_IN_DIR, GT_BASE_DIR, OUT_CSV_PATH
from medilect.utils.data_loader import UniversalDataLoader
from medilect.postprocessing.deid import HybridDeidentifier

def run_deid_validation_test():
    # Ensure output directory exists
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Harvest raw OCR Markdown files into memory using DataLoader
    print(f"📂 Loading OCR results from: {OCR_IN_DIR}")
    data_loader = UniversalDataLoader()
    data_stream = data_loader.load(OCR_IN_DIR)
    
    raw_input_dict = {}
    for document in data_stream:
        # Filter for text-based outputs
        if document.get("data_type") not in ["text", "markdown"]:
            continue
            
        # Safely extract the full path to get the parent folder (e.g., 'page_01')
        file_path = Path(document.get("filepath", document.get("path", "")))
        
        # Preserve your specific target filter
        if file_path.name != "paddleocr.txt":
            continue
            
        page_key = file_path.parent.name
        raw_input_dict[page_key] = document["data"]

    if not raw_input_dict:
        print(f"❌ No 'paddleocr.txt' files found in: {OCR_IN_DIR}")
        return

    # 2. Run the modular De-ID engine
    print(f"⚙️ Running Hybrid De-identifier on {len(raw_input_dict)} pages...")
    deidentifier = HybridDeidentifier()
    processed_corpus = deidentifier.run(raw_input_dict)

    # 3. Calculate metrics against Ground Truth
    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_tn = 0
    missed_records = []

    for page_key, text_versions in processed_corpus.items():
        orig_clean = text_versions["clean_ocr"].lower()
        final_scrubbed = text_versions["final_llm_scrubbed"].lower()

        gt_path = Path(GT_BASE_DIR) / page_key / "identifiable_info.json"
        if not gt_path.exists():
            print(f"⚠️ No ground truth found for {page_key}. Excluded from scoring.")
            continue

        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_json = json.load(f)

        # Safely parse Ground Truth targets
        raw_gt_terms = []
        if isinstance(gt_json, dict):
            if "ID" in gt_json:
                val = gt_json["ID"]
                if isinstance(val, list):
                    raw_gt_terms.extend([str(v) for v in val])
                else:
                    raw_gt_terms.append(str(val))
            else:
                for k, v in gt_json.items():
                    if isinstance(v, str): raw_gt_terms.append(v)
        elif isinstance(gt_json, list):
            for item in gt_json:
                if isinstance(item, dict) and "ID" in item:
                    raw_gt_terms.append(str(item["ID"]))

        # Map GT targets to their actual OCR manifestations
        valid_ocr_targets = set()
        for term in raw_gt_terms:
            actual_spellings = _find_ocr_manifestations(term, orig_clean)
            valid_ocr_targets.update(actual_spellings)

        # A. Score True Positives / False Negatives
        for target_phrase in valid_ocr_targets:
            if target_phrase not in final_scrubbed:
                total_tp += 1
            else:
                total_fn += 1
                
                # Grab surrounding context for the report
                s_idx = final_scrubbed.find(target_phrase)
                c_start = max(0, s_idx - 35)
                c_end = min(len(final_scrubbed), s_idx + len(target_phrase) + 35)
                snippet = final_scrubbed[c_start:c_end].replace('\n', ' ')

                missed_records.append({
                    "Page": page_key,
                    "Leaked_PHI": target_phrase,
                    "Context_Surrounding_Leak": f"...{snippet}..."
                })

        # B. Score Token-Level Over-redaction (FP / TN)
        orig_words = set(re.findall(r'\b\w+\b', orig_clean))
        scrub_words = set(re.findall(r'\b\w+\b', final_scrubbed))
        redacted_words = orig_words - scrub_words

        expected_phi_words = set()
        for t in valid_ocr_targets:
            expected_phi_words.update(re.findall(r'\b\w+\b', t))

        fp_words = redacted_words - expected_phi_words
        total_fp += len(fp_words)

        tn_count = len(orig_words) - len(expected_phi_words) - len(fp_words)
        total_tn += max(0, tn_count)

    # 4. Export scorecard
    sensitivity = (total_tp / (total_tp + total_fn)) * 100 if (total_tp + total_fn) > 0 else 0.0
    specificity = (total_tn / (total_tn + total_fp)) * 100 if (total_tn + total_fp) > 0 else 0.0

    print("\n" + "="*50)
    print("🚀 FINAL AUDIT SCORECARD")
    print("="*50)
    print(f"True Positives (Caught) : {total_tp}")
    print(f"False Negatives (Missed): {total_fn}")
    print(f"False Positives (Extra) : {total_fp} words")
    print(f"True Negatives (Safe)   : {total_tn} words")
    print("-" * 50)
    print(f"Sensitivity (Recall)    : {sensitivity:.2f}%")
    print(f"Specificity             : {specificity:.2f}%")
    print("="*50)

    df_out = pd.DataFrame(missed_records)
    df_out.to_csv(OUT_CSV_PATH, index=False)
    print(f"💾 False Negative Audit saved to:\n   {OUT_CSV_PATH}")

def _find_ocr_manifestations(term: str, text: str) -> list:
    t = term.lower()
    found = set()
    if t in text: found.add(t)
    words = re.findall(r'\b\w+\b', text)
    if len(t.split()) == 1:
        found.update(difflib.get_close_matches(t, words, n=5, cutoff=0.85))
    else:
        tw = t.split()
        ws = len(tw)
        for i in range(len(words) - ws + 1):
            phrase = " ".join(words[i:i+ws])
            if difflib.SequenceMatcher(None, t, phrase).ratio() >= 0.85:
                found.add(phrase)
    return list(found)

if __name__ == "__main__":
    run_deid_validation_test()