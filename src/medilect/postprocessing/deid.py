import re
import sys
import markdown
from bs4 import BeautifulSoup
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from .base import BasePostprocessor
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from ollama import chat

class PatientDataExtractor(BaseModel):
    contains_pii: bool
    extracted_identifiers: List[str] = Field(default_factory=list)

class HybridDeidentifier(BasePostprocessor):
    """
    Two-stage NLP pipeline for removing PII from OCR text.
    Stage 1: Deterministic Token Classification (RoBERTa i2b2)
    Stage 2: Generative Catch-All Sweep (Ollama LLM)
    """
    def __init__(self, roberta_model: str = "obi/deid_roberta_i2b2", ollama_model: str = "gemma4:e4b"):
        self.roberta_name = roberta_model
        self.llm_name = ollama_model
        
        print(f"⚙️ Booting Stage 1 De-ID Engine: {self.roberta_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.roberta_name)
            model = AutoModelForTokenClassification.from_pretrained(self.roberta_name)
            self.roberta_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        except Exception as e:
            print(f"❌ Critical failure loading RoBERTa: {e}")
            sys.exit(1)

        self.safe_words = {
            "and", "for", "the", "hospital", "hospitals", "on", "clinic",
            "january", "february", "march", "april", "may", "june", "july", 
            "august", "september", "october", "november", "december"
        }

    def run(self, raw_pages: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """
        Accepts: {'page_0000': '# Patient Name: Clive...', 'page_0001': ...}
        Returns: {'page_0000': {'clean_ocr': '...', 'roberta_only': '...', 'final_llm_scrubbed': '...'}}
        """
        audit_trail = {}

        for page_id, raw_md in raw_pages.items():
            print(f"Scrubbing {page_id}...")
            
            # Step 1: Strip Markdown formatting down to clean prose
            clean_text = self._strip_markdown(raw_md)
            if not clean_text.strip():
                audit_trail[page_id] = {
                    "clean_ocr": "", 
                    "roberta_only": "", 
                    "final_llm_scrubbed": ""
                }
                continue

            # Step 2: Deterministic RoBERTa pass (Date-Preserving)
            roberta_text = self._stage1_roberta_scrub(clean_text)

            # Step 3: Generative LLM Auditor Sweep
            final_scrubbed_text = self._stage2_llm_audit_scrub(clean_text, roberta_text)

            audit_trail[page_id] = {
                "clean_ocr": clean_text,
                "roberta_only": roberta_text,
                "final_llm_scrubbed": final_scrubbed_text
            }

        return audit_trail

    def _strip_markdown(self, md_text: str) -> str:
        html = markdown.markdown(md_text, extensions=['tables'])
        soup = BeautifulSoup(html, "html.parser")
        clean_text = soup.get_text(separator=' ', strip=True)
        return clean_text.replace('\n', ' ')

    def _stage1_roberta_scrub(self, text: str) -> str:
        entities = []
        offset = 0
        max_chars = 1500
        
        while offset < len(text):
            chunk_end = min(offset + max_chars, len(text))
            
            if chunk_end < len(text):
                last_space = text.rfind(' ', offset, chunk_end)
                if last_space != -1:
                    chunk_end = last_space
            
            chunk = text[offset:chunk_end]
            chunk_entities = self.roberta_ner(chunk)
            
            for ent in chunk_entities:
                label = ent.get('entity_group', ent.get('entity', ''))
                if 'DATE' in label.upper():
                    continue
                ent['start'] += offset
                ent['end'] += offset
                entities.append(ent)
                
            offset = chunk_end

        scrubbed = text
        for ent in sorted(entities, key=lambda x: x['start'], reverse=True):
            start = ent['start']
            end = ent['end']
            scrubbed = scrubbed[:start] + "***" + scrubbed[end:]
            
        return scrubbed

    def _stage2_llm_audit_scrub(self, original_clean_text: str, current_scrubbed_text: str) -> str:
        PROMPT_REGISTRY = [
            (
                "UK Postcodes & Addresses", 
                "GDPR UK ADDRESS and POSTCODE data-identification pipeline. Look specifically for: addresses, postcodes."
            ),
            (
                "Hospital Names", 
                "GDPR HOSPITAL name data-identification pipeline. Look specifically for: hospital names and return the full name of the hospital."
            ),
            (
                "Patient Names & DOBs", 
                "GDPR patient-name and date of birth pipeline. Look specifically for full names, and birthdates (NOT hospital visit dates)."
            )
        ]

        chunks = self._get_overlapping_chunks(original_clean_text, chunk_size=1500, overlap=150)
        harvested_pii = []
        
        # Bypass markdown parser issues by dynamically generating the triple backticks
        md_fence = chr(96) * 3

        for chunk_idx, chunk in enumerate(chunks):
            for task_title, task_instructions in PROMPT_REGISTRY:
                try:
                    response = chat(
                        model=self.llm_name,
                        messages=[
                            {
                                'role': 'system', 
                                'content': (
                                    f"You are a secure {task_instructions}\n"
                                    "CRITICAL RULES:\n"
                                    "1. If a value is already redacted with stars (e.g. 'Name: ***'), DO NOT extract it.\n"
                                    f"2. Output strictly raw JSON wrapped in {md_fence}json blocks matching this schema:\n"
                                    f"{md_fence}json\n{{\n  \"contains_pii\": true,\n  \"extracted_identifiers\": [\"...\"]\n}}\n{md_fence}"
                                )
                            },
                            {
                                'role': 'user', 
                                'content': f"Analyze this text snippet:\n\n{chunk}"
                            }
                        ],
                        options={
                            'temperature': 0.0, 
                            'num_ctx': 5120, 
                            'num_predict': 1536
                        }
                    )
                    
                    raw_content = response.message.content
                    if not raw_content or not raw_content.strip():
                        continue

                    # Using the dynamic md_fence variable so it doesn't break the IDE syntax parser
                    json_regex = rf"{md_fence}json\s*(.*?)\s*{md_fence}"
                    json_match = re.search(json_regex, raw_content, re.DOTALL)
                    
                    if json_match:
                        json_string = json_match.group(1).strip()
                    else:
                        json_string = raw_content[raw_content.find('{') : raw_content.rfind('}') + 1].strip()
                    
                    if not json_string:
                        continue

                    cleaned_data = PatientDataExtractor.model_validate_json(json_string)
                    harvested_pii.extend(cleaned_data.extracted_identifiers)

                except Exception as e:
                    # Silently skip misformatted JSON outputs from individual LLM passes
                    continue 

        unique_pii = list(set(harvested_pii))
        active_text = current_scrubbed_text

        for pii_snippet in unique_pii:
            for word in pii_snippet.split():
                w_clean = word.strip()
                if w_clean.isdigit():
                    continue
                
                if len(w_clean) > 2 and w_clean != "***" and w_clean.lower() not in self.safe_words:
                    escaped_word = re.escape(w_clean)
                    pattern = re.compile(rf"\b{escaped_word}\b", re.IGNORECASE)
                    active_text = pattern.sub("***", active_text)

        return active_text

    def _get_overlapping_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            if end < text_length:
                while end > start and text[end] not in [' ', '\n', '.', ',']:
                    end -= 1
                if end == start:
                    end = start + chunk_size
            chunks.append(text[start:end].strip())
            start = end - overlap

        return chunks