import os
from pathlib import Path

# This dynamically finds the root folder based on this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==============================================================================
# 📂 DIRECTORY CONFIGURATION
# ==============================================================================
# Use raw strings (r"...") so Windows backslashes don't break the code
RAW_DATA_DIR = Path(r"C:\Users\lawrence\Desktop\RWL\data")
TRANSCRIPTION_OUT_DIR = Path(r"C:\Users\lawrence\Desktop\RWL\Github\RWL_medical_record_transcription\tests\Transcription_Outputs")

# For De-identification testing, this is setup is for the de-identification folders
OCR_IN_DIR = Path(r"C:\Users\lawrence\Desktop\RWL\De-identification\models\paddleocr")
GT_BASE_DIR = Path(r"C:\Users\lawrence\Desktop\RWL\De-identification\GT_dataset")

# ==============================================================================
# ⚙️ TESSERACT OCR SETUP
# ==============================================================================
def configure_tesseract():
    """
    Injects the Tesseract paths into PyTesseract and your system environment.
    Call this at the top of any script that needs Tesseract.
    """
    # 1. Set the data prefix environment variable
    os.environ['TESSDATA_PREFIX'] = r'C:\Users\lawrence\miniconda3\envs\deepseek-ocr2\share\tessdata'
    
    # 2. Safely map the executable (fails gracefully if pytesseract isn't in this specific conda env)
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\lawrence\miniconda3\envs\deepseek-ocr2\Library\bin\tesseract.exe'
    except ImportError:
        pass