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
OUT_CSV_PATH = Path(r"C:\Users\lawrence\Desktop\RWL\De-identification\methods\roberta_paddleocr\summary_csv\modular_pipeline_audit.csv")

COMPOSITE_DATA_DIR = Path(r"C:\Users\lawrence\Desktop\RWL\Composite_Pages\pages")
COMPOSITE_OUT_DIR = Path(r"C:\Users\lawrence\Desktop\RWL\Github\RWL_medical_record_transcription\tests\rotation_pipeline_output")

# ==============================================================================
# ⚙️ TESSERACT OCR SETUP
# ==============================================================================
def configure_tesseract():
    """
    You may be asking yourself, why on earth have you designed a function to configure Tesseract? 
    Why not just set the environment variables and paths directly in this file? 
    Well, my friend, let me tell you a tale of woe and frustration.
    Windows and tesseract are an absolute headache to get working, Mac? easy. Linux? easy. Windows? Sorry please spend a full date chasing obscure files. 
    So this function exists because I needed to transverse envrionments. As the primary deployment of this is on linux I have uncommented the windows paths.
    Whomever takes over this project you can probably just delete this function and import pytesseract directly. 
    But for now, this is a helper function to make it easier to set up Tesseract on Windows.

    Apologies for the spaghetti, but this is a necessary evil to get Tesseract working on my Windows.
    """
    # 1. Set the data prefix environment variable
    os.environ['TESSDATA_PREFIX'] = r'C:\Users\lawrence\miniconda3\envs\deepseek-ocr2\share\tessdata'
    
    # 2. Safely map the executable (fails gracefully if pytesseract isn't in this specific conda env)
    #try:
    import pytesseract
    #    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\lawrence\miniconda3\envs\deepseek-ocr2\Library\bin\tesseract.exe'
    #except ImportError:
    #    pass