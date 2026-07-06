import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Union, Dict, Any, Optional
import pypdfium2 as pdfium

class UniversalDataLoader:
    """
    A robust, dynamic data ingestion utility.
    Accepts a file or directory path, automatically crawls it, and yields standardized 
    data payloads (OpenCV BGR arrays for documents/images, strings for text) 
    that the pipeline can digest.
    """
    
    def __init__(self, render_dpi: int = 200):
        # 200 DPI is the sweet spot for OCR: large enough for clear text, small enough for fast VLM processing
        self.render_dpi = render_dpi
        
        # Define the filetypes this loader knows how to handle
        self.supported_pdfs = {'.pdf'}
        self.supported_images = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}
        self.supported_texts = {'.txt', '.md', '.json', '.csv'}

    def load(self, source_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """
        The main entry point. Routes the user's path (file or directory) 
        and yields unified data dictionaries.
        """
        path = Path(source_path).resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"❌ The provided path does not exist: {path}")

        if path.is_file():
            yield from self._process_file(path)
            
        elif path.is_dir():
            print(f"📂 Crawling directory: {path.name}...")
            # rglob('*') recursively hunts through all nested subfolders
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    yield from self._process_file(file_path)

    def _process_file(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """
        Determines the file type and routes it to the correct specialized loader.
        Safely catches and isolates corrupted files without breaking the entire batch.
        """
        ext = file_path.suffix.lower()
        
        try:
            if ext in self.supported_pdfs:
                yield from self._load_pdf(file_path)
            elif ext in self.supported_images:
                yield from self._load_image(file_path)
            elif ext in self.supported_texts:
                yield from self._load_text(file_path)
            else:
                # Silently ignore unsupported files (like .DS_Store, .exe, etc.)
                pass
                
        except Exception as e:
            print(f"⚠️ Skipping corrupted or unreadable file '{file_path.name}': {e}")

    def _load_pdf(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """
        Extracts pages from a PDF and renders them to OpenCV arrays.
        """
        pdf = pdfium.PdfDocument(file_path)
        total_pages = len(pdf)
        
        for page_idx in range(total_pages):
            page = pdf[page_idx]
            # Convert PDF vector data to a rasterized PIL Image
            pil_img = page.render(scale=self.render_dpi / 72).to_pil()
            
            # Convert PIL RGB to OpenCV BGR array (pipeline standard)
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            yield self._build_payload(
                filepath=file_path,
                data=cv_img,
                data_type="image",
                page_number=page_idx + 1,
                total_pages=total_pages
            )

    def _load_image(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """
        Loads standard images into OpenCV arrays. 
        Uses np.fromfile to safely handle Windows paths with special/unicode characters.
        """
        # Read file bytes securely
        file_bytes = np.fromfile(str(file_path), dtype=np.uint8)
        # Decode into BGR array
        cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if cv_img is None:
            raise ValueError("Image decoder returned None. File may be corrupted.")
            
        yield self._build_payload(
            filepath=file_path,
            data=cv_img,
            data_type="image",
            page_number=1,
            total_pages=1
        )

    def _load_text(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """
        Reads raw text/markdown documents.
        """
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        yield self._build_payload(
            filepath=file_path,
            data=content,
            data_type="text",
            page_number=1,
            total_pages=1
        )

    def _build_payload(self, filepath: Path, data: Any, data_type: str, 
                       page_number: int, total_pages: int) -> Dict[str, Any]:
        """
        Constructs the unified dictionary that gets passed down the pipeline.
        """
        return {
            "source_path": filepath,
            "filename": filepath.name,
            "stem": filepath.stem,
            "extension": filepath.suffix.lower(),
            "data_type": data_type,    # "image" or "text"
            "data": data,              # np.ndarray or str
            "page_num": page_number,
            "total_pages": total_pages
        }