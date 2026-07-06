import tempfile
import subprocess
import cv2
import numpy as np
from pathlib import Path
from typing import List
from PIL import Image

class MinerUTranscriber:
    """
    Adapter for MinerU v3.x. 
    Accepts raw OpenCV image arrays and uses a temporary Ghost Buffer 
    to interface with MinerU's file-based CLI engine.
    """
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        print("⚙️ Booting MinerU (Adapter Mode)...")

    def run(self, images: List[np.ndarray]) -> List[str]:
        """
        Accepts a list of full-page OpenCV BGR arrays.
        Returns a list containing the parsed Markdown strings.
        """
        if not images:
            return []

        # 1. Convert OpenCV BGR arrays to PIL RGB Images
        pil_images = []
        for img in images:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb_img))

        # 2. Create a secure, self-destructing Temporary Directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_pdf_path = temp_dir_path / "ghost_document.pdf"
            output_dir = temp_dir_path / "output"

            # Instantly compile the image arrays into a temporary multi-page PDF
            pil_images[0].save(
                temp_pdf_path, 
                save_all=True, 
                append_images=pil_images[1:], 
                resolution=200.0
            )

            # 3. Execute the MinerU engine on the Ghost File
            command = [
                "mineru",
                "-p", str(temp_pdf_path),
                "-o", str(output_dir)
            ]

            try:
                # Run the CLI engine and capture any internal errors
                subprocess.run(command, check=True, capture_output=True, text=True)
                
                # MinerU creates a subfolder named after the PDF stem ("ghost_document")
                mineru_out_folder = output_dir / "ghost_document"
                md_files = list(mineru_out_folder.glob("*.md"))
                
                if md_files:
                    with open(md_files[0], 'r', encoding='utf-8') as f:
                        # Return the full markdown document as a single-item list 
                        # to match the unified List[str] return type
                        return [f.read()]
                else:
                    print("⚠️ MinerU finished, but no markdown was found.")
                    return []
                    
            except subprocess.CalledProcessError as e:
                print(f"❌ MinerU Engine Error:\n{e.stderr}")
                return []