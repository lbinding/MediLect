import csv
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

class DebugLogger:
    def __init__(self, output_dir: str):
        # Creates a dedicated debug folder inside your output directory
        self.out_dir = Path(output_dir) / "debug_report"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "pipeline_debug_report.csv"
        
        # Initialize CSV with headers
        with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", "Page", "Rotated", "Composite Image", "De-id Info Removed"])

    def save_step_image(self, image, filename: str, page: int, step_name: str):
        """Saves an intermediate image to the debug folder."""
        img_path = self.out_dir / f"{filename}_page_{page:02d}_{step_name}.jpg"
        
        if isinstance(image, Image.Image):
            image.convert("RGB").save(img_path)
        elif isinstance(image, np.ndarray):
            cv2.imwrite(str(img_path), image)

    def log_metadata(self, filename: str, page: int, rotated: bool, composite: bool, removed_info: list):
        """Appends a single page's metadata to the CSV."""
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Flatten list of entities into a single string for the CSV cell
            removed_str = ", ".join(map(str, removed_info)) if removed_info else "None"
            writer.writerow([filename, page, rotated, composite, removed_str])