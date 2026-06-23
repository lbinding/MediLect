import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union

ImageInput = Union[str, Path, Image.Image, np.ndarray]

def to_numpy_bgr(img_input: ImageInput) -> np.ndarray:
    """Guarantees that whatever the user passes in becomes a standard OpenCV BGR array."""
    if isinstance(img_input, (str, Path)):
        img = cv2.imread(str(img_input))
        if img is None:
            raise FileNotFoundError(f"OpenCV could not resolve image at: {img_input}")
        return img
    elif isinstance(img_input, Image.Image):
        return cv2.cvtColor(np.array(img_input.convert('RGB')), cv2.COLOR_RGB2BGR)
    elif isinstance(img_input, np.ndarray):
        return img_input.copy()
    
    raise TypeError(f"Cannot cast object of type {type(img_input)} to numpy array.")