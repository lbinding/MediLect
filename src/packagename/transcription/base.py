from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseTranscriber(ABC):
    """
    Abstract base class for all optical character recognition/transcription engines.
    """
    
    @abstractmethod
    def run(self, crops: List[np.ndarray]) -> List[str]:
        """
        Accepts a list of OpenCV image crops (from the layout/geometry engine)
        and returns a list of transcribed strings in the exact same order.
        """
        pass