from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BasePreprocessor(ABC):
    @abstractmethod
    def run(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Accepts a list of BGR image arrays, applies a transformation, 
        and returns a list of BGR image arrays.
        """
        pass