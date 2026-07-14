from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePostprocessor(ABC):
    """Abstract base class for all downstream text post-processors."""
    
    @abstractmethod
    def run(self, raw_pages: Dict[str, str]) -> Dict[str, Any]:
        """
        Accepts a dictionary of raw page texts and returns a processed version.
        """
        pass