"""
Text segmenter using NLTK for simple sentence tokenization.
"""
import nltk
from typing import List

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


class TextSegmenter:
    """Simple text segmenter using NLTK. Only tokenizes into sentences."""
    
    def __init__(self, language: str = "english"):
        """
        Initialize text segmenter.
        
        Args:
            language: Language for sentence tokenization.
        """
        self.language = language
    
    def segment(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK.
        
        Args:
            text: Input text to segment.
            
        Returns:
            List of sentences as strings.
        """
        return nltk.sent_tokenize(text, language=self.language)
