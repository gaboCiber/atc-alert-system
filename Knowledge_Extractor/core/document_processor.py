"""
Document processor for extracting text from PDFs.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")


@dataclass
class Page:
    """Represents a page extracted from a PDF."""
    number: int
    text: str
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentProcessor:
    """Extract text from PDF documents with support for custom margins."""
    
    def __init__(self, margins: Optional[Tuple[float, float, float, float]] = None):
        """
        Initialize document processor.
        
        Args:
            margins: Optional tuple of (left, bottom, right, top) in points.
                    If None, extracts full page text.
        """
        self.margins = margins
    
    def extract_text(self, pdf_path: str) -> List[Page]:
        """
        Extract text from all pages of a PDF.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of Page objects with extracted text.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        pages = []
        
        for i in range(doc.page_count):
            page = doc[i]
            text = self._extract_page_text(page)
            pages.append(Page(
                number=i + 1,
                text=text,
                metadata={
                    "width": page.rect.width,
                    "height": page.rect.height,
                }
            ))
        
        doc.close()
        return pages
    
    def _extract_page_text(self, page: fitz.Page) -> str:
        """Extract text from a single page, applying margins if configured."""
        if self.margins is None:
            return page.get_text()
        
        left, bottom, right, top = self.margins
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Convert margins to PyMuPDF coordinates (from top-left)
        x0 = left
        y0 = top
        x1 = page_width - right
        y1 = page_height - bottom
        
        clip_rect = fitz.Rect(x0, y0, x1, y1)
        return page.get_text(clip=clip_rect)
