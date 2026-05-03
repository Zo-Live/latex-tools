"""Base classes for PDF extraction pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ContentBlock:
    """A single block of extracted content."""

    text: str
    block_type: str = "text"  # text, heading, definition, theorem, proof, example
    level: int = 0  # 0 = body, 1 = section, 2 = subsection
    page_number: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    font_size: Optional[float] = None


@dataclass
class ExtractedContent:
    """Complete content extracted from one PDF."""

    source_file: Path
    title: str
    blocks: List[ContentBlock] = field(default_factory=list)


@dataclass
class PageTextBlock:
    """A positioned text block on one PDF page."""

    text: str
    bbox: Tuple[float, float, float, float]
    font_size: float
    block_type: str = "text"


@dataclass
class PdfPageContext:
    """All LLM-facing context for one PDF page."""

    page_number: int
    width: float
    height: float
    text_blocks: List[PageTextBlock] = field(default_factory=list)
    image_base64: Optional[str] = None
    image_mime_type: str = "image/png"

    @property
    def plain_text(self) -> str:
        return "\n".join(block.text for block in self.text_blocks if block.text)


@dataclass
class PdfDocumentContext:
    """Page-level PDF context used by the LLM conversion pipeline."""

    source_file: Path
    title: str
    pages: List[PdfPageContext] = field(default_factory=list)


class BaseExtractor(ABC):
    """Abstract base for all PDF extractors.

    Subclasses implement different extraction strategies:
    - TextExtractor: reads text layer via pymupdf
    - (future) ImageExtractor, FormulaExtractor
    """

    @abstractmethod
    def extract(self, pdf_path: Path) -> ExtractedContent:
        """Extract structured content from a PDF file."""
        ...
