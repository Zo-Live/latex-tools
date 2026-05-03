"""Text-layer PDF extractor using pymupdf."""

import base64
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import unicodedata

import pymupdf

from .base import (
    BaseExtractor,
    ContentBlock,
    ExtractedContent,
    PageTextBlock,
    PdfDocumentContext,
    PdfPageContext,
)


class TextExtractor(BaseExtractor):
    """Extracts content from the text layer of a PDF.

    Uses pymupdf to read text blocks with font size metadata.
    Section headers are identified by font size deltas.
    """

    def extract(self, pdf_path: Path) -> ExtractedContent:
        doc = pymupdf.open(pdf_path)
        title = pdf_path.stem
        blocks: List[ContentBlock] = []

        for page in doc:
            for text_block in self._extract_page_text_blocks(page):
                blocks.append(
                    ContentBlock(
                        text=text_block.text,
                        block_type=text_block.block_type,
                        level=1 if text_block.block_type == "heading" else 0,
                        page_number=page.number + 1,
                        bbox=text_block.bbox,
                        font_size=text_block.font_size,
                    )
                )

        doc.close()
        return ExtractedContent(source_file=pdf_path, title=title, blocks=blocks)

    def extract_context(
        self,
        pdf_path: Path,
        pages: Optional[Sequence[int]] = None,
        image_dpi: int = 160,
        include_images: bool = True,
    ) -> PdfDocumentContext:
        """Extract page-level context for the LLM conversion pipeline.

        Page numbers are 1-based. If ``pages`` is not provided, every PDF page is
        extracted.
        """
        doc = pymupdf.open(pdf_path)
        wanted_pages = set(pages) if pages is not None else None
        page_contexts: List[PdfPageContext] = []

        for page in doc:
            page_number = page.number + 1
            if wanted_pages is not None and page_number not in wanted_pages:
                continue

            image_base64 = None
            if include_images:
                image_base64 = self._render_page_base64(page, image_dpi)

            rect = page.rect
            page_contexts.append(
                PdfPageContext(
                    page_number=page_number,
                    width=rect.width,
                    height=rect.height,
                    text_blocks=self._extract_page_text_blocks(page),
                    image_base64=image_base64,
                )
            )

        doc.close()
        return PdfDocumentContext(
            source_file=pdf_path,
            title=pdf_path.stem,
            pages=page_contexts,
        )

    def _extract_page_text_blocks(self, page: pymupdf.Page) -> List[PageTextBlock]:
        text_dict = page.get_text("dict")
        blocks: List[PageTextBlock] = []

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = self._join_spans(spans)
                if not text:
                    continue
                font_size = self._max_font_size(spans)
                block_type = self._classify_block(text, font_size)
                bbox = tuple(float(value) for value in line.get("bbox", (0, 0, 0, 0)))
                blocks.append(
                    PageTextBlock(
                        text=text,
                        bbox=bbox,
                        font_size=font_size,
                        block_type=block_type,
                    )
                )

        return blocks

    def _join_spans(self, spans: Iterable[dict]) -> str:
        parts = [self._clean_text(span.get("text", "")) for span in spans]
        text = "".join(parts)
        return " ".join(text.split())

    def _max_font_size(self, spans: Iterable[dict]) -> float:
        sizes = [float(span.get("size", 12)) for span in spans]
        return max(sizes, default=12.0)

    def _render_page_base64(self, page: pymupdf.Page, image_dpi: int) -> str:
        zoom = image_dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return base64.b64encode(pixmap.tobytes("png")).decode("ascii")

    def _clean_text(self, text: str) -> str:
        return "".join(
            ch
            for ch in text
            if ch in "\t\n\r" or not unicodedata.category(ch).startswith("C")
        )

    def _classify_block(self, text: str, font_size: float) -> str:
        heading_keywords = ("定义", "定理", "证明", "例", "性质", "推论", "引理")

        if font_size > 14:
            return "heading"

        for kw in heading_keywords:
            if text.startswith(kw):
                return self._map_keyword_to_type(kw)

        return "text"

    def _map_keyword_to_type(self, keyword: str) -> str:
        mapping = {
            "定义": "definition",
            "定理": "theorem",
            "证明": "proof",
            "例": "example",
            "性质": "property",
            "推论": "corollary",
            "引理": "lemma",
        }
        return mapping.get(keyword, "text")
