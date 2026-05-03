"""Tests for text-layer page context extraction."""

from pathlib import Path

import pymupdf

from latex_tools.extract.base import PageTextBlock
from latex_tools.extract.text_extractor import TextExtractor


def _write_blank_pdf(path: Path, page_count: int) -> None:
    doc = pymupdf.open()
    try:
        for _ in range(page_count):
            doc.new_page()
        doc.save(path)
    finally:
        doc.close()


def test_iter_context_chunks_renders_only_selected_pages(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    _write_blank_pdf(pdf_path, page_count=5)
    rendered_pages = []

    def fake_render(self, page, image_dpi):
        rendered_pages.append(page.number + 1)
        return f"image-{page.number + 1}-{image_dpi}"

    def fake_text_blocks(self, page):
        return [
            PageTextBlock(
                text=f"page-{page.number + 1}",
                bbox=(0, 0, 1, 1),
                font_size=12,
            )
        ]

    monkeypatch.setattr(TextExtractor, "_render_page_base64", fake_render)
    monkeypatch.setattr(TextExtractor, "_extract_page_text_blocks", fake_text_blocks)

    extractor = TextExtractor()
    chunks = list(
        extractor.iter_context_chunks(
            pdf_path,
            pages=[5, 2, 99],
            image_dpi=144,
            include_images=True,
            chunk_size=1,
        )
    )

    assert rendered_pages == [2, 5]
    assert [chunk.chunk_index for chunk in chunks] == [1, 2]
    assert [chunk.total_chunks for chunk in chunks] == [2, 2]
    assert [[page.page_number for page in chunk.pages] for chunk in chunks] == [[2], [5]]
    assert chunks[0].pages[0].image_base64 == "image-2-144"
    assert chunks[1].pages[0].text_blocks[0].text == "page-5"


def test_extract_context_keeps_existing_page_selection_behavior(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _write_blank_pdf(pdf_path, page_count=4)

    context = TextExtractor().extract_context(
        pdf_path,
        pages=[3, 1, 3, 99],
        include_images=False,
    )

    assert context.title == "sample"
    assert [page.page_number for page in context.pages] == [1, 3]
    assert all(page.image_base64 is None for page in context.pages)
