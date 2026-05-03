"""Tests for the LLM PDF conversion pipeline."""

from pathlib import Path

from latex_tools.extract.base import PageTextBlock, PdfPageContext
from latex_tools.llm.pipeline import LLMPdfConverter, _append_tail, _tail


class FakeExtractor:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def extract_context(self, pdf_path, pages=None, image_dpi=160, include_images=True):
        self.calls.append(
            {
                "pdf_path": pdf_path,
                "pages": pages,
                "image_dpi": image_dpi,
                "include_images": include_images,
            }
        )
        return type(
            "Context",
            (),
            {
                "title": pdf_path.stem,
                "pages": self.pages,
            },
        )()


class FakeClient:
    def __init__(self, latex_fragments=None):
        self.calls = []
        self.latex_fragments = latex_fragments

    def generate_latex_chunk(
        self,
        *,
        document_title,
        pages,
        chunk_index,
        total_chunks,
        previous_latex_tail="",
        extra_prompt="",
    ):
        self.calls.append(
            {
                "document_title": document_title,
                "pages": [page.page_number for page in pages],
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "previous_latex_tail": previous_latex_tail,
                "extra_prompt": extra_prompt,
            }
        )
        latex = (
            self.latex_fragments[chunk_index - 1]
            if self.latex_fragments is not None
            else f"\\section{{Chunk {chunk_index}}}"
        )
        return type(
            "Result",
            (),
            {
                "latex": latex,
                "notes": [f"chunk-{chunk_index}"],
            },
        )()


def test_pipeline_chunks_pages_and_builds_document():
    pages = [
        PdfPageContext(
            page_number=1,
            width=1,
            height=1,
            text_blocks=[
                PageTextBlock(
                    text="第一页",
                    bbox=(0, 0, 1, 1),
                    font_size=12,
                )
            ],
            image_base64="aGVsbG8=",
        ),
        PdfPageContext(
            page_number=2,
            width=1,
            height=1,
            text_blocks=[
                PageTextBlock(
                    text="第二页",
                    bbox=(0, 0, 1, 1),
                    font_size=12,
                )
            ],
            image_base64="aGVsbG8=",
        ),
        PdfPageContext(
            page_number=3,
            width=1,
            height=1,
            text_blocks=[
                PageTextBlock(
                    text="第三页",
                    bbox=(0, 0, 1, 1),
                    font_size=12,
                )
            ],
            image_base64="aGVsbG8=",
        ),
    ]
    extractor = FakeExtractor(pages)
    client = FakeClient()
    converter = LLMPdfConverter(client, extractor=extractor, chunk_pages=2, image_dpi=144)

    result = converter.convert(Path("docs/sample.pdf"))

    assert "\\documentclass[UTF8]{ctexart}" in result.latex
    assert "\\section{Chunk 1}" in result.latex
    assert "\\section{Chunk 2}" in result.latex
    assert result.notes == ["chunk-1", "chunk-2"]
    assert extractor.calls[0]["image_dpi"] == 144
    assert client.calls[0]["pages"] == [1, 2]
    assert client.calls[1]["pages"] == [3]
    assert client.calls[1]["previous_latex_tail"]


def test_append_tail_matches_tail_of_joined_fragments():
    fragments = [
        "a" * 8,
        "b" * 8,
        "c" * 8,
    ]
    previous_latex_tail = ""

    for index, fragment in enumerate(fragments):
        previous_latex_tail = _append_tail(
            previous_latex_tail,
            fragment,
            has_previous_fragment=index > 0,
            max_chars=12,
        )
        assert previous_latex_tail == _tail("\n\n".join(fragments[: index + 1]), 12)


def test_pipeline_passes_incremental_tail_to_later_chunks():
    pages = [
        PdfPageContext(page_number=1, width=1, height=1),
        PdfPageContext(page_number=2, width=1, height=1),
        PdfPageContext(page_number=3, width=1, height=1),
    ]
    fragments = [
        "first-fragment",
        "second-fragment",
        "third-fragment",
    ]
    extractor = FakeExtractor(pages)
    client = FakeClient(latex_fragments=fragments)
    converter = LLMPdfConverter(client, extractor=extractor, chunk_pages=1)

    converter.convert(Path("docs/sample.pdf"))

    assert client.calls[0]["previous_latex_tail"] == ""
    assert client.calls[1]["previous_latex_tail"] == _tail(fragments[0])
    assert client.calls[2]["previous_latex_tail"] == _tail("\n\n".join(fragments[:2]))
