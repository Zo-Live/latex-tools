"""Tests for LaTeX document assembly."""

from latex_tools.convert.latex_converter import LatexConverter


def test_convert_fragments_strips_document_wrappers():
    converter = LatexConverter()
    latex = converter.convert_fragments(
        title="讲义",
        fragments=[
            r"\documentclass{article}\begin{document}\section{集合}\end{document}",
            r"\begin{definition}定义内容\end{definition}",
        ],
        notes=["removed header/footer"],
    )

    assert latex.count(r"\documentclass") == 1
    assert r"\section{集合}" in latex
    assert r"\begin{definition}定义内容\end{definition}" in latex
    assert "% LLM note: removed header/footer" in latex
