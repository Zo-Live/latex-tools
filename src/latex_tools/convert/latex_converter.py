"""Convert extracted content to LaTeX document source."""

import re

from typing import List

from ..extract.base import ExtractedContent


class LatexConverter:
    """Converts ExtractedContent to a complete LaTeX document string."""

    def __init__(self, use_ctex: bool = True):
        self.use_ctex = use_ctex

    def convert(self, content: ExtractedContent) -> str:
        lines: List[str] = []
        lines.append("% !TEX program = xelatex")
        docclass = r"\documentclass[UTF8]{ctexart}" if self.use_ctex else r"\documentclass{article}"
        lines.append(docclass)
        lines.append(r"\usepackage{amsmath}")
        lines.append(r"\usepackage{amsthm}")
        lines.append(r"\usepackage{amssymb}")
        lines.append(r"\newtheorem{definition}{定义}")
        lines.append(r"\newtheorem{theorem}{定理}")
        lines.append(r"\newtheorem{lemma}{引理}")
        lines.append(r"\newtheorem{property}{性质}")
        lines.append(r"\newtheorem{corollary}{推论}")
        lines.append(r"\newtheorem{example}{例}")
        lines.append("")
        lines.append(r"\title{" + self._escape_latex(content.title) + "}")
        lines.append(r"\date{\today}")
        lines.append("")
        lines.append(r"\begin{document}")
        lines.append(r"\maketitle")
        lines.append("")

        for block in content.blocks:
            text = self._escape_latex(block.text)

            if block.block_type == "heading":
                lines.append(r"\section{" + text + "}")
                lines.append("")
            elif block.block_type == "definition":
                lines.append(r"\begin{definition}")
                lines.append("  " + text)
                lines.append(r"\end{definition}")
                lines.append("")
            elif block.block_type == "theorem":
                lines.append(r"\begin{theorem}")
                lines.append("  " + text)
                lines.append(r"\end{theorem}")
                lines.append("")
            elif block.block_type == "lemma":
                lines.append(r"\begin{lemma}")
                lines.append("  " + text)
                lines.append(r"\end{lemma}")
                lines.append("")
            elif block.block_type == "property":
                lines.append(r"\begin{property}")
                lines.append("  " + text)
                lines.append(r"\end{property}")
                lines.append("")
            elif block.block_type == "corollary":
                lines.append(r"\begin{corollary}")
                lines.append("  " + text)
                lines.append(r"\end{corollary}")
                lines.append("")
            elif block.block_type == "example":
                lines.append(r"\begin{example}")
                lines.append("  " + text)
                lines.append(r"\end{example}")
                lines.append("")
            elif block.block_type == "proof":
                lines.append(r"\begin{proof}")
                lines.append("  " + text)
                lines.append(r"\end{proof}")
                lines.append("")
            else:
                lines.append(text)
                lines.append("")

        lines.append(r"\end{document}")
        lines.append("")
        return "\n".join(lines)

    def convert_fragments(
        self,
        *,
        title: str,
        fragments: List[str],
        notes: List[str] | None = None,
    ) -> str:
        """Build a complete LaTeX document from trusted body fragments."""
        lines = self._document_header(title)
        for note in notes or []:
            sanitized_note = note.replace("\n", " ").strip()
            if sanitized_note:
                lines.append("% LLM note: " + sanitized_note)
        if notes:
            lines.append("")

        for fragment in fragments:
            cleaned = self._clean_body_fragment(fragment)
            if not cleaned:
                continue
            lines.append(cleaned)
            lines.append("")

        lines.append(r"\end{document}")
        lines.append("")
        return "\n".join(lines)

    def _document_header(self, title: str) -> List[str]:
        docclass = r"\documentclass[UTF8]{ctexart}" if self.use_ctex else r"\documentclass{article}"
        return [
            "% !TEX program = xelatex",
            docclass,
            r"\usepackage{amsmath}",
            r"\usepackage{amsthm}",
            r"\usepackage{amssymb}",
            r"\newtheorem{definition}{定义}",
            r"\newtheorem{theorem}{定理}",
            r"\newtheorem{lemma}{引理}",
            r"\newtheorem{property}{性质}",
            r"\newtheorem{corollary}{推论}",
            r"\newtheorem{example}{例}",
            "",
            r"\title{" + self._escape_latex(title) + "}",
            r"\date{\today}",
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
        ]

    def _clean_body_fragment(self, fragment: str) -> str:
        text = fragment.strip()
        text = re.sub(r"\\documentclass(?:\[[^\]]*\])?\{[^}]+\}", "", text)
        text = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{[^}]+\}", "", text)
        text = text.replace(r"\begin{document}", "")
        text = text.replace(r"\end{document}", "")
        return text.strip()

    def _escape_latex(self, text: str) -> str:
        text = self._strip_invalid_chars(text)
        replacements = [
            ("\\", r"\textbackslash{}"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("$", r"\$"),
            ("&", r"\&"),
            ("#", r"\#"),
            ("^", r"\^{}"),
            ("_", r"\_"),
            ("%", r"\%"),
            ("~", r"\textasciitilde{}"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return self._replace_unicode_math(text)

    def _strip_invalid_chars(self, text: str) -> str:
        return "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 32)

    def _replace_unicode_math(self, text: str) -> str:
        replacements = [
            ("∉", r"\(\notin\)"),
            ("≠", r"\(\ne\)"),
            ("∉", r"\(\notin\)"),
            ("≤", r"\(\le\)"),
            ("≥", r"\(\ge\)"),
            ("≠", r"\(\ne\)"),
            ("∈", r"\(\in\)"),
            ("∅", r"\(\emptyset\)"),
            ("∀", r"\(\forall\)"),
            ("∃", r"\(\exists\)"),
            ("⊆", r"\(\subseteq\)"),
            ("⊂", r"\(\subset\)"),
            ("∩", r"\(\cap\)"),
            ("∪", r"\(\cup\)"),
            ("⇒", r"\(\Rightarrow\)"),
            ("⇐", r"\(\Leftarrow\)"),
            ("⇔", r"\(\Leftrightarrow\)"),
            ("↔", r"\(\leftrightarrow\)"),
            ("→", r"\(\to\)"),
            ("←", r"\(\leftarrow\)"),
            ("∨", r"\(\vee\)"),
            ("∧", r"\(\wedge\)"),
            ("¬", r"\(\neg\)"),
            ("ℵ", r"\(\aleph\)"),
            ("ϵ", r"\(\epsilon\)"),
            ("ε", r"\(\epsilon\)"),
            ("ϕ", r"\(\phi\)"),
            ("φ", r"\(\phi\)"),
            ("α", r"\(\alpha\)"),
            ("β", r"\(\beta\)"),
            ("γ", r"\(\gamma\)"),
            ("δ", r"\(\delta\)"),
            ("η", r"\(\eta\)"),
            ("θ", r"\(\theta\)"),
            ("κ", r"\(\kappa\)"),
            ("λ", r"\(\lambda\)"),
            ("ξ", r"\(\xi\)"),
            ("σ", r"\(\sigma\)"),
            ("τ", r"\(\tau\)"),
            ("ω", r"\(\omega\)"),
            ("Π", r"\(\Pi\)"),
            ("Θ", r"\(\Theta\)"),
            ("′", r"\(^{\prime}\)"),
            ("−", r"\(-\)"),
            ("×", r"\(\times\)"),
            ("±", r"\(\pm\)"),
            ("∗", r"\(\ast\)"),
            ("⋆", r"\(\star\)"),
            ("⊕", r"\(\oplus\)"),
            ("∫", r"\(\int\)"),
            ("∑", r"\(\sum\)"),
            ("√", r"\(\sqrt{\;}\)"),
            ("∼", r"\(\sim\)"),
            ("✓", r"\(\checkmark\)"),
            ("◦", r"\(\circ\)"),
            ("¯", r"\(\overline{\phantom{x}}\)"),
            ("\u0338", ""),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text
