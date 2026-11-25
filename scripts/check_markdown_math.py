"""Lint tracked Markdown files for GitHub GFM and MkDocs MathJax pitfalls.

Checks every real rendering failure mode shared by GitHub Flavored Markdown
(cmark-gfm + GitHub MathJax) and MkDocs Material (pymdownx.arithmatex + MathJax 3):

- unsupported-delimiter: ``\\(...\\)`` or ``\\[...\\]`` outside code blocks.
- disallowed-mathjax-macro: macros rejected by GitHub's restricted MathJax
  (e.g. ``\\operatorname``, ``\\DeclareMathOperator``, ``\\newcommand``).
- gfm-display-math-escape: punctuation backslash escapes inside ``$$`` display
  blocks (``\\{``, ``\\}``, ``\\|``, ``\\,``, ``\\;``, ``\\!``, ``\\_``) that
  GitHub's cmark-gfm unescapes before MathJax runs.
- gfm-bare-math-hazard: bare inline ``$...$`` spans vulnerable to GitHub's
  cmark-gfm emphasis/punctuation/backslash-unescape traps (use ``$`...`$``,
  which GitHub protects and ``scripts/build_docs_site.py`` converts for MkDocs).
- indented-display-math: ``$$`` indented, nested in lists, or over-indented in
  blockquotes.
- inline-display-math: ``$$`` mixed with prose on the same line or non-empty
  opening/closing fence lines.
- display-math-blank-lines: ``$$`` block lacking surrounding blank (or blank
  blockquote) lines.
- double-backslash-in-inline-math: ``\\\\`` inside bare inline ``$...$`` or
  single-line ``$$...$$`` (collapsed by GFM before MathJax runs).
- table-pipe-in-math: ``|`` or ``\\|`` inside inline ``$...$`` or ``$`...`$``
  on Markdown table rows (splits table cells before MathJax runs).
- space-padded-inline-math: leading or trailing spaces inside ``$...$``.
- unclosed-display-math: ``$$`` display block reaching EOF without closing.
- unmatched-dollar: unescaped single ``$`` that is not currency or a shell var.
- raw-latex-outside-math: raw LaTeX math commands outside math and code.
- emphasis-trap-in-inline-math: GFM underscore/bold/text-mode traps inside
  inline ``$...$`` (unescaped ``_`` inside ``\\text{}``, bare ``\\_`` outside
  text macros, ``__`` / ``**``, or space-flanked ``_..._``).
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

FENCE_OPEN_RE = re.compile(r"^(\s*(?:>\s*)*)(`{3,}|~{3,})(.*)$")
BQ_PREFIX_RE = re.compile(r"^((?: {0,3}> ?)+)")
BLANK_BQ_RE = re.compile(r"^\s*(?:>\s*)+$")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+$")
TABLE_SEP_RE = re.compile(
    r"^\s*(?:>\s*)*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
TABLE_SINGLE_SEP_RE = re.compile(r"^\s*(?:>\s*)*\|\s*:?-{3,}:?\s*\|\s*$")
TABLE_ROW_PIPE_RE = re.compile(r"^\s*(?:>\s*)*\|.*\|\s*$")

UNSUPPORTED_DELIM_RE = re.compile(r"(?<!\\)(?:\\\\)*(\\\(|\\\)|\\\[|\\\])")
DOUBLE_DOLLAR_RE = re.compile(r"\$\$")
SINGLE_DOLLAR_RE = re.compile(r"(?<!\$)\$(?!\$)")

# Currency like $5, $100, $1,250.50, $10k, $2.5M, $50-$100, $5/hr
CURRENCY_AFTER_RE = re.compile(
    r"^\d[\d,]*(?:\.\d+)?(?:[kKmMbBtT%]|\+)?(?=$|[^A-Za-z0-9_\\^])"
)
# Shell variables/expansions and template syntax: $HOME, ${VAR}, $(cmd), ${{ ... }}, $1
SHELL_AFTER_RE = re.compile(
    r"^(?:\{\{[^}]*\}\}|\{[A-Za-z_][A-Za-z0-9_:.?+\-=/%#]*\}|\([^)]*\)|[A-Z_][A-Z0-9_]+|[0-9@#?$!*](?=$|[^A-Za-z0-9_\\^]))"
)

# Unambiguous LaTeX math commands that should not appear raw outside math/code
RAW_LATEX_RE = re.compile(
    r"(?<!\\)(?:\\\\)*\\("
    r"mathbb|mathcal|mathfrak|mathrm|mathbf|boldsymbol|mathsf|mathtt|operatorname|"
    r"nabla|partial|infty|approx|equiv|simeq|cong|propto|"
    r"leq|geq|neq|times|cdot|cdots|ldots|ddots|vdots|oplus|otimes|odot|"
    r"notin|subset|subseteq|supset|supseteq|emptyset|varnothing|"
    r"forall|exists|nexists|implies|iff|"
    r"frac|dfrac|tfrac|sqrt|sum|prod|iint|oint|limsup|liminf|argmin|argmax|"
    r"lVert|rVert|lvert|rvert|Vert|langle|rangle|lfloor|rfloor|lceil|rceil|"
    r"widehat|widetilde|overline|"
    r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|"
    r"iota|kappa|lambda|mu|nu|xi|pi|varpi|rho|varrho|sigma|varsigma|"
    r"tau|upsilon|phi|varphi|chi|psi|omega|"
    r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|"
    r"mapsto|rightarrow|leftarrow|leftrightarrow|Rightarrow|Leftarrow|Leftrightarrow|"
    r"uparrow|downarrow|dagger|ddagger|quad|qquad|pm|mp"
    r")\b"
    r"|(?<!\\)(?:\\\\)*\\(?:begin|end|text)\{"
)

TEXT_ONLY_MACRO_RE = re.compile(r"\\(?:text|textbf|textit|textrm|texttt|textsf)\{([^{}]*)\}")
ANY_LABEL_MACRO_RE = re.compile(
    r"\\(?:text|textbf|textit|textrm|texttt|textsf|mathrm|mathbf|mathsf|mathtt|operatorname)\{([^{}]*)\}"
)
DISALLOWED_MATHJAX_MACRO_RE = re.compile(
    r"(?<!\\)(?:\\\\)*\\("
    r"operatorname|DeclareMathOperator|def|newcommand|renewcommand|let|"
    r"href|url|unicode|style|class|cssId|bbox|require|htmlClass|htmlStyle|htmlId"
    r")\b"
)
GFM_DISPLAY_ESCAPE_RE = re.compile(r"(?<!\\)(?:\\\\)*\\([{}|,;!_])")
GFM_INLINE_NONALPHA_ESCAPE_RE = re.compile(r"(?<!\\)(?:\\\\)*\\([^A-Za-z\\])")
URL_RE = re.compile(r"https?://\S+")
MD_LINK_DEST_RE = re.compile(r"\]\([^)\s]+(?:\s+\"[^\"]*\")?\)")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    rule: str
    message: str
    snippet: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "rule": self.rule,
            "message": self.message,
            "snippet": self.snippet,
        }

    def format_human(self) -> str:
        return f"{self.path}:{self.line}:{self.rule}: {self.message}"


def _is_escaped(text: str, index: int) -> bool:
    """Return True if the character at ``index`` is preceded by an odd number of backslashes."""
    backslashes = 0
    pos = index - 1
    while pos >= 0 and text[pos] == "\\":
        backslashes += 1
        pos -= 1
    return (backslashes % 2) == 1


def _strip_blockquote_prefix(line: str) -> tuple[int, str]:
    """Return ``(blockquote_depth, body_after_blockquote_prefix)``.

    Each ``>`` consumes at most one following space (per CommonMark), so extra
    indentation like ``>    $$`` remains visible in ``body``.
    """
    m = BQ_PREFIX_RE.match(line)
    if not m:
        return 0, line
    prefix = m.group(1)
    return prefix.count(">"), line[len(prefix) :]


def _is_blank_for_depth(lines: list[str], idx: int, bq_depth: int) -> bool:
    """Return True if ``lines[idx]`` acts as a blank separator at ``bq_depth``."""
    if idx < 0 or idx >= len(lines):
        return True
    line = lines[idx]
    if line.strip() == "":
        return True
    if bq_depth > 0 and BLANK_BQ_RE.match(line):
        return True
    return False


def _check_disallowed_macros(
    inner: str,
    *,
    path: str,
    line_no: int,
    snippet: str,
    violations: list[Violation],
) -> None:
    """Reject LaTeX macros disallowed by GitHub's restricted MathJax runtime."""
    m = DISALLOWED_MATHJAX_MACRO_RE.search(inner)
    if m:
        macro = m.group(1)
        hint = "use \\mathrm{...} instead of \\operatorname{...}" if macro == "operatorname" else (
            f"remove or rewrite disallowed macro '\\{macro}'"
        )
        violations.append(
            Violation(
                path=path,
                line=line_no,
                rule="disallowed-mathjax-macro",
                message=f"Macro '\\{macro}' is rejected by GitHub MathJax; {hint}",
                snippet=snippet[:120],
            )
        )


def _check_display_math_line(
    inner: str,
    *,
    path: str,
    line_no: int,
    violations: list[Violation],
) -> None:
    """Validate a line inside a ``$$`` display math block."""
    snippet = inner.strip()[:120]
    _check_disallowed_macros(
        inner, path=path, line_no=line_no, snippet=snippet, violations=violations
    )
    m_esc = GFM_DISPLAY_ESCAPE_RE.search(inner)
    if m_esc:
        ch = m_esc.group(1)
        violations.append(
            Violation(
                path=path,
                line=line_no,
                rule="gfm-display-math-escape",
                message=(
                    f"Backslash escape '\\{ch}' inside '$$' display math is stripped by GitHub GFM; "
                    "use \\lbrace/\\rbrace, \\lVert/\\rVert, \\thinspace/\\enspace, remove \\!, "
                    "or use hyphens instead of \\_ inside \\text{}"
                ),
                snippet=snippet,
            )
        )


def _mask_inline_code_and_check_gh_backtick(
    line: str,
    *,
    path: str,
    line_no: int,
    is_table_row: bool,
    violations: list[Violation],
) -> str:
    """Replace inline code spans and valid GitHub ``$`...`$`` math spans with spaces.

    Validates the body of ``$`...`$`` spans for disallowed GitHub MathJax macros,
    table pipes, and unescaped ``_`` inside ``\\text{}``.
    """
    chars = list(line)
    i = 0
    n = len(line)
    while i < n:
        if line[i] == "`":
            j = i
            while j < n and line[j] == "`":
                j += 1
            ticks = j - i
            k = j
            found = -1
            while k < n:
                if line[k] == "`":
                    m = k
                    while m < n and line[m] == "`":
                        m += 1
                    if m - k == ticks:
                        found = m
                        break
                    k = m
                else:
                    k += 1
            if found != -1:
                has_lead_dollar = (
                    ticks == 1
                    and i >= 1
                    and line[i - 1] == "$"
                    and not _is_escaped(line, i - 1)
                    and not (i >= 2 and line[i - 2] == "$")
                )
                has_trail_dollar = (
                    found < n
                    and line[found] == "$"
                    and not (found + 1 < n and line[found + 1] == "$")
                )
                if has_lead_dollar and has_trail_dollar:
                    inner = line[j : found - ticks]
                    snippet = line[i - 1 : found + 1][:120]
                    _check_disallowed_macros(
                        inner, path=path, line_no=line_no, snippet=snippet, violations=violations
                    )
                    if is_table_row and "|" in inner:
                        violations.append(
                            Violation(
                                path=path,
                                line=line_no,
                                rule="table-pipe-in-math",
                                message=(
                                    "Pipe '|' or '\\|' inside inline math on a Markdown table row breaks GFM tables; "
                                    "use \\lVert...\\rVert, \\lvert...\\rvert, \\vert, or \\mid"
                                ),
                                snippet=snippet,
                            )
                        )
                    for m_txt in TEXT_ONLY_MACRO_RE.finditer(inner):
                        if re.search(r"(?<!\\)_", m_txt.group(1)):
                            violations.append(
                                Violation(
                                    path=path,
                                    line=line_no,
                                    rule="emphasis-trap-in-inline-math",
                                    message=(
                                        "Unescaped '_' inside a LaTeX text macro breaks MathJax; "
                                        "escape as '\\_' inside $`...`$ or use hyphenated text"
                                    ),
                                    snippet=snippet,
                                )
                            )
                            break
                    for p in range(i - 1, found + 1):
                        chars[p] = " "
                    i = found + 1
                else:
                    for p in range(i, found):
                        chars[p] = " "
                    i = found
            else:
                i = j
        else:
            i += 1
    return "".join(chars)


def _compute_fenced_lines(lines: list[str]) -> list[bool]:
    """Identify lines inside YAML frontmatter or fenced code blocks."""
    n = len(lines)
    fenced = [False] * n
    start = 0

    # Skip YAML frontmatter at very top of file
    if n >= 2 and lines[0].strip() == "---":
        for idx in range(1, n):
            if lines[idx].strip() in ("---", "..."):
                for k in range(0, idx + 1):
                    fenced[k] = True
                start = idx + 1
                break

    in_fence = False
    fence_char = ""
    fence_len = 0
    for idx in range(start, n):
        line = lines[idx]
        if not in_fence:
            m = FENCE_OPEN_RE.match(line)
            if m:
                marker = m.group(2)
                rest = m.group(3)
                # Backtick code fence info strings cannot contain backticks
                if marker[0] == "`" and "`" in rest:
                    continue
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                fenced[idx] = True
        else:
            fenced[idx] = True
            bq_stripped = re.sub(r"^(\s*(?:>\s*)*)", "", line)
            if re.match(rf"^{re.escape(fence_char)}{{{fence_len},}}\s*$", bq_stripped):
                in_fence = False
    return fenced


def _mask_html_comments(lines: list[str], fenced: list[bool]) -> list[str]:
    """Mask HTML ``<!-- ... -->`` comments on non-fenced lines with spaces."""
    out = list(lines)
    in_comment = False
    for idx, line in enumerate(lines):
        if fenced[idx]:
            continue
        chars = list(line)
        i = 0
        n = len(line)
        while i < n:
            if not in_comment:
                pos = line.find("<!--", i)
                if pos == -1:
                    break
                end = line.find("-->", pos + 4)
                if end == -1:
                    for p in range(pos, n):
                        chars[p] = " "
                    in_comment = True
                    break
                for p in range(pos, end + 3):
                    chars[p] = " "
                i = end + 3
            else:
                end = line.find("-->", i)
                if end == -1:
                    for p in range(i, n):
                        chars[p] = " "
                    break
                for p in range(i, end + 3):
                    chars[p] = " "
                in_comment = False
                i = end + 3
        out[idx] = "".join(chars)
    return out


def _compute_table_rows(lines: list[str], fenced: list[bool]) -> list[bool]:
    """Identify Markdown table rows (both leading-pipe and pipe-separated)."""
    n = len(lines)
    is_table = [False] * n
    for idx, line in enumerate(lines):
        if fenced[idx]:
            continue
        if TABLE_SEP_RE.match(line) or TABLE_SINGLE_SEP_RE.match(line):
            is_table[idx] = True
            if idx > 0 and not fenced[idx - 1] and "|" in lines[idx - 1]:
                is_table[idx - 1] = True
            j = idx + 1
            while j < n and not fenced[j]:
                nxt = lines[j]
                if nxt.strip() == "" or "|" not in nxt:
                    break
                if re.match(r"^\s*(?:>\s*)*#", nxt):
                    break
                is_table[j] = True
                j += 1
        elif TABLE_ROW_PIPE_RE.match(line) and line.count("|") >= 2:
            is_table[idx] = True
    return is_table


def _has_non_alnum_underscore(inner: str) -> bool:
    """Return True if ``inner`` contains an underscore flanked by non-alnum (CommonMark emphasis)."""
    for idx, ch in enumerate(inner):
        if ch == "_":
            lch = inner[idx - 1] if idx > 0 else ""
            rch = inner[idx + 1] if idx + 1 < len(inner) else ""
            if not (lch.isalnum() and rch.isalnum()):
                return True
    return False


def _find_italic_positions(masked_without_math: str) -> list[int]:
    """Locate single/triple '*' emphasis markers outside code and math."""
    positions: list[int] = []
    p = 0
    m_bullet = re.match(r"^(\s*(?:>\s*)*)\*\s+", masked_without_math)
    if m_bullet:
        p = m_bullet.end()
    n = len(masked_without_math)
    while p < n:
        if masked_without_math[p] == "*" and not _is_escaped(masked_without_math, p):
            q = p
            while q < n and masked_without_math[q] == "*":
                q += 1
            if (q - p) in (1, 3):
                positions.append(p)
            p = q
        else:
            p += 1
    return positions


def _check_inline_math_content(
    inner: str,
    *,
    path: str,
    line_no: int,
    is_table_row: bool,
    violations: list[Violation],
    char_before: str = "",
    char_after: str = "",
    in_italics: bool = False,
) -> None:
    """Validate the body and boundaries of a bare inline ``$...$`` math span."""
    snippet = f"${inner}$"[:120]
    before_count = len(violations)

    _check_disallowed_macros(
        inner, path=path, line_no=line_no, snippet=snippet, violations=violations
    )

    # Rule d: pipes (| or \|) inside table-row inline math
    if is_table_row and "|" in inner:
        violations.append(
            Violation(
                path=path,
                line=line_no,
                rule="table-pipe-in-math",
                message=(
                    "Pipe '|' or '\\|' inside inline math on a Markdown table row breaks GFM tables; "
                    "use \\lVert...\\rVert, \\lvert...\\rvert, \\vert, or \\mid"
                ),
                snippet=snippet,
            )
        )

    # Rule e: double backslashes inside inline $...$
    if "\\\\" in inner:
        violations.append(
            Violation(
                path=path,
                line=line_no,
                rule="double-backslash-in-inline-math",
                message=(
                    "Double backslash '\\\\' inside inline math is collapsed by GFM before MathJax runs; "
                    "promote matrices/cases to a multi-line $$ display block or use $`...`$"
                ),
                snippet=snippet,
            )
        )

    # Rule g: genuine GFM underscore/emphasis traps inside inline $...$
    # 1. Double underscore '__' or double asterisk '**' (Markdown bold inside math)
    if "__" in inner or "**" in inner:
        violations.append(
            Violation(
                path=path,
                line=line_no,
                rule="emphasis-trap-in-inline-math",
                message=(
                    "Double emphasis marker ('__' or '**') inside inline math triggers Markdown bold"
                ),
                snippet=snippet,
            )
        )
    # 2. Space-flanked '_..._' inside inline math
    elif re.search(r"(?:^|\s)_[^_\s][^_]*?_(?=$|\s)", inner):
        violations.append(
            Violation(
                path=path,
                line=line_no,
                rule="emphasis-trap-in-inline-math",
                message=(
                    "Space-flanked '_..._' inside inline math triggers Markdown italic emphasis"
                ),
                snippet=snippet,
            )
        )
    else:
        # 3. Unescaped '_' inside text-mode macros (\text{}, \texttt{}, etc.)
        has_text_underscore = False
        for m in TEXT_ONLY_MACRO_RE.finditer(inner):
            body = m.group(1)
            if re.search(r"(?<!\\)_", body):
                has_text_underscore = True
                break
        if has_text_underscore:
            violations.append(
                Violation(
                    path=path,
                    line=line_no,
                    rule="emphasis-trap-in-inline-math",
                    message=(
                        "Unescaped '_' inside a LaTeX text macro breaks MathJax; "
                        "use code backticks outside math or hyphenated text"
                    ),
                    snippet=snippet,
                )
            )
        else:
            # 4. Bare '\_' outside text/operator macros (GFM strips '\' -> MathJax subscript error)
            stripped_label_macros = ANY_LABEL_MACRO_RE.sub(" ", inner)
            if r"\_" in stripped_label_macros:
                violations.append(
                    Violation(
                        path=path,
                        line=line_no,
                        rule="emphasis-trap-in-inline-math",
                        message=(
                            "Bare '\\_' inside inline math is unescaped by GFM to '_', "
                            "causing wrong subscripts or MathJax double-subscript errors; "
                            "use backtick code outside math or hyphenated notation"
                        ),
                        snippet=snippet,
                    )
                )

    # Rule h: GitHub cmark-gfm bare $...$ fragility hazards (use $`...`$ instead).
    # Skip if a more specific inline rule already flagged this span.
    if len(violations) == before_count:
        hazard_reason = ""
        if in_italics:
            hazard_reason = "bare $...$ inside *...* italic emphasis fails in GitHub GFM"
        elif char_before and char_before not in (
            " ",
            "\t",
            "(",
            "[",
            "{",
            "'",
            '"',
            "|",
        ):
            hazard_reason = f"bare $...$ immediately preceded by '{char_before}' fails in GitHub GFM"
        elif inner.endswith(")") and char_after == ")":
            hazard_reason = "bare $...$ ending with ')' immediately followed by ')' fails in GitHub GFM"
        elif char_after and (
            char_after.isalnum() or char_after in ("(", "/", "-", "*", "_", "~", "$", "\\")
        ):
            hazard_reason = f"bare $...$ immediately followed by '{char_after}' fails in GitHub GFM"
        elif GFM_INLINE_NONALPHA_ESCAPE_RE.search(inner):
            m_esc = GFM_INLINE_NONALPHA_ESCAPE_RE.search(inner)
            hazard_reason = (
                f"backslash escape '\\{m_esc.group(1)}' inside bare $...$ is stripped by GitHub GFM"
                if m_esc
                else "punctuation backslash escape inside bare $...$ is stripped by GitHub GFM"
            )
        elif "*" in inner:
            hazard_reason = "'*' inside bare $...$ can trigger CommonMark emphasis in GitHub GFM"
        elif _has_non_alnum_underscore(inner):
            hazard_reason = (
                "underscore adjacent to punctuation/braces inside bare $...$ triggers CommonMark emphasis in GitHub GFM"
            )
        if hazard_reason:
            violations.append(
                Violation(
                    path=path,
                    line=line_no,
                    rule="gfm-bare-math-hazard",
                    message=f"{hazard_reason}; wrap in GitHub-safe $`...`$ inline math",
                    snippet=snippet,
                )
            )


def _scan_inline_dollars(
    masked: str,
    *,
    path: str,
    line_no: int,
    is_table_row: bool,
    violations: list[Violation],
) -> list[tuple[int, int]]:
    """Parse inline ``$...$`` spans on a single non-display line.

    Returns list of ``(start, end_inclusive)`` spans for masking before raw-LaTeX
    checks. Carefully distinguishes currency (``$5``, ``$50-$100``) and shell
    variables (``$HOME``, ``${VAR}``) from broken math.
    """
    dollars = [
        m.start()
        for m in SINGLE_DOLLAR_RE.finditer(masked)
        if not _is_escaped(masked, m.start())
    ]
    if not dollars:
        return []

    n = len(masked)
    used: set[int] = set()
    spans: list[tuple[int, int]] = []

    # Pass 1: match strictly valid GFM/Arithmatex inline math spans.
    # Rules: opener not followed by whitespace; closer not preceded by whitespace
    # and not followed by an ASCII digit; no unescaped $ inside the span.
    idx = 0
    while idx < len(dollars):
        s = dollars[idx]
        if s in used:
            idx += 1
            continue
        after_open = masked[s + 1 :]
        if (
            not after_open
            or after_open[0].isspace()
            or after_open.startswith("{{")
            or after_open.startswith("$")
        ):
            idx += 1
            continue
        if idx + 1 < len(dollars):
            e = dollars[idx + 1]
            inner = masked[s + 1 : e]
            char_after_close = masked[e + 1] if e + 1 < n else ""
            if (
                inner
                and not inner[0].isspace()
                and not inner[-1].isspace()
                and not char_after_close.isdigit()
            ):
                used.add(s)
                used.add(e)
                spans.append((s, e))
                idx += 2
                continue
        idx += 1

    # Compute italic '*' positions outside code and matched inline math spans
    chars_no_math = list(masked)
    for s, e in spans:
        for p in range(s, e + 1):
            chars_no_math[p] = " "
    italic_positions = _find_italic_positions("".join(chars_no_math))

    def _in_italics(pos: int) -> bool:
        if len(italic_positions) < 2:
            return False
        before_cnt = sum(1 for ip in italic_positions if ip < pos)
        after_cnt = len(italic_positions) - before_cnt
        return (before_cnt % 2 == 1) and after_cnt >= 1

    for s, e in spans:
        inner = masked[s + 1 : e]
        char_before = masked[s - 1] if s > 0 else ""
        char_after = masked[e + 1] if e + 1 < n else ""
        _check_inline_math_content(
            inner,
            path=path,
            line_no=line_no,
            is_table_row=is_table_row,
            violations=violations,
            char_before=char_before,
            char_after=char_after,
            in_italics=_in_italics(s),
        )

    # Pass 2: inspect remaining unmatched $ tokens.
    remaining = [pos for pos in dollars if pos not in used]
    if not remaining:
        return spans

    def _is_currency_or_shell(pos: int) -> bool:
        after = masked[pos + 1 :]
        if CURRENCY_AFTER_RE.match(after):
            return True
        if SHELL_AFTER_RE.match(after):
            return True
        return False

    def _is_obvious_padded_closer(pos: int) -> bool:
        before = masked[pos - 1] if pos > 0 else ""
        after = masked[pos + 1] if pos + 1 < n else ""
        return before.isspace() and not (
            after.isdigit() or after.isalpha() or after in ("_", "{", "(")
        )

    # First pair obvious space-padded inline math spans (e.g. "$ x $", "$ x$", "$x $")
    # before filtering out standalone currency/shell tokens.
    paired_rem: set[int] = set()
    r_idx = 0
    while r_idx + 1 < len(remaining):
        s = remaining[r_idx]
        e = remaining[r_idx + 1]
        inner = masked[s + 1 : e]
        char_after_close = masked[e + 1] if e + 1 < n else ""
        s_is_curr = _is_currency_or_shell(s)
        e_is_curr = _is_currency_or_shell(e)

        is_padded_pair = False
        if not char_after_close.isdigit() and not e_is_curr:
            if not s_is_curr and (inner.startswith(" ") or inner.endswith(" ")):
                is_padded_pair = True
            elif s_is_curr and _is_obvious_padded_closer(e):
                is_padded_pair = True

        if is_padded_pair:
            paired_rem.add(s)
            paired_rem.add(e)
            spans.append((s, e))
            violations.append(
                Violation(
                    path=path,
                    line=line_no,
                    rule="space-padded-inline-math",
                    message=(
                        "Inline math '$...$' must not have leading or trailing spaces inside '$'"
                    ),
                    snippet=masked[s : e + 1][:120],
                )
            )
            _check_inline_math_content(
                inner.strip(),
                path=path,
                line_no=line_no,
                is_table_row=is_table_row,
                violations=violations,
            )
            r_idx += 2
            continue
        r_idx += 1

    # Check any remaining unpaired $ tokens.
    for pos in remaining:
        if pos in paired_rem:
            continue
        if _is_currency_or_shell(pos):
            continue
        violations.append(
            Violation(
                path=path,
                line=line_no,
                rule="unmatched-dollar",
                message="Unmatched '$' outside code blocks (escape literal '$' as '\\$' or close inline math)",
                snippet=masked[max(0, pos - 20) : min(n, pos + 40)].strip(),
            )
        )

    return spans


def check_markdown_text(text: str, *, path: str = "<string>") -> list[Violation]:
    """Scan Markdown text and return a list of math rendering violations."""
    raw_lines = text.splitlines()
    fenced = _compute_fenced_lines(raw_lines)
    lines = _mask_html_comments(raw_lines, fenced)
    table_rows = _compute_table_rows(lines, fenced)

    violations: list[Violation] = []
    in_display = False
    display_open_line = 0
    display_open_bq_depth = 0
    display_open_indented = False

    for idx, line in enumerate(lines):
        line_no = idx + 1
        if fenced[idx]:
            continue

        masked = _mask_inline_code_and_check_gh_backtick(
            line,
            path=path,
            line_no=line_no,
            is_table_row=table_rows[idx],
            violations=violations,
        )

        # Find unescaped $$ occurrences on this line
        dd_matches = [
            m
            for m in DOUBLE_DOLLAR_RE.finditer(masked)
            if not _is_escaped(masked, m.start())
        ]

        if dd_matches:
            bq_depth, body_raw = _strip_blockquote_prefix(line)
            _, body_masked = _strip_blockquote_prefix(masked)
            body_dd = [
                m
                for m in DOUBLE_DOLLAR_RE.finditer(body_masked)
                if not _is_escaped(body_masked, m.start())
            ]

            if not in_display:
                if len(body_dd) >= 2:
                    # Single-line $$...$$ (or multiple $$...$$ on one line)
                    first_dd = body_dd[0]
                    last_dd = body_dd[-1]
                    before = body_masked[: first_dd.start()]
                    after = body_masked[last_dd.end() :]
                    is_single_pair = len(body_dd) == 2

                    if LIST_MARKER_RE.match(before):
                        violations.append(
                            Violation(
                                path=path,
                                line=line_no,
                                rule="indented-display-math",
                                message=(
                                    "Display math '$$' nested inside a list item breaks GFM; "
                                    "use a top-level canonical multi-line $$ block with blank lines or inline $...$"
                                ),
                                snippet=line.strip()[:120],
                            )
                        )
                    elif before.strip() or after.strip() or not is_single_pair:
                        violations.append(
                            Violation(
                                path=path,
                                line=line_no,
                                rule="inline-display-math",
                                message=(
                                    "Display math '$$' must not share a line with text; "
                                    "place it in a standalone multi-line $$ block surrounded by blank lines"
                                ),
                                snippet=line.strip()[:120],
                            )
                        )
                    else:
                        # Standalone single-line $$...$$
                        if before != "":
                            violations.append(
                                Violation(
                                    path=path,
                                    line=line_no,
                                    rule="indented-display-math",
                                    message=(
                                        "Indented display math '$$' breaks GFM / creates code blocks; "
                                        "move to column 0 (or single-space '> $$') as a multi-line $$ block"
                                    ),
                                    snippet=line[:120],
                                )
                            )
                        prev_blank = _is_blank_for_depth(raw_lines, idx - 1, bq_depth)
                        next_blank = _is_blank_for_depth(raw_lines, idx + 1, bq_depth)
                        if not prev_blank or not next_blank:
                            violations.append(
                                Violation(
                                    path=path,
                                    line=line_no,
                                    rule="display-math-blank-lines",
                                    message=(
                                        "Display math '$$...$$' must be surrounded by blank lines "
                                        "(or blank '>' lines inside blockquotes); prefer canonical multi-line $$ blocks"
                                    ),
                                    snippet=line.strip()[:120],
                                )
                            )

                    # Inspect inner content of single-line $$...$$ for GFM \\ collapse and macros
                    for pair_i in range(0, len(body_dd) - 1, 2):
                        inner_dd = body_masked[
                            body_dd[pair_i].end() : body_dd[pair_i + 1].start()
                        ]
                        _check_display_math_line(
                            inner_dd, path=path, line_no=line_no, violations=violations
                        )
                        if "\\\\" in inner_dd:
                            violations.append(
                                Violation(
                                    path=path,
                                    line=line_no,
                                    rule="double-backslash-in-inline-math",
                                    message=(
                                        "Double backslash '\\\\' inside single-line $$...$$ is collapsed by GFM; "
                                        "split into a canonical multi-line $$ block"
                                    ),
                                    snippet=body_raw.strip()[:120],
                                )
                            )
                        if table_rows[idx] and "|" in inner_dd:
                            violations.append(
                                Violation(
                                    path=path,
                                    line=line_no,
                                    rule="table-pipe-in-math",
                                    message=(
                                        "Pipe '|' inside math on a Markdown table row breaks GFM tables"
                                    ),
                                    snippet=body_raw.strip()[:120],
                                )
                            )
                    continue

                # Exactly 1 $$ -> opens multi-line display block
                in_display = True
                display_open_line = line_no
                display_open_bq_depth = bq_depth
                display_open_indented = False

                first_dd = body_dd[0]
                before = body_masked[: first_dd.start()]
                after = body_masked[first_dd.end() :]

                if LIST_MARKER_RE.match(before):
                    display_open_indented = True
                    violations.append(
                        Violation(
                            path=path,
                            line=line_no,
                            rule="indented-display-math",
                            message=(
                                "Display math '$$' nested inside a list item breaks GFM; "
                                "use a top-level canonical multi-line $$ block surrounded by blank lines"
                            ),
                            snippet=line.strip()[:120],
                        )
                    )
                elif before.strip() or after.strip():
                    violations.append(
                        Violation(
                            path=path,
                            line=line_no,
                            rule="inline-display-math",
                            message=(
                                "Opening '$$' must sit alone on its own line with no surrounding text"
                            ),
                            snippet=line.strip()[:120],
                        )
                    )
                else:
                    if before != "":
                        display_open_indented = True
                        violations.append(
                            Violation(
                                path=path,
                                line=line_no,
                                rule="indented-display-math",
                                message=(
                                    "Indented display math '$$' breaks GFM / creates code blocks; "
                                    "unindent to column 0 (or single-space '> $$')"
                                ),
                                snippet=line[:120],
                            )
                        )
                    if not _is_blank_for_depth(raw_lines, idx - 1, bq_depth):
                        violations.append(
                            Violation(
                                path=path,
                                line=line_no,
                                rule="display-math-blank-lines",
                                message=(
                                    "Display math opening '$$' must be preceded by a blank line "
                                    "(or blank '>' line inside blockquotes)"
                                ),
                                snippet=line.strip()[:120],
                            )
                        )
                continue

            # Currently inside multi-line display block: this closes it
            in_display = False
            first_dd = body_dd[0]
            before = body_masked[: first_dd.start()]
            after = body_masked[first_dd.end() :]
            if before.strip() or after.strip() or len(body_dd) > 1:
                violations.append(
                    Violation(
                        path=path,
                        line=line_no,
                        rule="inline-display-math",
                        message=(
                            "Closing '$$' must sit alone on its own line with no surrounding text"
                        ),
                        snippet=line.strip()[:120],
                    )
                )
            else:
                if before != "" and not display_open_indented:
                    violations.append(
                        Violation(
                            path=path,
                            line=line_no,
                            rule="indented-display-math",
                            message="Closing '$$' must not be indented",
                            snippet=line[:120],
                        )
                    )
                if not _is_blank_for_depth(raw_lines, idx + 1, display_open_bq_depth):
                    violations.append(
                        Violation(
                            path=path,
                            line=line_no,
                            rule="display-math-blank-lines",
                            message=(
                                "Display math closing '$$' must be followed by a blank line "
                                "(or blank '>' line inside blockquotes)"
                            ),
                            snippet=line.strip()[:120],
                        )
                    )
            continue

        if in_display:
            _, disp_body = _strip_blockquote_prefix(line)
            _check_display_math_line(
                disp_body, path=path, line_no=line_no, violations=violations
            )
            continue

        # Rule a: unsupported LaTeX delimiters \(...\) or \[...\] outside code/display math
        m_delim = UNSUPPORTED_DELIM_RE.search(masked)
        if m_delim:
            violations.append(
                Violation(
                    path=path,
                    line=line_no,
                    rule="unsupported-delimiter",
                    message=(
                        f"LaTeX delimiter '{m_delim.group(1)}' is not rendered by GitHub GFM; "
                        "use $...$ for inline math or canonical multi-line $$ blocks for display math"
                    ),
                    snippet=line.strip()[:120],
                )
            )

        # Rules d, e, f, g: inline $...$ spans
        inline_spans = _scan_inline_dollars(
            masked,
            path=path,
            line_no=line_no,
            is_table_row=table_rows[idx],
            violations=violations,
        )

        # Rule f (part 2): raw LaTeX math commands outside math and code
        outside_chars = list(masked)
        for s, e in inline_spans:
            for k in range(s, e + 1):
                outside_chars[k] = " "
        outside = "".join(outside_chars)
        outside = URL_RE.sub(" ", outside)
        outside = MD_LINK_DEST_RE.sub("]", outside)
        m_raw = RAW_LATEX_RE.search(outside)
        if m_raw and not m_delim:
            violations.append(
                Violation(
                    path=path,
                    line=line_no,
                    rule="raw-latex-outside-math",
                    message=(
                        f"Raw LaTeX command '{m_raw.group(0)}' outside math delimiters; "
                        "wrap in $...$, a $$ block, or inline code backticks"
                    ),
                    snippet=line.strip()[:120],
                )
            )

    if in_display:
        violations.append(
            Violation(
                path=path,
                line=display_open_line,
                rule="unclosed-display-math",
                message="Unclosed '$$' display math block (reached end of file without closing '$$')",
                snippet="$$",
            )
        )

    return violations


def tracked_markdown_files(root: Path) -> list[Path]:
    """Return sorted repository-tracked ``.md`` files under ``root``."""
    root = Path(root).resolve()
    try:
        done = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.md"],
            cwd=str(root),
            capture_output=True,
            check=True,
        )
        rels = [name for name in done.stdout.decode().split("\0") if name]
        return [root / rel for rel in sorted(rels) if (root / rel).is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sorted(
            p for p in root.rglob("*.md") if ".git" not in p.parts and p.is_file()
        )


def _resolve_target_files(paths: Iterable[Path] | None, root: Path) -> list[Path]:
    if not paths:
        return tracked_markdown_files(root)
    files: list[Path] = []
    for raw in paths:
        p = raw if raw.is_absolute() else (root / raw)
        p = p.resolve()
        if p.is_dir():
            files.extend(sorted(item for item in p.rglob("*.md") if item.is_file()))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"Markdown target path does not exist: {raw}")
    # Deduplicate preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def _is_excluded(rel_posix: str, exclude: Iterable[str]) -> bool:
    for pattern in exclude:
        if rel_posix == pattern or fnmatch.fnmatch(rel_posix, pattern):
            return True
    return False


def check_markdown_file(path: Path, *, root: Path | None = None) -> list[Violation]:
    path = Path(path).resolve()
    if root is not None:
        try:
            display = path.relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            display = path.as_posix()
    else:
        display = path.as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    return check_markdown_text(text, path=display)


def check_paths(
    paths: Iterable[Path] | None = None,
    *,
    root: Path = ROOT,
    exclude: Iterable[str] = (),
) -> tuple[int, list[Violation]]:
    root = Path(root).resolve()
    exclude_list = list(exclude)
    targets = _resolve_target_files(paths, root)
    checked = 0
    violations: list[Violation] = []
    for file_path in targets:
        try:
            rel = file_path.relative_to(root).as_posix()
        except ValueError:
            rel = file_path.as_posix()
        if exclude_list and _is_excluded(rel, exclude_list):
            continue
        checked += 1
        violations.extend(check_markdown_file(file_path, root=root))
    return checked, violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root directory (default: repo root)",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        type=Path,
        default=None,
        help="Optional Markdown files or directories to check instead of all tracked .md files",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=(),
        help="Optional repo-relative glob patterns to exclude",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output machine-readable JSON report",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    files_checked, violations = check_paths(
        args.paths, root=root, exclude=args.exclude
    )
    counts_by_rule = dict(sorted(Counter(v.rule for v in violations).items()))

    if args.as_json:
        payload = {
            "passed": not violations,
            "files_checked": files_checked,
            "violation_count": len(violations),
            "counts_by_rule": counts_by_rule,
            "violations": [v.to_dict() for v in violations],
        }
        print(json.dumps(payload, indent=2))
        return 0 if not violations else 1

    if violations:
        for v in violations:
            print(v.format_human())
        summary = ", ".join(f"{rule}={count}" for rule, count in counts_by_rule.items())
        print(
            f"Markdown math check failed: {len(violations)} violation(s) across "
            f"{len({v.path for v in violations})} file(s) ({files_checked} checked) [{summary}]"
        )
        return 1

    print(f"Markdown math check passed: 0 violations ({files_checked} Markdown files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
