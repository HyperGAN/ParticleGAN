"""Unit tests and repository gate for ``scripts/check_markdown_math.py``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_markdown_math import (
    ROOT,
    check_markdown_text,
    check_paths,
    main,
)


def _rules(text: str) -> list[str]:
    return [v.rule for v in check_markdown_text(text, path="sample.md")]


def test_unsupported_delimiters_detected_and_ignored_in_code() -> None:
    assert "unsupported-delimiter" in _rules(r"Inline math \(x^2 + y^2\) fails on GFM.")
    assert "unsupported-delimiter" in _rules(r"Display \[a + b\] fails on GFM.")
    assert _rules(r"Escaped literal \\(not math\\) is fine.") == []
    assert _rules(r"Code span `\(x^2\)` and `\[y\]` are fine.") == []
    assert _rules("```latex\n\\(x^2\\)\n\\[y\\]\n```\n") == []


def test_github_backtick_math_accepted_and_validated() -> None:
    assert _rules("GitHub-safe $`x + y`$ and $`\\mathbb{E}_x[D_\\psi(x)]`$ pass.") == []
    assert _rules("Literal code spans `$$` and `$x$` are fine.") == []
    assert _rules("Double backticks ``$`x`$`` inside code are fine.") == []
    assert _rules("Banned macro $`\\operatorname{supp}(P_Z)`$ fails.") == ["disallowed-mathjax-macro"]
    bad_tbl = "| A | B |\n| --- | --- |\n| 1 | $`\\|x\\|_2`$ |\n"
    assert _rules(bad_tbl) == ["table-pipe-in-math"]
    good_tbl = "| A | B |\n| --- | --- |\n| 1 | $`\\lVert x \\rVert_2`$ |\n"
    assert _rules(good_tbl) == []


def test_disallowed_mathjax_macros_and_gfm_hazards() -> None:
    assert _rules(r"Bare $\operatorname{supp}(P_Z)$ fails.") == ["disallowed-mathjax-macro"]
    assert _rules("\n$$\n\\operatorname{rank}(A) = 1\n$$\n") == ["disallowed-mathjax-macro"]
    assert _rules("\n$$\n\\{1, 2\\}\n$$\n") == ["gfm-display-math-escape"]
    assert _rules("\n$$\n\\lbrace 1, 2 \\rbrace\n$$\n") == []
    assert _rules(r"Norm $\|J_G\|_\infty$ breaks GFM.") == ["gfm-bare-math-hazard"]
    assert _rules(r"Brace underscore $\mathbb{E}_x[D_\psi(x)]$ breaks GFM.") == ["gfm-bare-math-hazard"]
    assert _rules(r"Paren ($\kappa(J_G)$) breaks GFM.") == ["gfm-bare-math-hazard"]
    assert _rules(r"Prefix fixed-$\varepsilon$ breaks GFM.") == ["gfm-bare-math-hazard"]
    assert _rules(r"Equals decay=$0$ breaks GFM.") == ["gfm-bare-math-hazard"]
    assert _rules(r"Comma min(horizon,$1600$) breaks GFM.") == ["gfm-bare-math-hazard"]
    assert _rules(r"*Unconstrained $W_2$ channel:*") == ["gfm-bare-math-hazard"]
    # GitHub-safe $`...`$ forms pass cleanly
    assert _rules(r"Norm $`\lVert J_G \rVert_\infty`$ works.") == []
    assert _rules(r"Paren ($`\kappa(J_G)`$) works.") == []
    assert _rules(r"Prefix fixed-$`\varepsilon`$ and decay=$`0`$ work.") == []
    assert _rules(r"*Unconstrained $`W_2`$ channel:*") == []


def test_indented_and_list_nested_display_math_detected() -> None:
    text = "- Bullet item:\n  $$\n  x = 1\n  $$\n"
    assert "indented-display-math" in _rules(text)

    list_single = "1. $$x = 1$$\n"
    assert "indented-display-math" in _rules(list_single)

    bq_overindented = "> 1. Item:\n>\n>    $$x = 1$$\n"
    assert "indented-display-math" in _rules(bq_overindented)

    indented_close = "\n$$\nx = 1\n  $$\n"
    assert "indented-display-math" in _rules(indented_close)


def test_inline_display_math_and_blank_line_rules() -> None:
    assert "inline-display-math" in _rules("Prose $$x = 1$$ inline.")
    assert "inline-display-math" in _rules("\n$$x = 1\n+ 2\n$$\n")
    assert "inline-display-math" in _rules("\n$$\nx = 1$$\n")

    assert _rules("No blank before:\n$$\nx = 1\n$$\n") == ["display-math-blank-lines"]
    assert _rules("\n$$\nx = 1\n$$\nNo blank after.") == ["display-math-blank-lines"]
    assert _rules("> Quote:\n> $$x = 1$$\n> After.") == ["display-math-blank-lines"]

    canonical_top = "Before:\n\n$$\nx = 1\n$$\n\nAfter.\n"
    assert _rules(canonical_top) == []

    canonical_bq = "> Before:\n>\n> $$\n> x = 1\n> $$\n>\n> After.\n"
    assert _rules(canonical_bq) == []


def test_table_pipe_in_math_detected_and_safe_alternatives_pass() -> None:
    bad_table = (
        "| Metric | Formula |\n"
        "| :--- | :--- |\n"
        "| Norm | $\\|x\\|_2$ |\n"
        "| Cond | $P(Y | c)$ |\n"
    )
    assert _rules(bad_table) == ["table-pipe-in-math", "table-pipe-in-math"]

    good_table = (
        "| Metric | Formula |\n"
        "| :--- | :--- |\n"
        "| Norm | $\\lVert x \\rVert_2$ |\n"
        "| Abs | $\\lvert x \\rvert$ |\n"
        "| Cond | $P(Y \\mid c)$ |\n"
    )
    assert _rules(good_table) == []

    # Safe norm alternatives outside tables pass cleanly
    assert _rules("Outside tables, $\\lVert x \\rVert_2$ and $P(Y \\mid c)$ render fine.") == []


def test_double_backslash_in_inline_and_single_line_display_math() -> None:
    inline_mat = r"Matrix $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ inline."
    assert _rules(inline_mat) == ["double-backslash-in-inline-math"]

    single_line_disp = "\n$$\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}$$\n"
    assert _rules(single_line_disp) == ["double-backslash-in-inline-math"]

    canonical_multi = (
        "\n$$\n"
        "A = \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}\n"
        "$$\n"
    )
    assert _rules(canonical_multi) == []


def test_space_padded_unclosed_and_unmatched_dollars() -> None:
    assert _rules("Bad padding $ x + 1 $ here.") == ["space-padded-inline-math"]
    assert _rules("Bad left pad $ x+1$ here.") == ["space-padded-inline-math"]
    assert _rules("Bad right pad $x+1 $ here.") == ["space-padded-inline-math"]
    assert _rules("Valid $x + 1$ here.") == []

    assert _rules("\n$$\nx = 1\n") == ["unclosed-display-math"]
    assert _rules("Unclosed math $x + y in prose.") == ["unmatched-dollar"]
    assert _rules(r"Escaped dollar \$x + y is literal prose.") == []


def test_raw_latex_outside_math_detected_and_code_exempt() -> None:
    assert _rules(r"In \mathbb{R}^d we compute \nabla_x D.") == ["raw-latex-outside-math"]
    assert _rules(r"Use \operatorname{rank}(G) and \frac{1}{2}.") == ["raw-latex-outside-math"]
    assert _rules(r"Norm \lVert x \rVert outside dollars.") == ["raw-latex-outside-math"]
    assert _rules(r"Wrapped $\mathbb{R}^d$ and $\nabla_x D$ pass.") == []
    assert _rules(r"Inline code `\mathbb{R}` and `\nabla` pass.") == []
    assert _rules("---\ntitle: \\alpha \\mathbb{R}\n---\n\nClean prose.\n") == []
    assert _rules("<!-- \\mathbb{R} $unclosed -->\nClean prose.\n") == []
    assert _rules("Link [ref](https://example.com/\\alpha) is ignored.") == []


def test_emphasis_traps_in_inline_math() -> None:
    assert _rules("Double underscore $a__b$ breaks.") == ["emphasis-trap-in-inline-math"]
    assert _rules("Double star $a**b$ breaks.") == ["emphasis-trap-in-inline-math"]
    assert _rules("Space-flanked $a _italic_ b$ breaks.") == ["emphasis-trap-in-inline-math"]
    assert _rules(r"Text underscore $\text{foo_bar}$ breaks.") == ["emphasis-trap-in-inline-math"]
    assert _rules(r"Bare escaped underscore $foo\_bar$ breaks.") == ["emphasis-trap-in-inline-math"]

    # Legitimate math subscripts pass cleanly; complex punctuation-flanked subscripts use $`...`$
    assert _rules(r"Subscripts $a_i + b_j = c_k$ are valid.") == []
    assert _rules(r"Math font macro $`\mathbf{26\lVert G\rVert_F^4}`$ is valid.") == []
    assert _rules(r"Nested text $`r_{\text{stable}}`$ is valid.") == []


@pytest.mark.parametrize(
    "prose",
    [
        "GPU costs $5 per hour or $100/day.",
        "Budget range is $50-$100 or $1,250.50 total ($10k-$2.5M).",
        "Cloud runs cost $5 to $10 on L4.",
        "Pay $5 for exact $x^2$ evaluation.",
        "Shell vars $HOME, ${PYTHONPATH}, $(pwd), and $1 are fine.",
        "GitHub Actions uses ${{ matrix.shard }} and ${{ github.sha }}.",
        "Inline code `$$`, `$x$`, `\\|x\\|`, and `\\mathbb{R}` are fine.",
        "~~~\n\\(x\\) $unclosed $$bad$$\n~~~\n",
        "Speedup spans $`10^2`$–$`10^6\\times`$ across regimes.",
    ],
)
def test_no_false_positives_on_legitimate_prose(prose: str) -> None:
    assert _rules(prose) == []


def test_cli_json_paths_and_exclude(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "good.md"
    bad = tmp_path / "bad.md"
    good.write_text("# Clean\n\n$$\nx = 1\n$$\n", encoding="utf-8")
    bad.write_text("Broken \\(x\\) and $ y $.\n", encoding="utf-8")

    # Fails on bad.md with JSON report
    rc = main(["--root", str(tmp_path), "--paths", str(tmp_path), "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["files_checked"] == 2
    assert payload["violation_count"] == 2
    assert set(payload["counts_by_rule"]) == {
        "unsupported-delimiter",
        "space-padded-inline-math",
    }

    # Human output includes path:line:rule format
    rc_human = main(["--root", str(tmp_path), "--paths", str(bad)])
    assert rc_human == 1
    out_human = capsys.readouterr().out
    assert "bad.md:1:unsupported-delimiter:" in out_human
    assert "bad.md:1:space-padded-inline-math:" in out_human

    # Excluding bad.md passes cleanly
    rc_excl = main(
        ["--root", str(tmp_path), "--paths", str(tmp_path), "--exclude", "bad.md"]
    )
    assert rc_excl == 0
    assert "0 violations" in capsys.readouterr().out


def test_repository_markdown_math_clean() -> None:
    """Repository gate: every tracked Markdown file renders cleanly on GitHub GFM and MkDocs."""
    checked, violations = check_paths(root=ROOT)
    assert checked > 0
    assert violations == [], "\n".join(v.format_human() for v in violations[:25])
