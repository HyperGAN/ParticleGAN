"""The canonical regeneration path must retain each published research score."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from experiments.forge.family_reports import render_family
from experiments.forge.research_tiers import historical_seed_rows, tier_rows, _row
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_board_keeps_scored_research_and_existing_navigation():
    source = ROOT / "reports/forge/regenerate_technique_inventory.py"
    spec = importlib.util.spec_from_file_location("scored_research_regenerator", source)
    regenerator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regenerator)
    publication = json.loads((ROOT / "reports/forge/technique-inventory.json").read_text())
    original = deepcopy(publication)
    page = ROOT / "reports/forge/technique-inventory.md"
    first = regenerator._current_markdown(publication, ROOT, page)
    second = regenerator._current_markdown(publication, ROOT, page)
    assert publication == original
    assert first == second == page.read_text()
    lines = first.splitlines()
    header = ("| Family / selected recipe | Tier 1 requirements | Tier 2 requirements | Tier 3 requirements"
              " | Tier 4 requirements | Total view requirements |")
    assert next(line for line in lines if line.startswith("|")) == header
    selected, full_suite = first.split("## Full-suite results — failures included", 1)
    assert "<details>" not in selected and "<summary>Qualification subset" not in first
    assert "## Selected recipes" in selected
    assert "A held FAIL is still a failure" in full_suite
    details = {f["id"]: render_family(ROOT, publication, f)
               for f in publication["family_progress"]["families"] if f["id"] in {"atlas", "bcap-pure"}}
    for heading in (
        "## Preserved Atlas scores — separate original cohorts",
        "## Preserved catalog scores — original CPU geometry comparisons",
        "## Source-bound benchmark rows",
        "## G-only coherence: registered paired research",
        "## Confidence-LR correction: six-task screen and own retention",
    ):
        assert heading not in first, heading
        assert sum(detail.count(heading) for detail in details.values()) == 1, heading
    assert "0 PASS / 1 FAIL / 2 INCOMPLETE / 4 BLOCKED across 7 questions" in details["atlas"]
    bcap = details["bcap-pure"]
    assert "FAIL → PASS | 14 → 15 | 3 → 5" in bcap
    assert "69 of 72 observation checks" in bcap
    assert "**1/2 tasks per arm**" in bcap
    assert "Stationary checks improve in all six pairs" in bcap
    assert "original word hold **PASS 25/25**" in bcap
    assert bcap.count("BCAP initial Adam study") == 1
    assert first.count("[Prior-search measurements and gates]") == 1
    assert bcap.count("## Scoped non-Atlas discoveries") == 1
    assert bcap.count("## Quota-transition and Astra portfolio follow-up") == 1
    assert "[source-bound-board.json](../source-bound-board.json)" in bcap


def test_research_tiers_preserve_partial_coverage_and_each_failed_retention():
    rows = tier_rows(ROOT, ROOT / "reports/forge/technique-inventory.md")
    assert len(rows["bcap-pure"]) == 22  # Backfilled research, strict audit and actual deeper diagnostics.
    assert "1/7 (4 diagnostic attempts)" in rows["bcap-pure"][-2]
    # Seed-only studies are excluded from the backfill and listed separately, still unrun.
    assert not any("research seed" in row for row in rows["bcap-pure"])
    paired = [row for row in historical_seed_rows(ROOT, ROOT / "reports/forge/technique-inventory.md")["bcap-pure"]
              if "research seed" in row]
    assert len(paired) == 12
    for row in paired:
        assert "[1/6 (5 not run)]" in row and "[0/21 (20 not run)]" in row
        assert "[0/2 (2 not run)]" in row and "[1/29 (27 not run)]" in row
    finite = next(row for row in rows["bcap-pure"] if "Finite critic cap" in row)
    assert "[4/6 (1 timeout)]" in finite and "[7/21 (2 blocked)]" in finite
    assert "[11/29 (3 blocked, 1 timeout)]" in finite
    assert "[0/7 (4 incompatible)]" in rows["atlas"][0] and " | — | — | " in rows["atlas"][0]
    # A phase observation or an unassigned task cannot become a benchmark pass.
    with pytest.raises(ValueError, match="no recorded tier"):
        _row(ROOT, ROOT / "board.md", "invalid", "result.json", {"stationary_check": "PASS"},
             {"1": {"smoke"}, "2": {"retention"}, "3": set()})
