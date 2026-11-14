"""Root documentation links resolve and point newcomers to one current status page."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "RESEARCH_GLOSSARY.md", "EXPERIMENTATION.md", "FINDINGS.md")
LINK = re.compile(r"!?\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)\)")
INVENTORY = ROOT / "reports/forge/technique-inventory.md"
BANNER = "<!-- superseded-banner: generated -->"
# Dated pages that still read as "current"; each must carry the superseded banner.
STALE_PAGES = (
    "reports/cifar-ddgan/README.md", "reports/cifar-ddgan/READOUT.md",
    "reports/cifar-ddgan/RUNBOOK.md", "reports/cifar-ddgan/ANIMA_HANDOFF.md",
    "reports/toy100/README.md", "reports/toy100/gpu-leaderboard.md",
    "reports/toy100/formulation-search.md", "reports/toy100/h_stability/SEARCH.md",
    "reports/toy100/continuous-practical-leaderboard-cpu-history.md",
    "reports/toy100/formulation-search-before-dimension-rms.md",
    "reports/toy100/formulation-search-before-direct-particle.md",
    "reports/toy100/formulation-search-before-k3p.md",
    "reports/toy_audit/OPEN_PR_DECISIONS.md", "reports/transfer_suite/README.md",
)


def _slugs(headings):
    """GitHub heading anchors, including the -1, -2 suffixes of repeated headings."""
    seen, out = {}, set()
    for heading in headings:
        slug = re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        out.add(f"{slug}-{count}" if count else slug)
    return out


def _anchors(path):
    text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
    return _slugs(re.findall(r"^#{1,6} (.*)$", text, re.M))


def test_anchor_slugs_number_repeated_headings_like_github():
    assert _slugs(["Results", "Notes: v2", "Results", "Results"]) == {
        "results", "notes-v2", "results-1", "results-2"}


def _links(path):
    return LINK.findall(re.sub(r"```.*?```", "", path.read_text(), flags=re.S))


@pytest.mark.parametrize("name", DOCS)
def test_root_doc_relative_links_and_anchors_resolve(name):
    path = ROOT / name
    broken = []
    for target in _links(path):
        if re.match(r"[a-z]+:", target):
            continue
        file_part, _, anchor = target.partition("#")
        dest = (path.parent / file_part) if file_part else path
        if not dest.exists() or (anchor and dest.suffix == ".md" and anchor not in _anchors(dest)):
            broken.append(target)
    assert not broken, broken


def test_glossary_expands_or_marks_every_heavily_used_acronym():
    glossary = (ROOT / "RESEARCH_GLOSSARY.md").read_text()
    for name in ("EMPF", "BISPF", "SBNP", "BCAP", "K3P", "KA2", "E22", "UCD", "DV12", "hq"):
        assert re.search(rf"^\| \*\*{name}\*\*", glossary, re.M), name


def test_superseded_banners_link_to_the_single_current_inventory():
    bannered = [p for p in (ROOT / "reports").rglob("*.md")
                if "attempts" not in p.parts and BANNER in p.read_text(errors="replace")]
    rel = {p.relative_to(ROOT).as_posix() for p in bannered}
    missing = [name for name in STALE_PAGES if name not in rel]
    assert not missing, missing
    for path in bannered + [ROOT / "FINDINGS.md"]:
        targets = [(path.parent / t).resolve() for t in _links(path) if t.endswith("technique-inventory.md")]
        assert INVENTORY.resolve() in targets, path
