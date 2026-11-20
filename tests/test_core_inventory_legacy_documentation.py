"""Editorial configuration navigation preserves identity and explicit choices."""
from pathlib import Path

from experiments.forge.contracts import read_json
from experiments.forge.family_documentation import validate_family_documentation
from experiments.forge.trainer_families import (CURRENT_SELECTION, current_family_candidates,
                                               family_for_candidate, load_families)

ROOT = Path(__file__).resolve().parents[1]
LEGACY_IDS = {"bcap-develop-integration-" + name + "-v1" for name in ("combined", "winner")}
LEGACY_IDS.update("bcap-projection-baseline-review-" + name + "-v1" for name in ("baseline", "direction"))
LEGACY_IDS.update("bcap-three-phase-" + name + "-v1" for name in ("cap-margin", "finite-cap", "incumbent"))
LEGACY_IDS.update("bcap-tier1-stability-" + name + "-v1" for name in
                  ("combined", "incumbent", "projection-global", "projection-local", "projection",
                   "repairs-cap-margin", "repairs-finite-cap", "transport"))


def test_all_legacy_configuration_docs_are_valid_and_keep_original_identity():
    assert validate_family_documentation(ROOT) == {"status": "valid", "families": 40, "tags": 12}
    registry = load_families(ROOT)
    for name in LEGACY_IDS:
        card = read_json(ROOT / "configs/forge/ideas" / (name + ".json"))
        historical = family_for_candidate(ROOT, name, card, current_presentation=False)
        editorial = family_for_candidate(ROOT, name, card, current_presentation=True)
        assert historical["id"] == editorial["id"] == name
        assert editorial["candidates"] == [name] and editorial["canonical_candidate"] == name
        assert editorial["reporting_family"] == "bcap-pure"
        assert "active_search_by_backend" not in registry[name]


def test_legacy_editorial_aliases_do_not_add_or_replace_primary_benchmarks():
    card = read_json(ROOT / CURRENT_SELECTION)
    choices = current_family_candidates(ROOT, selection_card=card)
    assert not ({item["configuration_family"] for item in choices} & LEGACY_IDS)
    bcap = next(item for item in choices if item["family"] == "bcap-pure")
    expected = next(pin for pin in card["selections"] if pin["trainer_family"] == "bcap-dualnorm")
    assert bcap["configuration_family"] == "bcap-dualnorm"
    assert all(bcap[key] == value for key, value in expected.items())
