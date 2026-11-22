"""Editorial family descriptions and tags cannot create scientific evidence."""
from copy import deepcopy
import base64
import json
from pathlib import Path

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json
from experiments.forge.family_documentation import TRAINING_DETAILS, validate_documentation, validate_family_documentation
from experiments.forge.family_reports import build_progress, generated_pages, render_leaderboard


@pytest.fixture
def documented(tmp_path):
    source = tmp_path / "trainer.py"
    source.write_text("# documented implementation fixture\n")
    documentation = {
        "overview": "A fixed critic penalty with ordinary Adam updates.",
        "symbols": "D is the critic; G is the generator; x and z are samples.",
        "pseudocode": ["x, z = sample_batches()", "update D using loss + penalty", "update G using loss"],
        "training_details": {key: "Disabled or fixed as declared." for key, _ in TRAINING_DETAILS},
        "configuration_notes": ["Selected rows retain their original rates and source."],
        "references": [{"title": "Adam: A Method for Stochastic Optimization",
                        "url": "https://arxiv.org/abs/1412.6980", "component": "Optimizer only."}],
        "sources": ["trainer.py"],
    }
    atomic_json(tmp_path / "configs/forge/family-documentation/bcap.json", documentation)
    atomic_json(tmp_path / "configs/forge/family-tags.json", {"tags": {
        "constant-learning-rate": "Rates stay fixed throughout training.",
        "critic-gradient-penalty": "Regularize critic derivatives with respect to inputs.",
    }})
    atomic_json(tmp_path / "configs/forge/trainer-families.json", {"schema_version": 1, "families": [{
        "id": "bcap", "label": "BCAP", "tags": ["constant-learning-rate", "critic-gradient-penalty"],
    }]})
    publication = {"rows": [{"trainer_family": "bcap", "candidate_id": "bcap-config", "technique": "BCAP",
                             "attempt_ids": [], "runtime_cohort": {"execution_backend": "cuda"}}],
                   "view": "fixture", "view_revision": 1, "provenance": {"input_digest": "fixture"}}
    return tmp_path, publication


def refresh(root, publication):
    publication["family_progress"] = build_progress(root, publication)
    return publication["family_progress"]["families"][0]


def test_explanation_opens_family_page_and_references_are_last(documented):
    root, publication = documented
    recorded = deepcopy(publication["rows"])
    family = refresh(root, publication)
    page = generated_pages(root, publication)[root / family["page"]]
    assert page.index("## Technique overview") < page.index("## CUDA results")
    assert page.index("## Simplified pseudocode") < page.index("## Training details")
    assert "```text\nx, z = sample_batches()" in page
    assert "| Parameter-gradient clipping |" in page
    assert "<summary>Implementation and recipe sources</summary>" in page
    assert page.rsplit("## ", 1)[1].startswith("References\n")
    assert page.rstrip().endswith("Optimizer only.")
    assert publication["rows"] == recorded
    assert publication["family_progress"]["qualification_input"] is False
    assert family["tags"] == ["constant-learning-rate", "critic-gradient-penalty"]
    for path in ("configs/forge/family-documentation/bcap.json", "configs/forge/family-tags.json",
                 "configs/forge/trainer-families.json"):
        assert publication["family_progress"]["input_hashes"][path] == file_hash(root / path)


def test_tag_directory_is_reusable_and_family_labels_open_page_top(documented):
    root, publication = documented
    family = refresh(root, publication)
    inventory = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert "**[BCAP](families/bcap.md)**" in inventory
    assert "[0/0](families/bcap.md#cohort-cuda-" in inventory
    assert '<a name="tag-constant-learning-rate"></a>' in inventory
    assert inventory.index("### `constant-learning-rate`") < inventory.index("### `critic-gradient-penalty`")
    assert "Rates stay fixed throughout training." in inventory
    page = generated_pages(root, publication)[root / family["page"]]
    assert "[constant-learning-rate](../technique-inventory.md#tag-constant-learning-rate)" in page
    assert inventory == render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert publication["family_progress"] == build_progress(root, publication)


def test_historical_family_pages_have_documentation_and_tags_without_entering_totals(documented):
    root, publication = documented
    registry_path = root / "configs/forge/trainer-families.json"
    registry = read_json(registry_path)
    registry["historical_families"] = [{"id": "bcap-old", "label": "Historical BCAP",
                                        "tags": ["critic-gradient-penalty"]}]
    atomic_json(registry_path, registry)
    atomic_json(root / "configs/forge/family-documentation/bcap-old.json",
                read_json(root / "configs/forge/family-documentation/bcap.json"))
    historical = deepcopy(publication["rows"][0])
    historical.update(trainer_family="bcap-old", technique="Historical BCAP")
    publication["historical_family_rows"] = [historical]
    refresh(root, publication)
    assert len(publication["family_progress"]["families"]) == 1
    old = publication["family_progress"]["historical_families"][0]
    assert old["tags"] == ["critic-gradient-penalty"]
    assert "## Simplified pseudocode" in generated_pages(root, publication)[root / old["page"]]
    inventory = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert "[Historical BCAP](families/bcap-old.md) (historical cohort)" in inventory


@pytest.mark.parametrize("tags, error", [
    ([], "nonempty tag list"),
    (["critic-gradient-penalty", "constant-learning-rate"], "lexically sorted"),
    (["constant-learning-rate", "constant-learning-rate"], "unique"),
    (["Constant-Learning-Rate"], "lowercase hyphenated"),
    (["gradient_clipping"], "lowercase hyphenated"),
    (["unknown-tag"], "undefined tags"),
])
def test_family_tags_reject_malformed_or_undefined_values(documented, tags, error):
    root, publication = documented
    path = root / "configs/forge/trainer-families.json"
    registry = read_json(path)
    registry["families"][0]["tags"] = tags
    atomic_json(path, registry)
    with pytest.raises(ValueError, match=error):
        build_progress(root, publication)


def test_partial_documentation_is_an_error_once_enabled(documented):
    root, publication = documented
    (root / "configs/forge/family-documentation/bcap.json").unlink()
    with pytest.raises(ValueError, match="missing registered family documentation"):
        build_progress(root, publication)


def test_legacy_roots_without_documentation_have_explicit_unavailable_notice(documented):
    root, publication = documented
    path = root / "configs/forge/family-documentation"
    (path / "bcap.json").unlink()
    path.rmdir()
    family = refresh(root, publication)
    assert family["tags"] == [] and family["documentation"] is None
    text = generated_pages(root, publication)[root / family["page"]]
    assert "Maintained technique documentation is unavailable" in text
    assert "## References" not in text


@pytest.mark.parametrize("field", ["overview", "symbols", "pseudocode", "training_details",
                                  "configuration_notes", "references", "sources"])
def test_required_sections_cannot_be_silently_dropped(documented, field):
    root, _ = documented
    data = read_json(root / "configs/forge/family-documentation/bcap.json")
    del data[field]
    with pytest.raises(ValueError, match="needs exactly"):
        validate_documentation(root, data, "bcap")


@pytest.mark.parametrize("source, error", [
    ("missing.py", "existing repository file"),
    ("../trainer.py", "existing repository file"),
    ("/etc/passwd", "existing repository file"),
    ("https://github.com/example/project/blob/develop/trainer.py", "commit-pinned"),
    ("https://example.com/trainer.py", "commit-pinned"),
])
def test_sources_resolve_or_bind_an_immutable_historical_commit(documented, source, error):
    root, _ = documented
    data = read_json(root / "configs/forge/family-documentation/bcap.json")
    data["sources"] = [source]
    with pytest.raises(ValueError, match=error):
        validate_documentation(root, data, "bcap")


def test_pinned_historical_source_is_supported_and_missing_paper_is_explicit(documented):
    root, publication = documented
    path = root / "configs/forge/family-documentation/bcap.json"
    data = read_json(path)
    data.update(sources=["https://github.com/HyperGAN/ParticleGAN/blob/" + "a" * 40 + "/trainer.py"],
                references=[])
    atomic_json(path, data)
    family = refresh(root, publication)
    text = generated_pages(root, publication)[root / family["page"]]
    assert text.rstrip().endswith("Its implementation and recipe sources are linked above.")
    assert "no dedicated paper reference is declared" in text


def test_training_table_requires_every_characteristic_and_escapes_delimiters(documented):
    root, publication = documented
    path = root / "configs/forge/family-documentation/bcap.json"
    data = read_json(path)
    data["training_details"]["loss"] = "Variant A | Variant B\nTask-dependent."
    atomic_json(path, data)
    family = refresh(root, publication)
    text = generated_pages(root, publication)[root / family["page"]]
    assert "Variant A \\| Variant B Task-dependent." in text
    del data["training_details"]["gradient_clipping"]
    with pytest.raises(ValueError, match="every declared training characteristic"):
        validate_documentation(root, data, "bcap")


@pytest.mark.parametrize("reference", [
    {"title": "Paper", "url": "http://example.com", "component": "Optimizer."},
    {"title": "", "url": "https://example.com", "component": "Optimizer."},
    {"title": "Paper", "url": "https://example.com", "component": ""},
    {"title": "Paper", "url": "https://example.com"},
])
def test_references_require_a_link_and_a_specific_supported_component(documented, reference):
    root, _ = documented
    data = read_json(root / "configs/forge/family-documentation/bcap.json")
    data["references"] = [reference]
    with pytest.raises(ValueError):
        validate_documentation(root, data, "bcap")


def test_normal_forge_validate_checks_editorial_metadata_without_training(documented, capsys):
    from experiments.forge.__main__ import main
    root, _ = documented
    atomic_json(root / "configs/forge/tasks/fixture.json", {
        "schema_version": 1, "id": "fixture", "adapter": "fixture",
        "execution": {"initializer": "deterministic_orthogonal", "prior": {
            "kind": "mog", "sigma": .025, "standardize": False, "learnable": True}},
        "evaluation": {"kind": "fixture"}, "resources": {"timeout_seconds": 1},
        "requires_capabilities": [], "dependencies": [],
    })
    assert main(["--root", str(root), "--queue-root", str(root / "queue"), "validate"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["family_documentation"] == {"status": "valid", "families": 1, "tags": 2}
    assert result["training_launched"] is False
    path = root / "configs/forge/trainer-families.json"
    registry = read_json(path)
    registry["families"][0]["tags"] = ["unknown-tag"]
    atomic_json(path, registry)
    with pytest.raises(ValueError, match="undefined tags"):
        main(["--root", str(root), "--queue-root", str(root / "queue"), "validate"])


def test_validation_checks_registered_families_even_before_they_have_results(documented):
    root, publication = documented
    path = root / "configs/forge/trainer-families.json"
    registry = read_json(path)
    registry["families"].append({"id": "unmeasured", "tags": ["constant-learning-rate"]})
    atomic_json(path, registry)
    refresh(root, publication)  # The inventory only links its recorded family.
    with pytest.raises(ValueError, match="missing registered family documentation for unmeasured"):
        validate_family_documentation(root)


def _add_math_and_image(root):
    path = root / "configs/forge/family-documentation/bcap.json"
    data = read_json(path)
    asset = root / "reports/forge/families/assets/bcap-explainer.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aN1cAAAAASUVORK5CYII="))
    data.update(equations=[{"label": "Critic objective", "latex": r"L_D = L_{\mathrm{adv}} + \lambda R(D)",
                           "explanation": "The critic minimizes its adversarial loss plus the penalty."}],
                illustration={"path": asset.relative_to(root).as_posix(),
                              "alt": "Critic [input] gradient branches", "caption": "Conceptual mechanism, not measured data."})
    atomic_json(path, data)
    return path, data, asset


def test_math_renders_outside_text_fences_after_symbols_and_preserves_science(documented):
    root, publication = documented
    recorded = deepcopy(publication["rows"])
    _, data, _ = _add_math_and_image(root)
    family = refresh(root, publication)
    page = generated_pages(root, publication)[root / family["page"]]
    assert page.index(data["symbols"]) < page.index("$$") < page.index("## Simplified pseudocode")
    assert "$$\n" + data["equations"][0]["latex"] + "\n$$" in page
    assert page.index(data["equations"][0]["explanation"]) < page.index("```text")
    in_code = False
    for line in page.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if line == "$$":
            assert not in_code
    assert not in_code
    assert page.rsplit("\n## ", 1)[1].startswith("References\n")
    assert publication["rows"] == recorded


def test_accessible_illustration_opens_overview_and_binds_its_bytes(documented):
    root, publication = documented
    _, data, asset = _add_math_and_image(root)
    family = refresh(root, publication)
    page = generated_pages(root, publication)[root / family["page"]]
    assert "![Critic \\[input\\] gradient branches](assets/bcap-explainer.png)" in page
    assert page.index(data["overview"]) < page.index("![") < page.index("## Mathematical formulation")
    assert data["illustration"]["caption"] in page
    name = data["illustration"]["path"]
    original = publication["family_progress"]["input_hashes"][name]
    assert original == file_hash(asset)
    asset.write_bytes(asset.read_bytes() + b"editorial fixture revision")
    refresh(root, publication)
    assert publication["family_progress"]["input_hashes"][name] == file_hash(asset) != original


@pytest.mark.parametrize("equations", [None, {}, [{"label": "Objective", "latex": "x"}],
                                      [{"label": "", "latex": "x", "explanation": "Meaning."}],
                                      [{"label": "Objective", "latex": " ", "explanation": "Meaning."}],
                                      [{"label": "Objective", "latex": "x", "explanation": ""}]])
def test_equations_require_nonempty_labels_bodies_and_explanations(documented, equations):
    root, _ = documented
    data = read_json(root / "configs/forge/family-documentation/bcap.json")
    data["equations"] = equations
    with pytest.raises(ValueError, match="equation"):
        validate_documentation(root, data, "bcap")


@pytest.mark.parametrize("latex", ["$x$", "$$x$$", r"\(x\)", r"\[x\]", "```math\nx\n```", "~~~math\nx\n~~~"])
def test_equation_bodies_do_not_break_the_generators_math_delimiters(documented, latex):
    root, _ = documented
    data = read_json(root / "configs/forge/family-documentation/bcap.json")
    data["equations"] = [{"label": "Objective", "latex": latex, "explanation": "Meaning."}]
    with pytest.raises(ValueError, match="omit math delimiters and Markdown fences"):
        validate_documentation(root, data, "bcap")


def test_aligned_multiline_latex_accepts_spacing_and_legacy_optional_absence(documented):
    root, publication = documented
    path, data, _ = _add_math_and_image(root)
    data["equations"][0]["latex"] = r"\begin{aligned} L_D &= a+b \\[3pt] L_G &= c \end{aligned}"
    validate_documentation(root, data, "bcap")
    data.pop("illustration")
    data["equations"] = []
    atomic_json(path, data)
    family = refresh(root, publication)
    page = generated_pages(root, publication)[root / family["page"]]
    assert "## Mathematical formulation" not in page and "![" not in page
    assert data["symbols"] in page


@pytest.mark.parametrize("change, error", [
    ("missing_asset", "existing repository asset"), ("outside_repo", "existing repository asset"),
    ("remote_asset", "existing repository asset"), ("unsupported_format", "PNG, JPEG, GIF or WebP"),
    ("missing_alt", "repository path, alt text and caption"), ("empty_alt", "nonempty text"),
    ("empty_caption", "nonempty text"),
])
def test_illustrations_require_local_raster_assets_with_accessible_descriptions(documented, change, error):
    root, _ = documented
    _, data, asset = _add_math_and_image(root)
    if change == "missing_asset":
        asset.unlink()
    elif change == "outside_repo":
        data["illustration"]["path"] = "../elsewhere.png"
    elif change == "remote_asset":
        data["illustration"]["path"] = "https://example.com/image.png"
    elif change == "unsupported_format":
        asset.rename(asset.with_suffix(".svg"))
        data["illustration"]["path"] = data["illustration"]["path"].replace(".png", ".svg")
    elif change == "missing_alt":
        data["illustration"].pop("alt")
    else:
        data["illustration"]["alt" if change == "empty_alt" else "caption"] = " "
    with pytest.raises(ValueError, match=error):
        validate_documentation(root, data, "bcap")
