"""Maintained, descriptive family metadata; never a scientific grading input."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from urllib.parse import urlsplit

from .contracts import identifier, read_json
from .laws_registry import VALID_MECHANISM_STATUSES, load_laws


DOCUMENTATION_DIRECTORY = Path("configs/forge/family-documentation")
TAG_VOCABULARY = Path("configs/forge/family-tags.json")
FAMILY_REGISTRY = Path("configs/forge/trainer-families.json")
TRAINING_DETAILS = (
    ("loss", "Adversarial loss"),
    ("optimizer", "Optimizer"),
    ("learning_rates", "Learning rates and annealing"),
    ("gradient_clipping", "Parameter-gradient clipping"),
    ("critic_penalty", "Critic penalties and anchors"),
    ("damping_and_guards", "Damping and update guards"),
    ("noise", "Training and sampling noise"),
    ("averaging", "Parameter averaging and serving"),
)
_TAG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_LAW_ID = re.compile(r"[a-z][a-z0-9_-]*")
_PINNED_GITHUB_SOURCE = re.compile(r"/[^/]+/[^/]+/blob/[0-9a-f]{40}/.+")
_RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_MERMAID_HEADERS = (
    "flowchart ",
    "flowchart\n",
    "graph ",
    "graph\n",
    "sequenceDiagram",
    "stateDiagram",
    "classDiagram",
    "erDiagram",
    "xychart-beta",
)


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _texts(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    for item in value:
        _text(item, label)
    return value


def _https_url(value, label):
    _text(value, label)
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password
            or any(character.isspace() for character in value)):
        raise ValueError(f"{label} must be an HTTPS URL")
    return parsed


def _illustration(root, value):
    if not isinstance(value, dict) or set(value) != {"path", "alt", "caption"}:
        raise ValueError("illustration needs a repository path, alt text and caption")
    for key in ("path", "alt", "caption"):
        _text(value[key], "illustration " + key)
    path = Path(value["path"])
    resolved = (root / path).resolve()
    if (path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(root.resolve())
            or not resolved.is_file()):
        raise ValueError("illustration must be an existing repository asset")
    if path.suffix.lower() not in _RASTER_EXTENSIONS:
        raise ValueError("illustration must use PNG, JPEG, GIF or WebP")


def _law_mechanisms(root: Path, value, family_id: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"law_mechanisms for {family_id} must be a nonempty mapping")
    laws_doc = load_laws(root)
    valid_ids = {law["id"] for law in laws_doc["laws"]} if laws_doc is not None else None
    entry_fields = {"status", "mechanism", "how_it_works", "known_limits"}
    valid_statuses = set(VALID_MECHANISM_STATUSES)
    for law_id, entry in value.items():
        if not isinstance(law_id, str) or not _LAW_ID.fullmatch(law_id):
            raise ValueError(f"law_mechanisms for {family_id} has invalid law id {law_id!r}")
        if valid_ids is not None and law_id not in valid_ids:
            raise ValueError(
                f"law_mechanisms for {family_id} references unknown law id {law_id!r}; "
                f"expected one of {sorted(valid_ids)}"
            )
        if not isinstance(entry, dict) or set(entry) != entry_fields:
            raise ValueError(
                f"law_mechanisms[{law_id!r}] for {family_id} needs exact fields {sorted(entry_fields)}"
            )
        status = _text(entry["status"], f"law_mechanisms[{law_id!r}].status")
        if status not in valid_statuses:
            raise ValueError(
                f"law_mechanisms[{law_id!r}].status for {family_id} must be one of {sorted(valid_statuses)}"
            )
        for key in ("mechanism", "how_it_works", "known_limits"):
            _text(entry[key], f"law_mechanisms[{law_id!r}].{key}")


def _mermaid_diagram(value, family_id: str) -> None:
    text = _text(value, f"mermaid_diagram for {family_id}")
    if "```" in text or "~~~" in text:
        raise ValueError(f"mermaid_diagram for {family_id} must omit Markdown code fences")
    header = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue
        header = stripped
        break
    if header is None or not any(
        header == prefix.strip() or header.startswith(prefix) for prefix in _MERMAID_HEADERS
    ):
        raise ValueError(
            f"mermaid_diagram for {family_id} must begin with a supported Mermaid diagram declaration"
        )


def _quickstart(value, family_id: str) -> None:
    if not isinstance(value, dict) or set(value) != {"preset", "snippet"}:
        raise ValueError(f"quickstart for {family_id} needs exact fields ['preset', 'snippet']")
    _text(value["preset"], f"quickstart.preset for {family_id}")
    snippet = value["snippet"]
    if (not isinstance(snippet, list) or not snippet
            or not all(isinstance(line, str) for line in snippet)
            or not any(line.strip() for line in snippet)):
        raise ValueError(f"quickstart.snippet for {family_id} must be a nonempty list of code lines")
    if any("```" in line or "~~~" in line for line in snippet):
        raise ValueError(f"quickstart.snippet for {family_id} cannot contain Markdown code fences")


def validate_documentation(root: Path, value: dict, family_id: str) -> dict:
    """Validate maintained prose and resolvable source navigation, not its claims."""
    if not isinstance(value, dict):
        raise ValueError(f"family documentation for {family_id} must be an object")
    required = {"overview", "symbols", "pseudocode", "training_details", "configuration_notes",
                "references", "sources"}
    optional = {"equations", "illustration", "law_mechanisms", "mermaid_diagram", "quickstart"}
    if required - set(value) or set(value) - required - optional:
        raise ValueError(f"family documentation for {family_id} needs exactly the required fields {sorted(required)} "
                         f"and optional {sorted(optional)}")
    _text(value["overview"], "technique overview")
    _text(value["symbols"], "pseudocode symbols")
    _texts(value["pseudocode"], "pseudocode")
    if any("```" in line for line in value["pseudocode"]):
        raise ValueError("pseudocode cannot contain Markdown code fences")
    if "equations" in value:
        if not isinstance(value["equations"], list):
            raise ValueError("equations must be a list")
        for equation in value["equations"]:
            if not isinstance(equation, dict) or set(equation) != {"label", "latex", "explanation"}:
                raise ValueError("equations need a label, LaTeX body and explanation")
            for key in ("label", "latex", "explanation"):
                _text(equation[key], "equation " + key)
            latex = equation["latex"]
            if (any(token in latex for token in ("$", "```", "~~~"))
                    or re.search(r"(?<!\\)\\[\[\]()]", latex)):
                raise ValueError("equation LaTeX must omit math delimiters and Markdown fences")
    if "illustration" in value:
        _illustration(root, value["illustration"])
    if "law_mechanisms" in value:
        _law_mechanisms(root, value["law_mechanisms"], family_id)
    if "mermaid_diagram" in value:
        _mermaid_diagram(value["mermaid_diagram"], family_id)
    if "quickstart" in value:
        _quickstart(value["quickstart"], family_id)
    _texts(value["configuration_notes"], "configuration notes")
    details = value["training_details"]
    if not isinstance(details, dict) or set(details) != {key for key, _ in TRAINING_DETAILS}:
        raise ValueError("training details need every declared training characteristic")
    for key, _ in TRAINING_DETAILS:
        _text(details[key], "training detail " + key)
    references = value["references"]
    if not isinstance(references, list):
        raise ValueError("references must be a list; use an empty list for repository-specific techniques")
    urls = []
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {"title", "url", "component"}:
            raise ValueError("references need a title, URL and supported component")
        _text(reference["title"], "reference title")
        _text(reference["component"], "reference component")
        _https_url(reference["url"], "reference URL")
        urls.append(reference["url"])
    if len(urls) != len(set(urls)):
        raise ValueError("references must not repeat a URL")
    for source in _texts(value["sources"], "implementation sources"):
        if "://" in source:
            parsed = _https_url(source, "implementation source")
            if (parsed.netloc != "github.com" or not _PINNED_GITHUB_SOURCE.fullmatch(parsed.path)
                    or parsed.query):
                raise ValueError("historical sources must use a commit-pinned GitHub blob URL")
        else:
            path = Path(source)
            resolved = (root / path).resolve()
            if (path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(root.resolve())
                    or not resolved.is_file()):
                raise ValueError(f"implementation source must be an existing repository file: {source}")
    if len(value["sources"]) != len(set(value["sources"])):
        raise ValueError("implementation sources must not repeat a path")
    return deepcopy(value)


def load_family_documentation(root: Path, family_ids, load) -> tuple[dict, dict]:
    """Load current and historical reports through the caller's provenance loader.

    Small historical/synthetic report roots predate editorial documentation. If
    the directory is absent, render an explicit unavailable notice. Once the
    directory exists, incomplete coverage is an error rather than a silent gap.
    """
    root = Path(root)
    if not (root / DOCUMENTATION_DIRECTORY).is_dir():
        return {}, {}
    if not (root / FAMILY_REGISTRY).is_file() or not (root / TAG_VOCABULARY).is_file():
        raise ValueError("family documentation needs its family registry and tag vocabulary")
    vocabulary = load(TAG_VOCABULARY)
    definitions = vocabulary.get("tags") if isinstance(vocabulary, dict) else None
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("family tag vocabulary needs nonempty tag definitions")
    for tag, definition in definitions.items():
        if not isinstance(tag, str) or not _TAG.fullmatch(tag):
            raise ValueError("family tags must be lowercase hyphenated names")
        _text(definition, "family tag definition")
    registry = load(FAMILY_REGISTRY)
    registered = {}
    for family in registry.get("families", []) + registry.get("historical_families", []):
        family_id = identifier(family.get("id"), "documented trainer family")
        if family_id in registered:
            raise ValueError("documented trainer family ids must be distinct")
        tags = family.get("tags")
        if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"family {family_id} needs a nonempty tag list")
        if len(tags) != len(set(tags)) or tags != sorted(tags):
            raise ValueError(f"family {family_id} tags must be unique and lexically sorted")
        if any(not _TAG.fullmatch(tag) for tag in tags):
            raise ValueError("family tags must be lowercase hyphenated names")
        if set(tags) - definitions.keys():
            raise ValueError(f"family {family_id} has undefined tags: {sorted(set(tags) - definitions.keys())}")
        registered[family_id] = family
    documentation = {}
    for family_id in family_ids:
        identifier(family_id, "documented trainer family")
        path = DOCUMENTATION_DIRECTORY / (family_id + ".json")
        if family_id not in registered or not (root / path).is_file():
            raise ValueError(f"missing registered family documentation for {family_id}")
        documentation[family_id] = {
            "tags": list(registered[family_id]["tags"]),
            "documentation": validate_documentation(root, load(path), family_id),
        }
    return documentation, dict(sorted(definitions.items()))


def validate_family_documentation(root: Path) -> dict:
    """Validate all registered editorial metadata without rendering or training."""
    root = Path(root)
    if not (root / DOCUMENTATION_DIRECTORY).is_dir():
        return {"status": "unavailable", "families": 0, "tags": 0}
    if not (root / FAMILY_REGISTRY).is_file():
        raise ValueError("family documentation needs its family registry and tag vocabulary")
    registry = read_json(root / FAMILY_REGISTRY)
    families = [family["id"] for family in registry.get("families", []) + registry.get("historical_families", [])]
    documentation, tags = load_family_documentation(root, families, lambda path: read_json(root / path))
    return {"status": "valid", "families": len(documentation), "tags": len(tags)}
