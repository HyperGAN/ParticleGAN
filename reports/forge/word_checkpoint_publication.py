"""Bridge frozen word and Gaussian graders' own-state proofs into board aggregation.

This display adapter preserves task declarations and frozen numerical grading.
It verifies the selected parent checkpoint before bypassing the older generic
requirement for an uninterrupted run. No receipt is modified or re-executed.
"""
from copy import deepcopy


def verified_parent(child, parent):
    evidence = child.get("evidence") or {}
    proof = evidence.get("continuity") or {}
    checkpoint = (parent.get("evidence") or {}).get("checkpoint") or {}
    return bool(checkpoint.get("sha256") and checkpoint.get("state_sha256")
                and proof.get("restored_exactly") is True
                and proof.get("history_reset") is False
                and proof.get("same_recipe_prior_architecture") is True
                and proof.get("parent_smoke_status") == "PASS"
                and proof.get("parent_checkpoint_sha256") == checkpoint["sha256"]
                and proof.get("parent_state_sha256") == checkpoint["state_sha256"]
                and proof.get("parent_compatibility_key") == parent.get("compatibility_key")
                and proof.get("parent_confirmed_step") == checkpoint.get("completed_steps")
                and proof.get("prefix_steps") == checkpoint.get("completed_steps"))


def verified_gaussian_parent(child, parent):
    evidence = child.get("evidence") or {}
    proof = evidence.get("continuity") or {}
    parent_evidence = parent.get("evidence") or {}
    checkpoint = parent_evidence.get("checkpoint") or {}
    return bool(checkpoint.get("sha256") and checkpoint.get("state_sha256")
                and proof.get("restored_exactly") is True
                and proof.get("history_reset") is False
                and proof.get("diagnostic_continuation") is False
                and proof.get("parent_smoke_status") == "PASS"
                and proof.get("parent_checkpoint_sha256") == checkpoint["sha256"]
                and proof.get("parent_state_sha256") == checkpoint["state_sha256"]
                and proof.get("parent_compatibility_key") == parent.get("compatibility_key")
                and proof.get("prefix_steps") == parent_evidence.get("completed_steps"))


_GAUSSIAN_PAIRS = (
    ("gaussian1d_smoke", "gaussian1d_stability"),
    ("gaussian1d_shallow_smoke", "gaussian1d_shallow_stability"),
)


def install():
    from experiments.forge import views
    if getattr(views.qualify, "_word_checkpoint_publication", False):
        return
    original = views.qualify

    def qualify(view, tasks, results, *, candidate=None, **kwargs):
        safe = deepcopy(results)
        for row in safe:
            if isinstance(row.get("evidence"), dict) and row["evidence"].get("continuity") is None:
                row["evidence"].pop("continuity", None)
        parents = [row for row in safe if row.get("task_id") == "five_word_joint_smoke"]
        children = [row for row in safe if row.get("task_id") == "five_word_joint_hold"]
        supported = bool(len(parents) == 1 and children
                         and all(verified_parent(child, parents[0]) for child in children))
        supported_gaussian = set()
        for smoke_id, stability_id in _GAUSSIAN_PAIRS:
            g_parents = [row for row in safe if row.get("task_id") == smoke_id]
            g_children = [row for row in safe if row.get("task_id") == stability_id]
            if len(g_parents) == 1 and g_children and all(
                verified_gaussian_parent(child, g_parents[0]) for child in g_children
            ):
                supported_gaussian.add(smoke_id)
        dependencies = views._dependencies

        def declared_dependencies(task):
            rows = dependencies(task)
            kind = task.get("evaluation", {}).get("kind")
            if supported and kind == "word_hold":
                return [{**row, "kind": "gate"} if row == {"task": "five_word_joint_smoke", "kind": "checkpoint"}
                        else row for row in rows]
            if supported_gaussian and kind == "gaussian_stability":
                return [{**row, "kind": "gate"}
                        if row.get("kind") == "checkpoint" and row.get("task") in supported_gaussian
                        else row for row in rows]
            return rows

        views._dependencies = declared_dependencies
        try:
            return original(view, tasks, safe, candidate=candidate, **kwargs)
        finally:
            views._dependencies = dependencies

    qualify._word_checkpoint_publication = True
    views.qualify = qualify
