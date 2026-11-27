"""Frozen architecture/initialization transfer for opt-in native tasks.

The original MLP tasks are unchanged. This profile borrows a declared affine
architecture and resources, not historical seeds, recipe settings or results.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .contracts import canonical, identifier, stable_hash
from .initialization import task_initializer

ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "native_affine_square_named_v1"
PROFILE_SOURCE = "configs/toy100/constraints_simple_regularization.json"
PROFILE_SHA256 = "4af9863a319378b362bfb925b161d9ae8b8b07c9ecf1a452bb645570e04b99b7"
RELEASE_PROFILE_ID = "release07_public_default_named_v1"
RELEASE_PROFILE_SOURCE = "configs/toy100/release07_public_default_host.json"
RELEASE_PROFILE_SHA256 = "aa6c8d2ab22682dfeae90e829d58c0aab8a0626aa39722cbde12156e7569894e"
COMPONENTS = {"generator", "discriminator", "prior"}
PLAIN_NATIVE_MODEL = {"hidden": 128, "layers": 3, "fourier": 2}
DECLARED_DEEP_MLP16_MODEL = {"hidden": 128, "layers": 16, "fourier": 2}
DECLARED_RES16_TANH_MODEL = {
    "hidden": 128,
    "layers": 16,
    "architecture": "residual_mlp",
    "output_activation": "tanh",
    "tanh_scale": 5.25,
    "fourier": 2,
}
DECLARED_SCALED_RES16_TANH_MODEL = {
    "hidden": 128,
    "layers": 16,
    "architecture": "scaled_residual_mlp",
    "output_activation": "tanh_rail_guard",
    "tanh_scale": 5.25,
    "rail_guard": 1e-4,
    "fourier": 3,
}
DECLARED_SCALED_RES16_TANH_V2_MODEL = {
    "hidden": 128,
    "layers": 16,
    "architecture": "scaled_residual_mlp",
    "output_activation": "tanh_rail_guard",
    "tanh_scale": 5.25,
    "rail_guard": 1e-4,
    "fourier": 3,
    "preserve_latent_norm": False,
}
DECLARED_NATIVE_MODELS = (
    PLAIN_NATIVE_MODEL,
    DECLARED_DEEP_MLP16_MODEL,
    DECLARED_RES16_TANH_MODEL,
    DECLARED_SCALED_RES16_TANH_MODEL,
    DECLARED_SCALED_RES16_TANH_V2_MODEL,
)


def profile_declaration(profile_id=PROFILE_ID):
    if profile_id == RELEASE_PROFILE_ID:
        return {"id": profile_id, "revision": 1,
                "source": {"path": RELEASE_PROFILE_SOURCE, "sha256": RELEASE_PROFILE_SHA256}}
    if profile_id != PROFILE_ID:
        raise ValueError("unsupported native profile")
    return {"id": PROFILE_ID, "revision": 1, "source": {"path": PROFILE_SOURCE, "sha256": PROFILE_SHA256}}


def _profile_id(task):
    declaration = task["execution"]["native_profile"]
    profile_id = declaration.get("id")
    if canonical(declaration) != canonical(profile_declaration(profile_id)):
        raise ValueError("unsupported or mismatched native profile declaration")
    return profile_id


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def resolve_host_initialization(value, *, initializer="deterministic_orthogonal", prior=None, requirements=None):
    """Validate task-owned component policies before construction or RNG draws.

    Requirements are a future explicit API constraint, not the legacy fallback
    initializer string. No new idea-file field is implicitly accepted here.
    """
    if requirements is not None and not isinstance(requirements, dict):
        raise ValueError("invalid explicit component initializer requirements")
    if value is None:
        if requirements:
            raise ValueError("explicit component initializer requirements need a declared host policy")
        return None
    if not isinstance(value, dict) or set(value) != {"schema_version", "owner", "profile_sha256", "components"}:
        raise ValueError("host initialization has missing or unsupported fields")
    digest = value["profile_sha256"]
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["owner"] != "task"
            or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("invalid host initialization identity")
    policies = value["components"]
    if not isinstance(policies, dict) or not policies or not set(policies) <= COMPONENTS:
        raise ValueError("host initialization needs known component policies")
    if initializer == "supplied":
        raise ValueError("supplied initialization conflicts with task-pinned component policies")
    if requirements is not None and (not isinstance(requirements, dict) or not set(requirements) <= COMPONENTS):
        raise ValueError("invalid explicit component initializer requirements")
    for component, policy in policies.items():
        if not isinstance(policy, dict):
            raise ValueError("component initializer must be a declaration, not a callback")
        method = policy.get("method")
        if method == "identity_linear_v1":
            if set(policy) != {"method"}:
                raise ValueError("identity initializer has unsupported fields")
        elif method == "xavier_uniform_zero_bias_v1":
            if set(policy) != {"method", "gain"} or not _positive(policy["gain"]):
                raise ValueError("Xavier initializer requires a positive finite gain")
        elif method == "sample_distributions_v1":
            parameters = policy.get("parameters")
            if set(policy) != {"method", "parameters"} or not isinstance(parameters, dict) or not parameters:
                raise ValueError("sample initializer requires explicit parameter distributions")
            for name, distribution in parameters.items():
                if not isinstance(name, str) or not name or not isinstance(distribution, dict):
                    raise ValueError("invalid parameter distribution declaration")
                kind = distribution.get("kind")
                expected = {"uniform": {"kind", "low", "high"}, "normal": {"kind", "mean", "std"}, "keep": {"kind"}}
                if kind not in expected or set(distribution) != expected[kind]:
                    raise ValueError("unsupported literal parameter distribution")
                numeric = [v for k, v in distribution.items() if k != "kind"]
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in numeric):
                    raise ValueError("distribution bounds must be finite numbers")
                if kind == "uniform" and distribution["low"] >= distribution["high"]:
                    raise ValueError("uniform initialization requires low < high")
                if kind == "normal" and distribution["std"] <= 0:
                    raise ValueError("normal initialization requires positive std")
        else:
            raise ValueError("unsupported component initialization method")
        if component == "prior" and (method != "sample_distributions_v1" or set(policy["parameters"]) != {"z"}):
            raise ValueError("public prior initialization requires an explicit z distribution")
        if component in (requirements or {}) and canonical(requirements[component]) != canonical(policy):
            raise ValueError(f"candidate initializer requirement conflicts with pinned {component} policy")
    if set(requirements or {}) - set(policies):
        raise ValueError("required component initializer is not pinned by this host")
    if "prior" in policies and "init_std" in (prior or {}):
        raise ValueError("explicit prior.init_std conflicts with task-pinned location initialization")
    return deepcopy(value)


def _profile_spec(root, profile_id=PROFILE_ID):
    if profile_id == RELEASE_PROFILE_ID:
        data = (Path(root) / RELEASE_PROFILE_SOURCE).read_bytes()
        if hashlib.sha256(data).hexdigest() != RELEASE_PROFILE_SHA256:
            raise ValueError("release host source changed; declare a new profile revision")
        return json.loads(data)["host_definition"]
    data = (Path(root) / PROFILE_SOURCE).read_bytes()
    if hashlib.sha256(data).hexdigest() != PROFILE_SHA256:
        raise ValueError("native profile source changed; declare a new profile revision")
    source = json.loads(data)
    if source["toy100_model"] != "affine_square_v1" or source["z_dim"] != 2:
        raise ValueError("native source no longer declares the selected affine host")
    return {"generator": {"kind": "linear", "in_features": source["z_dim"], "out_features": 2, "bias": True},
        "discriminator": {"kind": "simple_mlp", "in_dim": 2, "hidden_dim": source["d_hidden"],
                          "n_hidden": source["n_hidden"], "fourier": source["fourier"]},
        "resources": {key: source[key] for key in ("z_dim", "num_particles", "batch_size")},
        "initialization": {"generator": {"method": "identity_linear_v1"},
            "discriminator": {"method": "xavier_uniform_zero_bias_v1", "gain": 1.0},
            "prior": {"method": "sample_distributions_v1", "parameters": {"z": {"kind": "uniform", "low": -5., "high": 5.}}}}}


def _measurement(task):
    from benchmarks.toy100.accuracy import LIMITS
    from benchmarks.toy100.metrics import REQUIREMENTS
    from .sampling import PUBLIC_PRIOR_CLEAN, executed_receipt
    expected = {"kind": "native_accuracy", "scoring_weights": "live", "eval_interval": 250,
        "early_eval_steps": [0, 1, 10, 25, 50, 100], "eval_samples": 20000, "holdout_samples": 100000,
        "minimum_stable_checks": 5, "coverage_thresholds": REQUIREMENTS, "accuracy_limits": LIMITS,
        "evaluator": "benchmarks.toy100.accuracy_gate:evaluate_suite",
        "coverage_evaluator": "benchmarks.toy100.gate:evaluate_suite"}
    expected.update(executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean"))
    if any(canonical(task["evaluation"].get(key)) != canonical(value) for key, value in expected.items()):
        raise ValueError("native profile must retain the full frozen clean live coverage and accuracy protocol")


def resolve_native_spec(task, *, root=None):
    """Return a validated explicit profile, or None for an unchanged old task."""
    execution = task["execution"]
    if "native_profile" not in execution:
        if task.get("adapter") in {"native100", "native100_continuation"}:
            if {"host_definition", "host_initialization", "initialization"} & set(execution):
                raise ValueError("native model overrides require an explicit profile")
            # Tasks predating explicit ownership keep the old source fallback.
            # Current declarations freeze the existing MLP and resource values.
            if "initializer" in execution or {"model", "resources"} & set(execution):
                model_card = execution.get("model")
                if not any(canonical(model_card) == canonical(allowed) for allowed in DECLARED_NATIVE_MODELS):
                    raise ValueError("native model variants require an explicit profile")
                resources = execution.get("resources")
                if (not isinstance(resources, dict) or set(resources) != {"num_particles", "z_dim", "batch_size"}
                        or any(type(value) is not int or value <= 0 for value in resources.values())):
                    raise ValueError("native resources must explicitly declare positive num_particles, z_dim and batch_size")
        return None
    if task.get("adapter") not in {"native100", "native100_continuation"}:
        raise ValueError("native profiles apply only to native tasks")
    profile_id = _profile_id(task)
    if "host_initialization" in execution:
        raise ValueError("native component policies belong in the frozen host_definition, not a second override")
    if {"resources", "initialization"} & set(execution):
        raise ValueError("native resource and initializer fields belong in the frozen host_definition")
    expected = _profile_spec(ROOT if root is None else root, profile_id)
    if canonical(execution.get("host_definition")) != canonical(expected):
        raise ValueError("native host card differs from its frozen profile")
    if execution.get("problem") not in {"grid100", "rotated100", "staggered100"}:
        raise ValueError("unsupported native profile problem")
    continuation = task["adapter"] == "native100_continuation"
    if profile_id == RELEASE_PROFILE_ID and continuation:
        raise ValueError("release default host has no registered continuation")
    if execution.get("steps") != (14000 if continuation else 7000):
        raise ValueError("native profile must retain its complete qualification horizon")
    if continuation and any(execution.get(key) != value for key, value in {
            "incremental_steps": 7000, "preserve_prefix_steps": 7000, "original_schedule_horizon": 7000}.items()):
        raise ValueError("native continuation must preserve its original 7k schedule and prefix")
    if execution.get("original_schedule_horizon", 7000) != 7000:
        raise ValueError("native profile schedule horizon must remain 7000")
    _measurement(task)
    return deepcopy(expected)


def native_profile_source_files(task):
    if "native_profile" not in task.get("execution", {}):
        return []
    return [profile_declaration(_profile_id(task))["source"]["path"]]


def native_host_initialization(task, *, root=None):
    spec = resolve_native_spec(task, root=root)
    if spec is None:
        return None
    # No task ID/problem/observation/budget fields: 7k and its 14k own-state
    # continuation must construct precisely the same initial static context.
    return {"schema_version": 1, "owner": "task", "profile_sha256": stable_hash({
        "native_profile": task["execution"]["native_profile"], "host_definition": spec}),
        "components": deepcopy(spec["initialization"])}


def native_profile_blockers(task, candidate, *, root=None, explicit_initializer=True):
    try:
        initializer = task_initializer(task, candidate, explicit=explicit_initializer)
        spec = resolve_native_spec(task, root=root)
        if spec is None:
            return []
        release = _profile_id(task) == RELEASE_PROFILE_ID
        from .priors import task_prior
        from .priors import recipe_owned_prior
        prior = task_prior(task)
        learned_ok = recipe_owned_prior(task) or prior.get("learnable") is True
        if release:
            if (prior.get("kind") != "particle_cloud" or prior.get("sigma") != 0
                    or prior.get("standardize") is not False or not learned_ok
                    or not isinstance(prior.get("exception_reason"), str) or not prior["exception_reason"].strip()
                    or set(prior) - {"kind", "sigma", "standardize", "learnable", "exception_reason"}):
                raise ValueError("release host requires its explicit unstandardized learned particle-cloud exception")
        elif (not isinstance(prior, dict) or prior.get("kind") != "mog" or not _positive(prior.get("sigma")) or prior.get("standardize") is not False
                or not learned_ok or set(prior) - {"kind", "sigma", "standardize", "learnable", "init_std"}):
            raise ValueError("native affine profile requires learned MoG locations, positive explicit width, no standardization and uniform masses")
        resolve_host_initialization(native_host_initialization(task, root=root), initializer=initializer, prior=prior)
        fixed = {**spec["resources"], "total_steps": 7000}
        for key, value in fixed.items():
            if key in candidate.get("recipe_overrides", {}) and candidate["recipe_overrides"][key] != value:
                raise ValueError(f"candidate overrides frozen native host resource {key}")
    except (KeyError, TypeError, ValueError, OSError) as error:
        return [f"{task.get('id', '<task>')}: {error}"]
    return []


def validate_native_continuation(parent, child, *, root=None):
    a, b = resolve_native_spec(parent, root=root), resolve_native_spec(child, root=root)
    if a is None and b is None:
        for field in ("initializer", "resources", "model"):
            if canonical(parent["execution"].get(field)) != canonical(child["execution"].get(field)):
                raise ValueError(f"native continuation changes task-owned {field}")
        return
    if (a is None or b is None or canonical(a) != canonical(b)
            or parent["adapter"] != "native100" or child["adapter"] != "native100_continuation"
            or parent["execution"]["problem"] != child["execution"]["problem"]
            or child["execution"].get("continuation_of") != parent["id"]
            or {"task": parent["id"], "kind": "checkpoint"} not in child.get("dependencies", [])):
        raise ValueError("native continuation must bind its matching profile and own 7k parent")
    if native_host_initialization(parent, root=root) != native_host_initialization(child, root=root):
        raise ValueError("native continuation changes host initialization identity")
    if parent["execution"].get("initializer") != child["execution"].get("initializer"):
        raise ValueError("native continuation changes task initializer")


def task_from_profile(base_task, task_id, *, parent_task_id=None, root=None):
    identifier(task_id, "native profile task id")
    if task_id == base_task["id"]:
        raise ValueError("native profile needs a distinct task id")
    task = deepcopy(base_task)
    task["id"] = task_id
    task["execution"].update(native_profile=profile_declaration(),
                             host_definition=_profile_spec(ROOT if root is None else root))
    task["execution"].pop("resources", None)
    task["execution"].pop("model", None)
    if task["adapter"] == "native100_continuation":
        if not parent_task_id:
            raise ValueError("profile continuation requires its explicit matching parent")
        old = task["execution"]["continuation_of"]
        task["execution"]["continuation_of"] = parent_task_id
        for dependency in task["dependencies"]:
            if dependency["task"] == old:
                dependency["task"] = parent_task_id
    resolve_native_spec(task, root=root)
    return task


def build_native_models(context, spec):
    """Use the pure shared model classes; context owns init, prior and trainer."""
    from benchmarks.toy100.native_models import native_component
    return tuple(context.construct(lambda role=role: native_component(spec[role], role=role), component=role).to(context.device)
                 for role in ("generator", "discriminator"))


def build_declared_native_models(context, model_spec):
    """Construct the declared native100 generator and 3-layer critic for published cards."""
    from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
    if canonical(model_spec) == canonical(PLAIN_NATIVE_MODEL):
        g = context.construct(
            lambda: SimpleMLPGenerator(context.recipe.z_dim, model_spec["hidden"], model_spec["layers"], 2),
            component="generator",
        )
        d = context.construct(
            lambda: SimpleMLPDiscriminator(2, model_spec["hidden"], model_spec["layers"], model_spec["fourier"]),
            component="discriminator",
        )
        return g.to(context.device), d.to(context.device)
    if canonical(model_spec) == canonical(DECLARED_DEEP_MLP16_MODEL):
        g = context.construct(
            lambda: SimpleMLPGenerator(context.recipe.z_dim, model_spec["hidden"], model_spec["layers"], 2),
            component="generator",
        )
        d = context.construct(
            lambda: SimpleMLPDiscriminator(2, model_spec["hidden"], 3, model_spec["fourier"]),
            component="discriminator",
        )
        return g.to(context.device), d.to(context.device)
    if canonical(model_spec) == canonical(DECLARED_RES16_TANH_MODEL):
        from benchmarks.toy_audit.api_vectors import ResidualMLPGenerator
        n_blocks = (int(model_spec["layers"]) - 2) // 2
        scale = float(model_spec["tanh_scale"])
        g = context.construct(
            lambda: ResidualMLPGenerator(context.recipe.z_dim, model_spec["hidden"], n_blocks, 2, scale=scale),
            component="generator",
        )
        d = context.construct(
            lambda: SimpleMLPDiscriminator(2, model_spec["hidden"], 3, model_spec["fourier"]),
            component="discriminator",
        )
        return g.to(context.device), d.to(context.device)
    if canonical(model_spec) in (
        canonical(DECLARED_SCALED_RES16_TANH_MODEL),
        canonical(DECLARED_SCALED_RES16_TANH_V2_MODEL),
    ):
        from benchmarks.toy_audit.api_vectors import ScaledResidualMLPGenerator
        n_blocks = (int(model_spec["layers"]) - 2) // 2
        scale = float(model_spec["tanh_scale"])
        # Legacy grid100_res16_scaled_tanh omits preserve_latent_norm and stays
        # frozen at True for archived receipt compatibility; v2 sets False explicitly.
        preserve_latent_norm = bool(model_spec.get("preserve_latent_norm", True))
        g = context.construct(
            lambda: ScaledResidualMLPGenerator(
                context.recipe.z_dim,
                model_spec["hidden"],
                n_blocks,
                2,
                scale=scale,
                use_norm=True,
                preserve_latent_norm=preserve_latent_norm,
            ),
            component="generator",
        )
        d = context.construct(
            lambda: SimpleMLPDiscriminator(2, model_spec["hidden"], 3, model_spec["fourier"]),
            component="discriminator",
        )
        return g.to(context.device), d.to(context.device)
    raise ValueError("native model variants require an explicit profile")
