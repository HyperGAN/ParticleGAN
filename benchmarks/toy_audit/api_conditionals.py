"""Current public-API conditional toys with frozen, question-specific gates.

These are new hosts/protocols, not revisions of archived qualifications. The
stochastic hosts fit a clean conditional kernel directly (not a DDGAN chain).
The paired hosts fit error against noise through the public RpGAN/KA2 factories.
No oracle prediction enters a generator and no pass is assumed from its recipe.
"""
from __future__ import annotations

from .reproducibility import DEFAULT_SEED, construction_rng

from copy import deepcopy
import math
from functools import wraps

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from particlegan import MoGParticlePrior, ParticlePrior, get_recipe, init, scale_learning_rates
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.priors import ManipulatedPrior, NormalNoiseManipulator, unwrap_prior
from lib.sparse_toy import SparseMixedToy
from lib.denoising_toy import GaussianGrid
from lib.trajectory import Routes
from lib.transition import Transitions
from benchmarks.toy_audit import source_conditional_scoring as density
from benchmarks.toy_audit.api_sources import circle, sprite, misgan

VERSION = "conditional-api-v1"
MISGAN_VERSION = "misgan-api-v2-formula-faithful"
EVAL_SEED = 99123


def _definition(name, legacy, title, goal, steps, *, kind="paired", thresholds=None,
                scope=None, sampling=None):
    return dict(id="api-" + name, legacy_ids=[legacy], title=title, goal=goal,
        kind=kind, default_steps=steps, batch_size=64, eval_samples=1024,
        default_recipe="ka2", thresholds=thresholds or {"relative_mse_max": .10},
        sampling=sampling or "private named training stream; independent fixed heldout rows",
        scope=scope or "New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged.")


_CASES = [
    _definition("sparse-identity", "source-family-00", "Sparse identity symbols", "Recover the joint class, sparse mode and matching symbol law", 8000, kind="conditional_distribution",
        thresholds=density.GATES, sampling="8 uniform classes, 8 uniform modes/class, 3 active coordinates; Gaussian .05 only on active support; identity symbol",
        scope="Direct clean joint conditional kernel; no diffusion-chain or historical source convergence claim."),
    _definition("sparse-split", "source-family-01", "Sparse split symbols", "Recover both valid symbols per class and their pairing with each sparse mode", 8000, kind="conditional_distribution",
        thresholds=density.GATES, sampling="Same fixed 64-mode sparse bank; split symbol 2*class+(mode//8)%2",
        scope="Direct clean joint conditional kernel with 16 symbols; no diffusion-chain claim."),
    *[_definition(f"posterior-{classes}class", f"source-family-{2 if classes == 1 else 3:02}", f"Analytic posterior, {classes} class(es)", "Sample the full clean conditional posterior, including its mode masses and local variance", 7000, kind="conditional_distribution",
        thresholds={"quality_mass_tv_max": .10, "hq_min": .95, "shape": density.GATES},
        sampling="Exact GaussianGrid posterior at class/observation/alpha_bar; heldout zero and displaced observations at alpha_bar .5 and .9",
        scope="Direct posterior sampler over four fixed observation/time panels per class; not iterative denoising or all possible observations.") for classes in (1, 4)],
    *[_definition(f"routes-{mode}", f"source-family-{4 if mode == 'discrete' else 5:02}", f"Two-route sequence ({mode} exposure)", "Cover both class-weighted obstacle routes at unseen geometries with genuine within-route variation", 28000, kind="conditional_distribution",
        thresholds={"valid_fraction_min": .95, "quality_route_mass_tv_max": .10, "coefficient_support_min": .95,
                    "coefficient_mean_abs_max": .10, "coefficient_variance": [.22, .45], "coefficient_ks_max": .10},
        sampling=f"Original Routes conditional kernel; {mode} training geometries, four unseen test geometries, p(upper|class)=(.8,.3), 3 independent U[-1,1] coefficients",
        scope="Direct complete 64-point trajectory generator; class and geometry only, never route/coefficients as input.") for mode in ("discrete", "continuous")],
    *[_definition(f"transitions-{mode}", f"source-family-{6 if mode == 'discrete' else 7:02}", f"Local route transitions ({mode} exposure)", "Fit a coherent joint state/action/successor law conditional on class, geometry and tick", 28000, kind="conditional_distribution",
        thresholds={"joint_consistency_rms_max": .002, "max_projection_cdf_error_max": .12},
        sampling=f"Original Transitions 6D kernel, {mode} training contexts and uniform ticks 0..62; heldout geometry with ticks 0,8,31,54,62",
        scope="Direct joint conditional transition kernel; five frozen time panels include both boundaries. This is not a policy or full-trajectory rollout.") for mode in ("discrete", "continuous")],
    *[_definition(f"transport-{task}", f"source-family-{8 if task == 'affine2' else 9:02}", f"Paired {task} transport", "Recover the correct source-to-target map rather than just the target marginal", 6000,
        thresholds={"normalized_mse_max": .01, "p95_l2_max": .20}, sampling="Independent uniform square [-sqrt(3),sqrt(3)]^2; affine rotation/translation or radius-dependent 1.7-radian swirl",
        scope="New direct conditional MLP and public paired-error game; target map and heldout pairing retained, archived dense adapter recipe not replayed.") for task in ("affine2", "swirl2")],
    _definition("trajectory-edit", "develop-trajectory", "Paired slow-to-fast trajectory edit", "Change angular speed while preserving each trajectory's radius and starting phase", 400,
        thresholds={"paired_mse_max": .02}, sampling="Original 12 phases, radii .45/.7/.95, 8 frames; slow .45 to fast 2.2 angular speed"),
    _definition("trajectory-residual", "develop-residual_student", "Residual trajectory edit with identity hold", "Recover the fast trajectory with a residual head and reject a correct marginal with wrong identities", 400,
        thresholds={"paired_mse_max": .02, "wrong_pair_margin_min": .02}, sampling="Same 12 exact pairs as trajectory-edit; residual output is added to its own slow input"),
    _definition("unipolar-hold", "develop-unipolar", "Positive edit with a learned neutral hold", "Make the positive 4D edit while holding the free scale-zero origin", 400,
        thresholds={"cover_min": .85, "off_axis_max": .05, "neutral_hold_min": .85}, sampling="Scale 0/1 with equal mass; plus=[1,0,0,0], free trainable neutral output"),
    _definition("unused-token-hold", "develop-unused_token_hold", "Unused slot hold during concept motion", "Move the concept slot on its target axis while keeping the unused slot fixed", 200,
        thresholds={"concept_move_min": .85, "unused_hold_min": .85}, sampling="Two slots; unused [1,0], concept [0,0]; concept target [0,1]; shared residual plus per-slot correction"),
    _definition("guarded-leftover", "develop-cover_leftover", "Guarded bipolar edit and content", "Cover both signed poles while preserving content and removing the guarded leak", 800,
        thresholds={"pole_relative_l2_max": .10, "leak_ratio_max": .10}, sampling="Original LeftoverField .55 content, .45 leak; faithful_guard_e target at -1,0,+1"),
    _definition("midscale-identity", "develop-mid_scale_identity", "Intermediate identity and concept control", "Retain identity at half strength in addition to correct neutral and signed poles", 800,
        thresholds={"relative_mse_max": .10, "max_context_l2_max": .15}, sampling="Original smile_teacher guarded concept and .55 content identity at -1,0,.5,+1; the new nonlinear conditional MLP does not impose linear interpolation"),
    _definition("circle-controller", "pr22", "Contextual circle action and 1024-step playback", "Recover radius and signed speed at unseen geometry/speed cells under true closed-loop playback", 2000,
        thresholds={"radial_rmse_max": .10, "signed_speed_error_max": .03, "direction_agreement_min": .95,
                    "recovery_radial_rmse_max": .10, "turns_min": 1.}, sampling="Pinned PR22 independent rows, disjoint heldout geometry/speed buckets, both directions; 128 episodes, main1024/recovery64",
        scope="New direct paired action network, exact source split/plant/rollout gates; no archived four-module pretraining or passed old PR22 qualification."),
    _definition("sprite-dynamics", "pr153", "Fully observed sprite state dreams and exact render", "Given all six state coordinates, predict the next state and free-run 5/20/50-step ID and ceiling-bounce OOD dreams", 8000,
        thresholds={"one_step_mse_max": .0025, "dream_position_rmse_max": .05, "dream_state_rmse_max": .10,
                    "dream_render_rmse_max": .20, "out_of_box_fraction_max": .01}, sampling="Pinned PR153 dynamics; independent ID starts and disjoint upward/fast ceiling-bounce OOD starts; 80-step episodes",
        scope="Fully observed six-coordinate learned world model plus exact source renderer. Narrower than an image-only latent world model; no oracle next-state input to the predictor."),
    _definition("previous-command-action", "source-family-13", "Action depends on previous command", "Recover the paired tanh(-2.2*previous) action, rather than a state/action marginal", 400,
        thresholds={"relative_mse_max": .18}, sampling="Independent state and previous command U[-1,1]; exact action=tanh(-2.2*previous)",
        scope="New direct two-input action network and current public paired-error GAN. This isolates command dependence; it does not replay the historical 250-step autoencoder pretrain or its 400-step latent-joint game."),
    *[_definition("mask-" + mechanism.replace("_", "-"), "pr196-" + mechanism, "Oracle-complete-label imputer: " + mechanism,
        "Using oracle complete training labels, preserve observed coordinates and recover ambiguous conditional posteriors; this does not test learning from incomplete data alone", 7000,
        kind="conditional_distribution", thresholds={"samples_per_row_min":4096,"observed_max_error_max": 1e-6, "posterior_mode_tv_max": .12,
            "missing_mean_rms_sigma_max": .15, "missing_variance_ratio": [.70, 1.30], "orthogonal_rms_ratio": [.70, 1.30]},
        sampling=f"Pinned PR196 lift/observed-standardization with 20000 training/10000 test rows; {mechanism}; exact conditional Bayes samples; 16 fixed ambiguous single-sensor queries (a positive-probability mask, explicitly oversampled at evaluation)",
        scope="Oracle-complete-data supervised conditional GAN reform, explicitly different information access from original incomplete-data MisGAN. Joint recovery is not inferred from imputation, and no archived MisGAN arm is qualified.")
        for mechanism in ("block", "mcar_p20", "mcar_p50", "mcar_p80")],
    _definition(
        "misgan-block-recovery", "pr196-block-recovery", "2-pair MisGAN recovery on 8D block sensor-dropout",
        "2-pair incomplete-data MisGAN with detached mask gradient in L_x on 8D block sensor-dropout problem; recovers 2D manifold, 8D subspace, mask distribution and ppost Bayesian imputation.", 7000,
        kind="misgan_recovery",
        thresholds={"modes_min": 100, "hq_min": 0.90, "mass_tv_max": 0.10, "off_plane_max": 0.04, "mask_tv_max": 0.05, "acc_min": 0.96, "acc_lo_min": 0.70},
        sampling="8D block sensor-dropout problem; 2-pair MisGAN (G_x, D_x) and (G_m, D_m); ppost imputation",
        scope="Tier 4 incomplete-data 2-pair MisGAN with detached mask gradient in L_x."),
    _definition(
        "misgan-block-neural-imputer", "pr196-block-neural-imputer", "3-pair MisGAN neural imputer on 8D block sensor-dropout",
        "3-pair MisGAN grading neural imputer G_i([x*m, m, z_i]) on held-out ambiguous single-sensor queries without ppost.", 7000,
        kind="misgan_neural_imputer",
        thresholds={"observed_max_error_max": 1e-6, "acc_min": 0.95, "acc_lo_min": 0.65, "missing_std_ratio": [0.70, 1.30]},
        sampling="8D block sensor-dropout problem; 3-pair MisGAN (G_x, D_x), (G_m, D_m), (G_i, D_i); ambiguous single-sensor queries",
        scope="Tier 4 3-pair MisGAN grading neural imputer G_i([x*m, m, z_i])."),
    _definition(
        "misgan-mcar-p50-recovery", "pr196-mcar-p50-recovery", "2-pair MisGAN recovery on 8D MCAR p=0.5 coordinate dropout",
        "2-pair incomplete-data MisGAN on 8D MCAR p=0.5 coordinate-wise dropout (resampling empty masks); recovers 2D 100-mode manifold, 8D subspace, MCAR coordinate observation rate and ppost Bayesian imputation.", 4000,
        kind="misgan_recovery",
        thresholds={"modes_min": 90, "hq_min": 0.85, "mass_tv_max": 0.12, "off_plane_max": 0.05, "mask_tv_max": 0.05, "acc_min": 0.90, "impute_rmse_max": 0.12},
        sampling="8D MCAR p=0.5 coordinate-wise dropout (resampling empty masks); 2-pair MisGAN (G_x, D_x) and (G_m, D_m); ppost imputation",
        scope="Tier 4 incomplete-data 2-pair MisGAN on MCAR p=0.5 coordinate dropout with detached mask gradient in L_x."),
    _definition(
        "misgan-mcar-p80-anchored", "pr196-mcar-p80-anchored", "3-pair MisGAN neural imputer on 8D MCAR p=0.80 with 10% anchors and GAIN hint",
        "3-pair MisGAN on 8D extreme MCAR p=0.80 missingness with 10% complete anchor rows and neural imputer G_i trained with GAIN partial-hint conditioning on D_i([x_imp, h]).", 4000,
        kind="misgan_anchored_imputer",
        thresholds={"modes_min": 80, "hq_min": 0.80, "mass_tv_max": 0.15, "off_plane_max": 0.06, "imp_acc_min": 0.65, "imp_off_plane_max": 0.06, "impute_rmse_max": 0.18},
        sampling="8D extreme MCAR p=0.80 missingness with 10% complete anchor rows; 3-pair MisGAN (G_x, D_x), (G_m, D_m), (G_i, D_i) with GAIN partial-hint conditioning",
        scope="Tier 4 3-pair MisGAN with GAIN partial-hint conditioning and 10% complete anchor rows under 80% MCAR missingness."),
]
for _row in _CASES:
    if _row["id"].startswith(("api-mask-","api-posterior-","api-misgan-")): _row["eval_samples"]=4096
    if _row["id"].startswith("api-misgan-"):
        # Preserve the standalone host's actual implementation identity: a
        # cloud has zero storage width; its explicit normal transform supplies
        # sampling noise. This is not the Forge positive-width MoG cohort.
        _row["initial_prior"] = dict(
            kind="particle_cloud", sigma=0., standardize=False, init_std=1.,
            initializer="deterministic_orthogonal", sampling="uniform_with_replacement",
            transform=dict(kind="normal_noise", sigma=.025, kappa_radial=1.))
    if _row["id"] in ("api-misgan-mcar-p50-recovery", "api-misgan-mcar-p80-anchored"):
        _row["batch_size"] = 512
        _row["default_recipe"] = "bispf"
    if _row["id"] in ("api-misgan-block-recovery", "api-misgan-block-neural-imputer"):
        _row["batch_size"] = 2048
        _row["default_recipe"] = "bispf"
    if _row["id"].startswith("api-sparse-"):
        _row["thresholds"]={k:v for k,v in _row["thresholds"].items() if k not in ("full_original_update_budget","terminal_observations")}
    _fixed_rows={"api-trajectory-edit":12,"api-trajectory-residual":12,"api-unipolar-hold":2,
        "api-unused-token-hold":2,"api-guarded-leftover":4,"api-midscale-identity":4,
        "api-circle-controller":128,"api-sprite-dynamics":64}
    if _row["id"] in _fixed_rows:
        _row["eval_samples"]=_fixed_rows[_row["id"]]
        _row["evaluation_count_policy"]="Fixed exact source panel, not redrawn when a caller supplies n"
CASES = {row["id"]: row for row in _CASES}


_STRETCH_PROVIDERS = (
    "benchmarks.toy_audit.api_pacgan",
    "benchmarks.toy_audit.api_infogan",
    "benchmarks.toy_audit.api_cgan_overlap",
    "benchmarks.toy_audit.api_cramergan",
    "benchmarks.toy_audit.api_cyclegan",
)


def _iter_stretch_providers():
    from importlib import import_module
    for mod_name in _STRETCH_PROVIDERS:
        yield import_module(mod_name)


def list_cases():
    rows = deepcopy(_CASES)
    for mod in _iter_stretch_providers():
        rows.extend(mod.list_cases())
    return rows


def _rng(seed, device="cpu"):
    return torch.Generator(device=device).manual_seed(int(seed))


def _bound(metrics, name, lo=None, hi=None):
    value = float(metrics[name])
    return [] if math.isfinite(value) and (lo is None or value >= lo) and (hi is None or value <= hi) else [name]


def observation(metrics, failures, views):
    return dict(metrics=metrics, passed=not failures, failed_bounds=sorted(set(failures)), views=views)


def view(title, target, samples, kind="scatter", **extra):
    return dict(kind=kind, title=title, target=target.detach().cpu() if isinstance(target, torch.Tensor) else target,
                samples=samples.detach().cpu() if isinstance(samples, torch.Tensor) else samples, **extra)


def cdf_error(points, reference):
    """Fixed-projection two-sample CDF distance, not a coordinate-mean proxy."""
    dimension = points.shape[1]
    mean, scale = reference.mean(0), reference.std(0).clamp_min(1e-5)
    x, y = (points-mean)/scale, (reference-mean)/scale
    directions = torch.cat((torch.eye(dimension), torch.randn(16, dimension, generator=_rng(5501))))
    directions = directions / directions.norm(dim=1, keepdim=True)
    worst = 0.
    for direction in directions:
        a, b = (x@direction).sort().values, (y@direction).sort().values
        # Check both sides of jumps; ties are allowed and empirical target is explicit.
        knots = torch.cat((a, b)).sort().values
        for right in (False, True):
            distance = (torch.searchsorted(a, knots, right=right)/len(a)
                        - torch.searchsorted(b, knots, right=right)/len(b)).abs().max()
            worst = max(worst, float(distance))
    return worst


class SparseTask:
    paired = False
    def __init__(self, split):
        self.toy = SparseMixedToy(symbol_map="split" if split else "identity", n_symbols=16 if split else 8)
        self.context_dim, self.output_dim = 8, 24+self.toy.n_symbols
    def batch(self, n, rng):
        c = torch.randint(8, (n,), generator=rng)
        x, s, _ = self.toy.sample_given_class(c, rng)
        return F.one_hot(c, 8).float(), torch.cat((x, F.one_hot(s, self.toy.n_symbols).float()), 1)
    def decode(self, raw):
        # A declared hard serving adapter gives exact sparsity; noise can only
        # occur on the three largest learned coordinates. Symbol remains learned.
        x, symbol = raw[:, :24], raw[:, 24:].argmax(1)
        active = x.abs().topk(3, dim=1).indices
        return torch.zeros_like(x).scatter(1, active, x.gather(1, active)), symbol
    def evaluate(self, predict, n, seed):
        c = torch.arange(8).repeat_interleave(n)
        raw = predict(F.one_hot(c, 8).float(), _rng(seed))
        x, symbols = self.decode(raw)
        result = density.sparse_metrics(self.toy, x, symbols, c)
        local = result["local_shape"]
        metrics = {key: result[key] for key in ("sample_count", "modes", "hq", "cond_acc", "sym_acc_mode", "exact_zero_frac", "max_conditional_quality_mass_tv", "max_symbol_tv")}
        metrics.update(mean_error_sigma=local["mean_error_sigma"], covariance_min=min(local["covariance_eigenvalues"]), covariance_max=max(local["covariance_eigenvalues"]), radial_ks=local["radial_ks"], projection_ks=local["max_projection_ks"])
        bounds = [("modes", 64, None), ("hq", .95, None), ("exact_zero_frac", 1, None), ("max_conditional_quality_mass_tv", None, .10), ("max_symbol_tv", None, .10), ("mean_error_sigma", None, .10), ("covariance_min", .85, None), ("covariance_max", None, 1.15), ("radial_ks", None, .075), ("projection_ks", None, .06), ("sample_count", 1024, None)]
        failures = [item for name, lo, hi in bounds for item in _bound(metrics, name, lo, hi)]
        target, target_s, _ = self.toy.sample_given_class(c, _rng(seed+1))
        views = [view("Matched requested-class sparse coordinates", target[:32], x[:32], "image", vmin=-2.2,vmax=2.2,caption="Rows are conditional draws, not uniquely paired targets; exact inactive zeros are part of the served adapter."),
                 view("Symbol distribution conditioned on each class", self.toy.p_symbol_given_class, torch.tensor(result["confusion"]), "image", caption="Both valid split symbols and mode-symbol coherence are scored.")]
        return observation(metrics, failures, views)


class PosteriorTask:
    paired = False
    def __init__(self, classes):
        self.toy = GaussianGrid(classes=classes)
        self.context_dim, self.output_dim = classes+3, 2
        self.panels = [(c, x, a) for c in range(classes) for x, a in [((0., 0.), .5), ((.2, -.3), .5), ((0., 0.), .9), ((.4, -.2), .9)]]
    def condition(self, c, x, a):
        return torch.cat((F.one_hot(c, self.toy.classes).float(), x, torch.full((len(c), 1), float(a))), 1)
    def batch(self, n, rng):
        c = torch.randint(self.toy.classes, (n,), generator=rng)
        clean = self.toy.sample(c, rng)
        a = (.5, .9)[int(torch.randint(2, (), generator=rng))]
        x = math.sqrt(a)*clean+math.sqrt(1-a)*torch.randn(n, 2, generator=rng)
        return self.condition(c, x, a), self.toy.oracle_clean(x, c, a, rng)
    def evaluate(self, predict, n, seed):
        results, views = [], []
        for i, (cls, xy, abar) in enumerate(self.panels):
            c, x = torch.full((n,), cls), torch.tensor(xy).expand(n, 2)
            points = predict(self.condition(c, x, abar), _rng(seed+i))
            result = density.posterior_metrics(self.toy, points, x[0], cls, abar)
            results.append(result)
            if i in (0, len(self.panels)-1):
                target = self.toy.oracle_clean(x, c, abar, _rng(seed+100+i))
                views.append(view(f"Posterior at class={cls}, observation={xy}, alpha_bar={abar}", target, points,
                    caption="Exact clean conditional draws and learned samples; noise input is not the target."))
        metrics = dict(panel_count=len(results), sample_count_per_panel=n,
            worst_quality_mass_tv=max(r["conditional_quality_mass_tv"] for r in results),
            min_hq=min(r["hq"] for r in results), worst_mean_sigma=max(r["local_shape"]["mean_error_sigma"] for r in results),
            min_covariance=min(min(r["local_shape"]["covariance_eigenvalues"]) for r in results),
            max_covariance=max(max(r["local_shape"]["covariance_eigenvalues"]) for r in results),
            worst_radial_ks=max(r["local_shape"]["radial_ks"] for r in results),
            worst_projection_ks=max(r["local_shape"]["max_projection_ks"] for r in results))
        failures = [item for name, lo, hi in [("sample_count_per_panel",1024,None),("worst_quality_mass_tv",None,.10),("min_hq",.95,None),("worst_mean_sigma",None,.10),("min_covariance",.85,None),("max_covariance",None,1.15),("worst_radial_ks",None,.075),("worst_projection_ks",None,.06)] for item in _bound(metrics,name,lo,hi)]
        return observation(metrics, failures, views)


class RouteTask:
    paired = False
    def __init__(self, mode, transition=False):
        self.transition = transition
        self.toy = Transitions(geometry_mode=mode) if transition else Routes(geometry_mode=mode)
        self.context_dim, self.output_dim = (6, 6) if transition else (5, 128)
    def condition(self, c, geom, tick=None):
        parts = [F.one_hot(c,2).float(),geom]
        if self.transition: parts.append((tick.float()/63)[:,None])
        return torch.cat(parts,1)
    def batch(self, n, rng):
        if self.transition:
            c,g,t,x=self.toy.batch(n,rng)
            return self.condition(c,g,t),x
        c,g,x=self.toy.batch(n,rng)
        return self.condition(c,g),x.flatten(1)
    def evaluate(self,predict,n,seed):
        cc,gg=self.toy.contexts("test")
        records,views=[],[]
        for i,(c0,g0) in enumerate(zip(cc,gg)):
            c,g=c0.expand(n),g0.expand(n,3)
            if self.transition:
                for tick0 in (0,8,31,54,62):
                    tick=torch.full((n,),tick0)
                    points=predict(self.condition(c,g,tick),_rng(seed+i*64+tick0))
                    target=self.toy.sample(c,g,tick,_rng(seed+1000+i*64+tick0))
                    records.append(dict(consistency=float((points[:,:2]+points[:,2:4]-points[:,4:]).square().mean().sqrt()),cdf=cdf_error(points,target)))
                    if i==0 and tick0 in (8,31,54):
                        views.append(view(f"Heldout joint transition at tick {tick0}", target[:,:2],points[:,:2],caption="State cloud; adjacent view shows displacement and successor, all sampled from the same 6D output."))
                        views.append(view(f"Action and successor, tick {tick0}",target[:,2:],points[:,2:],"image",vmin=-1.2,vmax=1.2,caption="Rows align state/action/successor; algebra and joint conditional CDF are both scored."))
            else:
                points=predict(self.condition(c,g),_rng(seed+i)).reshape(n,2,64)
                target,_=self.toy.sample(c,g,_rng(seed+1000+i))
                d=self.toy.diagnose(points,g)
                coeff=d["coeff"]
                masses=torch.bincount(d["route"][d["valid"]],minlength=2).float()/n
                p=self.toy.upper_probability[int(c0)]
                tv=float(((masses-torch.tensor([1-p,p])).abs().sum()+1-masses.sum())/2)
                ks=max(density.ks(coeff[:,k].numpy(),lambda a:((a+1)/2).clip(0,1)) for k in range(3))
                records.append(dict(valid=float(d["valid"].float().mean()),tv=tv,
                    support=float((coeff.abs()<=1.0001).all(1).float().mean()),mean=float(coeff.mean(0).abs().max()),
                    var_min=float(coeff.var(0,unbiased=False).min()),var_max=float(coeff.var(0,unbiased=False).max()),ks=ks))
                if i in (0,1,6,7):
                    views.append(view(f"Unseen geometry {i//2}, class {int(c0)}: 64-point routes", target[:12].transpose(1,2),points[:12].transpose(1,2),"line",
                        caption="Both routes, class weights, continuous segment clearance and uniform coefficient variation are scored; obstacle center/radius="+str(g0.tolist()),xlim=[-1.1,1.1],ylim=[-.9,1.0]))
        if self.transition:
            metrics=dict(panel_count=len(records),sample_count_per_panel=n,joint_consistency_rms=max(r["consistency"] for r in records),max_projection_cdf_error=max(r["cdf"] for r in records))
            limits=[("sample_count_per_panel",256,None),("joint_consistency_rms",None,.002),("max_projection_cdf_error",None,.12)]
        else:
            metrics=dict(panel_count=len(records),sample_count_per_panel=n,min_valid_fraction=min(r["valid"] for r in records),max_quality_route_mass_tv=max(r["tv"] for r in records),
                min_coefficient_support=min(r["support"] for r in records),max_coefficient_mean=max(r["mean"] for r in records),min_coefficient_variance=min(r["var_min"] for r in records),max_coefficient_variance=max(r["var_max"] for r in records),max_coefficient_ks=max(r["ks"] for r in records))
            limits=[("sample_count_per_panel",256,None),("min_valid_fraction",.95,None),("max_quality_route_mass_tv",None,.10),("min_coefficient_support",.95,None),("max_coefficient_mean",None,.10),("min_coefficient_variance",.22,None),("max_coefficient_variance",None,.45),("max_coefficient_ks",None,.10)]
        return observation(metrics,[item for name,lo,hi in limits for item in _bound(metrics,name,lo,hi)],views)


class MaskTask:
    paired = False
    def __init__(self,mechanism):
        self.problem=misgan.Problem(mechanism,20000,10000,0,"cpu")
        self.context_dim,self.output_dim=16,8
        x,m=self.problem.x_test*self.problem.m_test,self.problem.m_test
        probs,_=misgan.bayes_posterior(self.problem,x,m)
        ids=((m.sum(1)<=1)&(probs.max(1).values<.95)).nonzero().flatten()[:16]
        if len(ids)<16:
            # Under p20 these masks are very rare. Keep the original training
            # mechanism, but explicitly condition the evaluation query on the
            # supported one-sensor pattern; never silently change training masks.
            m=torch.zeros_like(self.problem.m_test);m[:,7]=1
            x=self.problem.x_test*m
            probs,_=misgan.bayes_posterior(self.problem,x,m)
            ids=(probs.max(1).values<.95).nonzero().flatten()[:16]
        if len(ids)<16: raise ValueError("required 16 ambiguous posterior queries absent")
        self.ids=ids;self.eval_mask=m[ids].clone()
    def batch(self,n,rng):
        p=self.problem
        x,m,full=p.train_batch(n,rng)
        # Explicit oracle-full access: this is a supervised imputer reform, not MisGAN.
        return torch.cat((x,m),1),full
    def adapt(self,raw,context):
        x,m=context[:,:8],context[:,8:]
        return x*m+(1-m)*raw
    def evaluate(self,predict,n,seed):
        p,ids=self.problem,self.ids
        m=self.eval_mask;x=p.x_test[ids]*m
        probs,target=misgan.bayes_posterior(p,x,m,draws=n,generator=_rng(seed+1))
        # Same test rows and same fixed missingness; never resample the mask between draws.
        context=torch.cat((x,m),1).repeat(n,1)
        points=predict(context,_rng(seed)).reshape(n,len(ids),8)
        raw=p.raw(points); reference=p.raw(target)
        mu=misgan.grid_centers(dtype=torch.float64)
        projected=raw@p.A
        assigned=torch.cdist(projected.flatten(0,1),mu).argmin(1).reshape(n,len(ids))
        tv=[]
        for j in range(len(ids)):
            mass=torch.bincount(assigned[:,j],minlength=100).float()/n
            tv.append(float((mass-probs[j]).abs().sum()/2))
        missing=(1-m).bool()
        target_var=reference.var(0,unbiased=False)
        # Use exact finite Bayes reference variability for each missing coordinate;
        # observed coordinates are checked separately and never enter the ratio.
        variance_ratio=(raw.var(0,unbiased=False)/target_var.clamp_min(1e-12))[missing]
        mean_error=((raw.mean(0)-reference.mean(0))/target_var.clamp_min(1e-12).sqrt())[missing]
        ortho=raw-raw@p.A@p.A.T
        target_ortho=reference-reference@p.A@p.A.T
        ortho_ratio=float(ortho.square().mean().sqrt()/target_ortho.square().mean().sqrt())
        metrics=dict(ambiguous_rows=len(ids),samples_per_row=n,observed_max_error=float(((points-x)*m).abs().max()),
            max_posterior_mode_tv=max(tv),missing_mean_rms_sigma=float(mean_error.square().mean().sqrt()),
            min_missing_variance_ratio=float(variance_ratio.min()),max_missing_variance_ratio=float(variance_ratio.max()),orthogonal_rms_ratio=ortho_ratio)
        limits=[("ambiguous_rows",16,None),("samples_per_row",4096,None),("observed_max_error",None,1e-6),("max_posterior_mode_tv",None,.12),("missing_mean_rms_sigma",None,.15),("min_missing_variance_ratio",.70,None),("max_missing_variance_ratio",None,1.30),("orthogonal_rms_ratio",.70,1.30)]
        failures=[item for name,lo,hi in limits for item in _bound(metrics,name,lo,hi)]
        views=[view("Ambiguous row 0: conditional 2D posterior",p.to2d(target[:,0]),p.to2d(points[:,0]),caption=f"Oracle complete-label training baseline; heldout row {int(ids[0])}, mask {m[0].tolist()}. This does not test original incomplete-data-only MisGAN learning. All eight coordinates are scored."),
               view("Missing-coordinate draws across fixed rows",target[:16].reshape(-1,8),points[:16].reshape(-1,8),"image",vmin=-2.2,vmax=2.2,caption="Oracle complete-label baseline; observed entries must stay fixed, and posterior variance plus orthogonal lift noise must remain. Original incomplete-data-only learning remains unverified.")]
        return observation(metrics,failures,views)


class PairedTask:
    paired=True
    def __init__(self,name):
        self.name=name
        if name.startswith("transport"):
            self.context_dim=self.output_dim=2
        elif name.startswith("trajectory"):
            from benchmarks.locked_shared.trajectory import trajectories
            self.slow,self.fast=trajectories()
            self.context_dim=self.output_dim=16
        elif name=="unipolar-hold": self.context_dim,self.output_dim=1,4
        elif name=="unused-token-hold": self.context_dim,self.output_dim=1,4
        elif name in ("guarded-leftover","midscale-identity"):
            self.context_dim,self.output_dim=1,4
            from benchmarks.locked_shared.hosts.cover_leftover import LeftoverField,teacher_poles
            if name=="guarded-leftover":
                plus,minus,neutral=teacher_poles(LeftoverField(),"faithful_guard_e")
                self.neutral,self.direction=neutral,(plus-minus)/2
            else:
                from benchmarks.locked_shared.hosts.mid_scale_identity import smile_teacher
                teacher=smile_teacher(); self.neutral,self.direction=teacher.identity,teacher.concept
        elif name=="circle-controller": self.context_dim,self.output_dim=6,2
        elif name=="sprite-dynamics": self.context_dim=self.output_dim=6
        elif name=="previous-command-action": self.context_dim,self.output_dim=2,1
        else: raise ValueError(name)
    def targets(self,c):
        name=self.name
        if name=="transport-affine2": return c@c.new_tensor([[.8,-.6],[.6,.8]])+c.new_tensor([.2,-.3])
        if name=="transport-swirl2":
            angle=1.7*(c/math.sqrt(3)).square().sum(1)
            x,y=c.unbind(1); return torch.stack((x*angle.cos()-y*angle.sin(),x*angle.sin()+y*angle.cos()),1)
        if name=="unipolar-hold": return c*c.new_tensor([[1.,0.,0.,0.]])
        if name=="unused-token-hold": return c.new_tensor([[1.,0.,0.,0.]])+c*c.new_tensor([[0.,0.,0.,1.]])
        if name in ("guarded-leftover","midscale-identity"): return self.neutral+c*self.direction
        if name=="circle-controller": return circle.expert_transition(c[:,:2],c[:,2:4],c[:,4],c[:,5])[0]
        if name=="sprite-dynamics": return sprite.step(c)
        if name=="previous-command-action": return (-2.2*c[:,1:]).tanh()
        raise ValueError("trajectory targets have explicit pair indices")
    def batch(self,n,rng):
        name=self.name
        if name.startswith("transport"): c=(torch.rand(n,2,generator=rng)*2-1)*math.sqrt(3)
        elif name=="previous-command-action": c=torch.rand(n,2,generator=rng)*2-1
        elif name.startswith("trajectory"):
            ids=torch.randint(12,(n,),generator=rng); return self.slow[ids],self.fast[ids]
        elif name in ("unipolar-hold","unused-token-hold"): c=torch.randint(2,(n,1),generator=rng).float()
        elif name in ("guarded-leftover","midscale-identity"):
            scales=torch.tensor([-1.,0.,1.] if name=="guarded-leftover" else [-1.,0.,.5,1.])
            c=scales[torch.randint(len(scales),(n,1),generator=rng)]
        elif name=="circle-controller":
            b=circle.sample_rows(n,rng); c=torch.cat((b.position,b.context()),1)
        elif name=="sprite-dynamics":
            # ID episodes only. Training draws independent episode/time rows.
            starts=sprite.sample_starts(n,(.2,.6),(0.,1.),(0.,2*math.pi),np.random.default_rng(int(torch.randint(2**31,(),generator=rng))))
            states=torch.tensor(starts,dtype=torch.float32)
            ticks=torch.randint(80,(n,),generator=rng)
            for t in range(int(ticks.max())):
                states=torch.where((ticks>t)[:,None],sprite.step(states),states)
            c=states
        else: raise ValueError(name)
        return c,self.targets(c)
    def evaluate(self,predict,n,seed):
        name=self.name
        if name=="circle-controller": return circle_observation(predict)
        if name=="sprite-dynamics": return sprite_observation(predict,seed)
        c,t=self.batch(max(n,12),_rng(seed))
        if name.startswith("trajectory"): c,t=self.slow,self.fast
        elif name in ("unipolar-hold","unused-token-hold"): c=torch.tensor([[0.],[1.]]); t=self.targets(c)
        elif name in ("guarded-leftover","midscale-identity"):
            c=torch.tensor([[-1.],[0.],[.5],[1.]]); t=self.targets(c)
        p=predict(c,_rng(seed+1))
        error=p-t; metrics=dict(evaluation_rows=len(c),paired_mse=float(error.square().mean()),max_context_l2=float(error.norm(dim=1).max()))
        failures=[]
        if name.startswith("transport"):
            metrics.update(normalized_mse=float(error.square().mean()/t.var(0,unbiased=False).mean()),p95_l2=float(torch.quantile(error.norm(dim=1),.95)))
            failures=_bound(metrics,"normalized_mse",hi=.01)+_bound(metrics,"p95_l2",hi=.20)
        elif name=="previous-command-action":
            metrics["relative_mse"]=float(error.square().mean()/t.square().mean())
            failures=_bound(metrics,"relative_mse",hi=.18)
        elif name.startswith("trajectory"):
            failures=_bound(metrics,"paired_mse",hi=.02)
            metrics["wrong_pair_margin"]=float((p-t.roll(6,0)).square().mean()-error.square().mean())
            if name=="trajectory-residual": failures+=_bound(metrics,"wrong_pair_margin",lo=.02)
        elif name=="unipolar-hold":
            norm=float(p[1].norm()); cosine=float(F.cosine_similarity(p[1:2],t[1:2]))
            metrics.update(cover=max(0.,cosine)*max(0.,1-abs(norm-1)),off_axis=float(p[1,1:].square().sum()/(p[1].square().sum()+1e-12)),neutral_hold=1-min(1.,float(p[0].norm())))
            failures=_bound(metrics,"cover",lo=.85)+_bound(metrics,"off_axis",hi=.05)+_bound(metrics,"neutral_hold",lo=.85)
        elif name=="unused-token-hold":
            delta=p[1]-t[0]; metrics.update(concept_move=1-min(1.,float((delta[2:]-torch.tensor([0.,1.])).norm())),unused_hold=1-min(1.,float(delta[:2].norm())))
            failures=_bound(metrics,"concept_move",lo=.85)+_bound(metrics,"unused_hold",lo=.85)
        else:
            baseline=(t-self.neutral).square().mean()
            metrics["relative_mse"]=float(error.square().mean()/baseline)
            if name=="guarded-leftover":
                metrics["pole_relative_l2"]=float((error[[0,3]].norm(dim=1)/t[[0,3]].norm(dim=1)).max())
                metrics["leak_ratio"]=float((p[3]-self.neutral)[2].abs()/(p[3]-self.neutral)[0].abs().clamp_min(1e-12))
                failures=_bound(metrics,"pole_relative_l2",hi=.10)+_bound(metrics,"leak_ratio",hi=.10)
            else: failures=_bound(metrics,"relative_mse",hi=.10)+_bound(metrics,"max_context_l2",hi=.15)
        if name.startswith("trajectory"):
            views=[view("Paired fast trajectories with starting identity retained",t.reshape(12,8,2),p.reshape(12,8,2),"line",caption="Each learned path is compared with its own phase/radius target, not nearest target.")]
        elif name=="previous-command-action":
            order=c[:,1].argsort()
            views=[view("Previous command to action, with independent state",torch.cat((c[order,1:2],t[order]),1),torch.cat((c[order,1:2],p[order]),1),"line",caption="Horizontal coordinate is the actual previous command; the independent state cannot replace this condition.")]
        elif name.startswith("transport"):
            views=[view("Heldout source-to-target displacement",torch.stack((c[:16],t[:16]),1),torch.stack((c[:16],p[:16]),1),"line",caption="Each path starts at the same heldout source. A correct marginal with permuted correspondence fails.")]
        else:
            views=[view("Matched heldout inputs: target and learned output",t,p,"scatter" if self.output_dim==2 else "image",vmin=-1.5,vmax=1.5,caption="Target/output rows share the same heldout input; neutral and intermediate cases are included.")]
        return observation(metrics,failures,views)


def circle_observation(predict):
    metrics,views={"episodes_per_panel":128},[]
    failures=[]
    for kind,horizon in (("main",1024),("recovery",64)):
        b=circle.evaluation_panel("test",kind)
        position=b.position.clone(); exact=position.clone()
        ids=torch.cat(((b.angular_step>0).nonzero().flatten()[:2],(b.angular_step<0).nonzero().flatten()[:2]))
        paths,targets=[position[ids].clone()],[exact[ids].clone()]
        radial,speed,direction,angular=[],[],[],[]
        completed=0;nonfinite=0
        for _ in range(horizon):
            context=torch.cat((position,b.context()),1)
            action=predict(context,_rng(EVAL_SEED))
            nxt=position+action
            nonfinite=int((~torch.isfinite(nxt).all(1)).sum())
            if nonfinite: break
            # Evaluate large but finite rollout errors in float64. A genuine
            # nonfinite learned successor stops the observed prefix; it never
            # gets clipped, reset to the oracle, or counted as completed.
            oldq, newq=(position-b.center).double(),(nxt-b.center).double()
            angle=torch.atan2(oldq[:,0]*newq[:,1]-oldq[:,1]*newq[:,0],(oldq*newq).sum(1))
            radial.append((newq.norm(dim=1)/b.radius-1).square())
            speed.append((angle-b.angular_step).abs()); direction.append((angle*b.angular_step>0).float()); angular.append(angle)
            position=nxt
            completed+=1
            exact=circle.expert_transition(exact,b.center,b.radius,b.angular_step)[1]
            paths.append(position[ids].clone());targets.append(exact[ids].clone())
        # Recovery is scored after its fixed64 window, not while correcting the initial offset.
        metrics[kind+"_completed_rollout_steps"]=completed
        metrics[kind+"_nonfinite_episodes"]=nonfinite
        metrics[kind+"_radial_rmse"]=float((radial[-1] if kind=="recovery" else torch.stack(radial).mean(0)).mean().sqrt()) if radial else 1e6
        failures+=_bound(metrics,kind+"_completed_rollout_steps",lo=horizon)+_bound(metrics,kind+"_nonfinite_episodes",hi=0)
        if kind=="main":
            metrics.update(signed_speed_error=float(torch.stack(speed).mean()) if speed else 1e6,direction_agreement=float(torch.stack(direction).mean()) if direction else 0.,min_turns=float(torch.stack(angular).sum(0).abs().min()/(2*math.pi)) if angular else 0.)
            failures+=_bound(metrics,"signed_speed_error",hi=.03)+_bound(metrics,"direction_agreement",lo=.95)+_bound(metrics,"min_turns",lo=1.)
        failures+=_bound(metrics,kind+"_radial_rmse",hi=.10)
        views.append(view(f"True closed-loop {kind}, requested {horizon} transitions",torch.stack(targets,1),torch.stack(paths,1),"line",caption=f"Observed {completed}/{horizon} finite transitions; nonfinite successors={nonfinite}. No teacher successor is fed back, reset or clipped. Both signed directions are present; an incomplete evaluation cannot pass."))
    return observation(metrics,failures,views)


def sprite_observation(predict,seed):
    metrics,views,failures={"episodes_per_split":64},[],[]
    for ood in (False,True):
        name="ood" if ood else "id"
        starts=sprite.sample_starts(64,(.75,.9) if ood else (.2,.6),(1.2,1.6) if ood else (0.,1.),(math.pi/6,5*math.pi/6) if ood else (0.,2*math.pi),np.random.default_rng(seed+(3000 if ood else 2000)))
        state=torch.tensor(starts,dtype=torch.float32); exact=state.clone()
        metrics[name+"_one_step_mse"]=float((predict(state,_rng(seed))-sprite.step(state)).square().mean())
        failures+=_bound(metrics,name+"_one_step_mse",hi=.0025)
        positions,truth=[state[:4,:2].clone()],[exact[:4,:2].clone()]
        for step in range(1,51):
            state=predict(state,_rng(seed));exact=sprite.step(exact)
            positions.append(state[:4,:2].clone());truth.append(exact[:4,:2].clone())
            if step in (5,20,50):
                prefix=f"{name}_{step}"
                metrics[prefix+"_position_rmse"]=float((state[:,:2]-exact[:,:2]).square().mean().sqrt())
                metrics[prefix+"_state_rmse"]=float((state-exact).square().mean().sqrt())
                metrics[prefix+"_render_rmse"]=float((sprite.render(state)-sprite.render(exact)).square().mean().sqrt())
                metrics[prefix+"_out_of_box"]=float(((state[:,:2]<.1)|(state[:,:2]>.9)).any(1).float().mean())
                for suffix,hi in (("position_rmse",.05),("state_rmse",.10),("render_rmse",.20),("out_of_box",.01)):
                    failures+=_bound(metrics,prefix+"_"+suffix,hi=hi)
                if step==20:
                    views.append(view(f"{name} 20-step dreamed frames",sprite.render(exact[:4]),sprite.render(state[:4]),"image",caption="Fully observed six-state learned model and exact renderer, narrower than the original image-only latent world model. Actual predictions; no interpolation or oracle next-state input."))
        views.append(view(f"{name} 50-step free-running positions",torch.stack(truth,1),torch.stack(positions,1),"line",caption="OOD starts rise into ceiling bounces absent from the training episode band."))
    return observation(metrics,failures,views)


def _task(name):
    if name.startswith("sparse"): return SparseTask(name.endswith("split"))
    if name.startswith("posterior"): return PosteriorTask(int(name.split("-")[1].split("class")[0]))
    if name.startswith("routes"): return RouteTask(name.split("-")[1])
    if name.startswith("transitions"): return RouteTask(name.split("-")[1],True)
    if name.startswith("mask"): return MaskTask(name.removeprefix("mask-").replace("-","_"))
    if name == "misgan-block-recovery": return MisGANBlockRecoveryTask()
    if name == "misgan-block-neural-imputer": return MisGANBlockNeuralImputerTask()
    if name == "misgan-mcar-p50-recovery": return MisGANMCARP50RecoveryTask()
    if name == "misgan-mcar-p80-anchored": return MisGANMCARP80AnchoredTask()
    return PairedTask(name)


class ConditionalGenerator(nn.Module):
    def __init__(self,task,z_dim):
        super().__init__();self.task_name=task.name if isinstance(task,PairedTask) else None
        self.paired=task.paired
        self.net=nn.Sequential(nn.Linear(task.context_dim+(0 if task.paired else z_dim),64),nn.LeakyReLU(.2),nn.Linear(64,64),nn.LeakyReLU(.2),nn.Linear(64,task.output_dim))
    def forward(self,context,z=None):
        raw=self.net(context if self.paired else torch.cat((context,z),1))
        return raw+context if self.task_name=="trajectory-residual" else raw


class ConditionalCritic(nn.Module):
    def __init__(self,dim,context_dim):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim+context_dim,64),nn.LeakyReLU(.2),nn.Linear(64,64),nn.LeakyReLU(.2),nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(-1)


class TokenHoldGenerator(nn.Module):
    """Retain the source's shared-vector coupling and independent slot correction."""
    def __init__(self):
        super().__init__()
        self.shared=nn.Parameter(torch.zeros(2));self.slot=nn.Parameter(torch.zeros(2,2))
        self.register_buffer("neutral",torch.tensor([[1.,0.],[0.,0.]]))
    def forward(self,context,z=None):
        return (self.neutral+context[:,:,None]*(self.shared+self.slot)[None]).flatten(1)


init.register(TokenHoldGenerator,{"shared":init.KEEP,"slot":init.KEEP})


class ConditionalFixture:
    api_components=("Recipe.make_prior","Recipe.make_optimizers","Recipe.make_loss","Recipe.make_critic_penalty","scale_learning_rates")
    def __init__(self,case,task,seed,recipe_name,max_steps):
        if recipe_name not in ("ka2","k3p","mog"):
            raise ValueError("These caller-owned conditional samplers require the declared KA2/MoG adaptation; independent Atlas row controls are not bound. Use --recipe auto or ka2.")
        self.case,self.task,self.seed=case,task,int(seed);self.completed_steps=0
        self.limit=case["default_steps"] if max_steps is None else max_steps
        if type(self.limit) is not int or not 0<self.limit<=case["default_steps"]: raise ValueError("max_steps must be in the declared budget")
        self.recipe=get_recipe("k3p" if recipe_name=="k3p" else "ka2",
            batch_size=case["batch_size"],total_steps=case["default_steps"],num_particles=128,z_dim=8,
            prior_kind="mog",sigma_rel=.25,standardize=False,input_noise_std=0.,output_noise_std=0.,lr=.002,
            particle_birth_death=False,row_evidence_gate=False,serve_average=0.).replace(name="conditional_"+recipe_name)
        with construction_rng(seed, "cpu"):
            self.G=TokenHoldGenerator() if isinstance(task,PairedTask) and task.name=="unused-token-hold" else ConditionalGenerator(task,8)
            self.D=ConditionalCritic(task.output_dim,task.context_dim)
            self.prior=None if task.paired else self.recipe.make_prior()
        init.deterministic_orthogonal_(self.G,seed=seed+1);init.deterministic_orthogonal_(self.D,seed=seed+2)
        if self.prior is not None: init.deterministic_orthogonal_(self.prior,seed=seed+3)
        if self.prior is None: self.api_components=tuple(name for name in self.api_components if name!="Recipe.make_prior")
        self.opt_g,self.opt_d=self.recipe.make_optimizers(self.G,self.D,self.prior,ema_critic=deepcopy(self.D),foreach=False)
        self.base_rates=[[group["lr"] for group in opt.param_groups] for opt in (self.opt_g,self.opt_d)]
        self.loss=self.recipe.make_loss();self.penalty=self.recipe.make_critic_penalty(self.opt_d,collect_stats=True)
        self.data_rng,self.latent_rng,self.noise_rng=_rng(seed+10),_rng(seed+11),_rng(seed+12)
        self.last_losses={}
        self.initialization="public deterministic orthogonal; no target/oracle weights"
    def _generate(self,c,rng):
        z=None if self.prior is None else self.prior.sample(len(c),generator=rng)[0]
        raw=self.G(c,z)
        return self.task.adapt(raw,c) if isinstance(self.task,MaskTask) else raw
    def step(self):
        if self.completed_steps>=self.limit: raise RuntimeError("declared execution cap reached")
        scale_learning_rates(self.completed_steps,self.recipe,[self.opt_g,self.opt_d],self.base_rates,self.prior)
        c,t=self.task.batch(self.recipe.batch_size,self.data_rng)
        paired=self.task.paired
        def coordinates(pred,epsilon):
            real=epsilon if paired else t
            fake=epsilon+pred-t if paired else pred
            return torch.cat((real,c),1),torch.cat((fake,c),1)
        self.G.train();self.D.train()
        with torch.no_grad():
            pred=self._generate(c,self.latent_rng);eps=.10*torch.randn(t.shape,generator=self.noise_rng)
            real,fake=coordinates(pred,eps)
        self.opt_d.zero_grad();dl=self.loss.d_loss(self.D(real),self.D(fake));reg=self.penalty(self.D,real,fake)
        (dl+reg).backward();self.opt_d.step()
        flags=[p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            pred=self._generate(c,self.latent_rng);eps=.10*torch.randn(t.shape,generator=self.noise_rng)
            real,fake=coordinates(pred,eps)
            with torch.no_grad(): real_logits=self.D(real)
            gl=self.loss.g_loss(self.D(fake),real_logits)
            self.opt_g.zero_grad();gl.backward();self.opt_g.step()
        finally:
            for p,f in zip(self.D.parameters(),flags): p.requires_grad_(f)
        self.completed_steps+=1
        self.last_losses=dict(g_gan=float(gl.detach()),d_gan=float(dl.detach()),critic_penalty=float(reg.detach()))
        if not all(math.isfinite(v) for v in self.last_losses.values()): raise FloatingPointError("nonfinite actual GAN update")
        return dict(step=self.completed_steps,**self.last_losses)
    @torch.no_grad()
    def observe(self,n=1024,seed=EVAL_SEED):
        if type(n) is not int or n<2: raise ValueError("evaluation count must be >=2")
        flags={m:m.training for model in (self.G,self.D) for m in model.modules()}
        try:
            self.G.eval();self.D.eval()
            return self.task.evaluate(self._generate,n,int(seed))
        finally:
            for m,flag in flags.items(): m.training=flag
    def state_dict(self):
        models=(("G",self.G),("D",self.D))+(() if self.prior is None else (("prior",self.prior),))
        task_state={key:value for key,value in vars(self.task).items() if key not in ("toy","problem")}
        if hasattr(self.task,"toy"): task_state["sampler"]=vars(self.task.toy)
        if hasattr(self.task,"problem"): task_state["problem"]=vars(self.task.problem)
        return deepcopy(dict(version=VERSION,case_id=self.case["id"],seed=self.seed,limit=self.limit,
            completed_steps=self.completed_steps,recipe=self.recipe.to_dict(),generator=self.G.state_dict(),critic=self.D.state_dict(),
            prior=None if self.prior is None else self.prior.state_dict(),opt_g=self.opt_g.state_dict(),opt_d=self.opt_d.state_dict(),
            data_rng=self.data_rng.get_state(),latent_rng=self.latent_rng.get_state(),noise_rng=self.noise_rng.get_state(),
            modes={role:{name:m.training for name,m in model.named_modules()} for role,model in models},
            requires_grad={role:{name:p.requires_grad for name,p in model.named_parameters()} for role,model in models},
            gradients={role:{name:None if p.grad is None else p.grad.clone() for name,p in model.named_parameters()} for role,model in models},
            fixed_task=task_state,global_rng=torch.random.get_rng_state(),last_losses=self.last_losses))


def misgan_masked_sqdist(x, m, table):
    xm = x * m
    d = m @ (table * table).T - 2 * xm @ table.T + (xm * x).sum(1, keepdim=True)
    return d.clamp_min(0)


def misgan_ppost_indices(table, x, m, sigma=0.005, draws=1, generator=None, chunk=2048):
    if len(table) == 0:
        raise ValueError("candidate particle table cannot be empty")
    table_dtype = table.dtype if table.dtype.is_floating_point else torch.float32
    table = table.to(dtype=table_dtype)
    x = x.to(device=table.device, dtype=table_dtype)
    m = m.to(device=table.device, dtype=table_dtype)

    table_sq_t = (table * table).T
    table_t = table.T
    scale = -1.0 / (2.0 * sigma * sigma)
    n_table = table.shape[0]

    out = []
    for xb, mb in zip(x.split(chunk), m.split(chunk)):
        xm = xb * mb
        d = mb @ table_sq_t - 2.0 * xm @ table_t + (xm * xb).sum(1, keepdim=True)
        d = d.clamp_min(0.0)
        w = torch.softmax(d * scale, dim=1)
        if not torch.all(torch.isfinite(w)):
            nan_mask = ~torch.isfinite(w).all(dim=1)
            w = w.clone()
            w[nan_mask] = 1.0 / n_table
        if generator is not None and generator.device.type == "cpu" and w.is_cuda:
            sampled = torch.multinomial(w.cpu(), draws, replacement=True, generator=generator).to(table.device)
        else:
            sampled = torch.multinomial(w, draws, replacement=True, generator=generator)
        out.append(sampled.T)
    return torch.cat(out, 1)


@torch.no_grad()
def score_misgan_recovery(x, m, problem=None, eval_seed=99123, use_ppost=True):
    if problem is None:
        problem = misgan.Problem("block", 20000, 10000, 0, "cpu")
    
    if not torch.all(torch.isfinite(x)) or not torch.all(torch.isfinite(m)):
        metrics = dict(
            modes=0,
            hq=0.0,
            mass_tv=1.0,
            off_plane=float("inf"),
            mask_tv=1.0,
            acc=0.0,
            acc_lo=0.0,
        )
        failures = ["modes", "hq", "mass_tv", "off_plane", "mask_tv", "acc", "acc_lo", "nonfinite_output"]
        return observation(metrics, failures, [])

    off_plane = problem.off_plane(x)

    x2 = problem.to2d(x)
    centers = misgan.grid_centers(device=x.device, dtype=torch.float32)
    dists = torch.cdist(x2, centers)
    d, nearest = dists.min(1)
    hq = float((d <= 0.09).float().mean())
    hq_counts = torch.bincount(nearest[d <= 0.09], minlength=100)
    modes = int((hq_counts >= 10).sum())
    all_counts = torch.bincount(nearest, minlength=100).float()
    mass_tv = float(0.5 * (all_counts / len(x) - 0.01).abs().sum())

    hard_m = (m > 0.5).float()
    emp = torch.bincount(misgan.pattern_index(hard_m).cpu(), minlength=256).double() / len(m)
    mask_tv = float(0.5 * (emp - problem.probs.cpu()).abs().sum())

    centers_eval = centers.to(device=problem.x_test.device, dtype=problem.x_test.dtype)
    y_test_eval = problem.y_test.to(device=problem.x_test.device)
    lo = problem.m_test.sum(1) <= 1

    if use_ppost:
        k = misgan_ppost_indices(x, problem.x_test, problem.m_test, sigma=0.005, draws=4, generator=_rng(eval_seed))
        hit_list = []
        imp_x2 = None
        for draw_idx in range(k.shape[0]):
            x_k = x[k[draw_idx]].to(device=problem.x_test.device, dtype=problem.x_test.dtype)
            imp_d = problem.m_test * problem.x_test + (1 - problem.m_test) * x_k
            imp_x2_d = problem.to2d(imp_d)
            if draw_idx == 0:
                imp_x2 = imp_x2_d
            imp_nearest = torch.cdist(imp_x2_d, centers_eval).argmin(1)
            hit_d = (imp_nearest == y_test_eval).float()
            hit_list.append(hit_d)
        hit = torch.stack(hit_list, dim=0).mean(dim=0)
    else:
        _, imp_draws = misgan.bayes_posterior(problem, problem.x_test, problem.m_test, draws=1, generator=_rng(eval_seed))
        imp = imp_draws.squeeze(0)
        imp_x2 = problem.to2d(imp)
        imp_nearest = torch.cdist(imp_x2, centers_eval).argmin(1)
        hit = (imp_nearest == y_test_eval).float()

    acc = float(hit.mean())
    acc_lo = float(hit[lo].mean()) if lo.any() else 0.0

    metrics = dict(
        modes=modes,
        hq=hq,
        mass_tv=mass_tv,
        off_plane=off_plane,
        mask_tv=mask_tv,
        acc=acc,
        acc_lo=acc_lo,
    )

    limits = [
        ("modes", 100, 100),
        ("hq", 0.90, None),
        ("mass_tv", None, 0.10),
        ("off_plane", None, 0.04),
        ("mask_tv", None, 0.05),
        ("acc", 0.96, None),
        ("acc_lo", 0.70, None),
    ]
    failures = [item for name, lo_val, hi_val in limits for item in _bound(metrics, name, lo_val, hi_val)]
    views = [
        view("Generated 8D data projected to 2D grid", centers, x2[:1024], "scatter"),
        view("Mask pattern distribution vs true", problem.probs, emp.float(), "bar"),
        view("Imputed test data projected to 2D", centers, imp_x2[:1024], "scatter"),
    ]
    return observation(metrics, failures, views)


@torch.no_grad()
def score_misgan_neural_imputer(predict_fn, problem=None, eval_seed=99123):
    if problem is None:
        problem = misgan.Problem("block", 20000, 10000, 0, "cpu")
    
    rng = _rng(eval_seed, device=problem.x_test.device)
    if callable(predict_fn):
        draws = predict_fn(problem.x_test, problem.m_test, rng)
    else:
        draws = predict_fn
    
    if draws.ndim == 2:
        draws = draws.unsqueeze(0)
    
    if not torch.all(torch.isfinite(draws)):
        metrics = dict(
            observed_max_error=float("inf"),
            acc=0.0,
            acc_lo=0.0,
            missing_std_ratio=0.0,
        )
        failures = ["observed_max_error", "acc", "acc_lo", "missing_std_ratio", "nonfinite_output"]
        return observation(metrics, failures, [])

    draws = draws.to(device=problem.x_test.device, dtype=problem.x_test.dtype)
    obs_err = float(((draws - problem.x_test[None]) * problem.m_test[None]).abs().max())

    x2 = problem.to2d(draws.reshape(-1, 8))
    centers = misgan.grid_centers(device=draws.device, dtype=torch.float32)
    nearest = torch.cdist(x2, centers).argmin(1).reshape(len(draws), len(problem.x_test))
    hit = (nearest == problem.y_test[None]).float()
    lo = problem.m_test.sum(1) <= 1
    acc = float(hit.mean())
    acc_lo = float(hit[:, lo].mean()) if lo.any() else 0.0

    ref_rng = _rng(eval_seed + 1, device=problem.x_test.device)
    _, ref_draws = misgan.bayes_posterior(problem, problem.x_test, problem.m_test, draws=len(draws), generator=ref_rng)
    miss = 1 - problem.m_test
    nmiss = miss.sum().clamp_min(1)
    cand_std = (draws.std(0, correction=0) * miss).sum() / nmiss
    ref_std = (ref_draws.std(0, correction=0) * miss).sum() / nmiss
    missing_std_ratio = float(cand_std / ref_std.clamp_min(1e-6))

    metrics = dict(
        observed_max_error=obs_err,
        acc=acc,
        acc_lo=acc_lo,
        missing_std_ratio=missing_std_ratio,
    )

    limits = [
        ("observed_max_error", None, 1e-6),
        ("acc", 0.95, None),
        ("acc_lo", 0.65, None),
        ("missing_std_ratio", 0.70, 1.30),
    ]
    failures = [item for name, lo_val, hi_val in limits for item in _bound(metrics, name, lo_val, hi_val)]
    views = [
        view("Neural imputer ambiguous draws projected to 2D", centers, x2[:1024], "scatter"),
    ]
    return observation(metrics, failures, views)


def build_mcar_p50_problem(n_train=20000, n_test=10000, seed=0, device="cpu"):
    """MCAR p=0.5 problem where all-zero mask rows are resampled so >=1 coordinate is observed."""
    problem = misgan.Problem("mcar_p50", n_train, n_test, seed, device)
    rng_resample = torch.Generator(device=problem.m_train.device).manual_seed(int(seed) + 777)
    for attr in ("m_train", "m_test"):
        m = getattr(problem, attr).clone()
        empty = (m.sum(1) == 0).nonzero(as_tuple=False).squeeze(1)
        while len(empty) > 0:
            m[empty] = (torch.rand(len(empty), misgan.DIM, generator=rng_resample, device=m.device) >= 0.5).float()
            empty = (m.sum(1) == 0).nonzero(as_tuple=False).squeeze(1)
        setattr(problem, attr, m)
    problem.x_train = problem.x_train_full * problem.m_train
    p_zero = float(problem.probs[0])
    problem.probs = problem.probs.clone()
    problem.probs[0] = 0.0
    problem.probs = problem.probs / (1.0 - p_zero)
    problem.missing_rate = (problem.probs[:, None] * (1 - (torch.arange(2 ** misgan.DIM)[:, None] // misgan.BITS) % 2)).sum(0)
    return problem


@torch.no_grad()
def score_misgan_mcar_p50_recovery(x, m, problem=None, eval_seed=99123, use_ppost=True):
    if problem is None:
        problem = build_mcar_p50_problem(20000, 10000, 0, "cpu")

    if not torch.all(torch.isfinite(x)) or not torch.all(torch.isfinite(m)):
        metrics = dict(
            modes=0,
            hq=0.0,
            mass_tv=1.0,
            off_plane=float("inf"),
            mask_tv=1.0,
            acc=0.0,
            impute_rmse=float("inf"),
        )
        failures = ["modes", "hq", "mass_tv", "off_plane", "mask_tv", "acc", "impute_rmse", "nonfinite_output"]
        return observation(metrics, failures, [])

    off_plane = problem.off_plane(x)

    x2 = problem.to2d(x)
    centers = misgan.grid_centers(device=x.device, dtype=torch.float32)
    dists = torch.cdist(x2, centers)
    d, nearest = dists.min(1)
    hq = float((d <= 0.09).float().mean())
    hq_counts = torch.bincount(nearest[d <= 0.09], minlength=100)
    modes = int((hq_counts >= 10).sum())
    all_counts = torch.bincount(nearest, minlength=100).float()
    mass_tv = float(0.5 * (all_counts / len(x) - 0.01).abs().sum())

    hard_m = (m > 0.5).float()
    emp_obs_rate = hard_m.mean(0).double().cpu()
    target_obs_rate = (1.0 - problem.missing_rate).double().cpu()
    mask_tv = float((emp_obs_rate - target_obs_rate).abs().mean())

    centers_eval = centers.to(device=problem.x_test.device, dtype=problem.x_test.dtype)
    y_test_eval = problem.y_test.to(device=problem.x_test.device)
    x_test = problem.x_test
    m_test = problem.m_test
    miss = (1.0 - m_test).clamp_min(0.0)

    if use_ppost:
        k = misgan_ppost_indices(x, x_test, m_test, sigma=0.005, draws=4, generator=_rng(eval_seed))
        hit_list = []
        imp_draws = []
        imp_x2 = None
        for draw_idx in range(k.shape[0]):
            x_k = x[k[draw_idx]].to(device=x_test.device, dtype=x_test.dtype)
            imp_d = m_test * x_test + (1.0 - m_test) * x_k
            imp_draws.append(imp_d)
            imp_x2_d = problem.to2d(imp_d)
            if draw_idx == 0:
                imp_x2 = imp_x2_d
            imp_nearest = torch.cdist(imp_x2_d, centers_eval).argmin(1)
            hit_list.append((imp_nearest == y_test_eval).float())
        hit = torch.stack(hit_list, dim=0).mean(dim=0)
        imp_mean = torch.stack(imp_draws, dim=0).mean(dim=0)
    else:
        _, imp_draws_t = misgan.bayes_posterior(problem, x_test, m_test, draws=4, generator=_rng(eval_seed, device=x_test.device))
        imp_mean = imp_draws_t.mean(0)
        imp_x2 = problem.to2d(imp_draws_t[0])
        imp_nearest = torch.cdist(problem.to2d(imp_mean), centers_eval).argmin(1)
        hit = (imp_nearest == y_test_eval).float()

    acc = float(hit.mean())
    row_mse = ((imp_mean - x_test).pow(2) * miss).sum(1) / miss.sum(1).clamp_min(1.0)
    impute_rmse = float(row_mse.sqrt().mean())

    metrics = dict(
        modes=modes,
        hq=hq,
        mass_tv=mass_tv,
        off_plane=off_plane,
        mask_tv=mask_tv,
        acc=acc,
        impute_rmse=impute_rmse,
    )

    limits = [
        ("modes", 90, 100),
        ("hq", 0.85, None),
        ("mass_tv", None, 0.12),
        ("off_plane", None, 0.05),
        ("mask_tv", None, 0.05),
        ("acc", 0.90, None),
        ("impute_rmse", None, 0.12),
    ]
    failures = [item for name, lo_val, hi_val in limits for item in _bound(metrics, name, lo_val, hi_val)]
    views = [
        view("Generated 8D data projected to 2D grid", centers, x2[:1024], "scatter"),
        view("Coordinate observation rates vs MCAR target", target_obs_rate.float(), emp_obs_rate.float(), "bar"),
        view("Imputed test data projected to 2D", centers, imp_x2[:1024], "scatter"),
    ]
    return observation(metrics, failures, views)


def build_mcar_p80_anchored_problem(n_train=20000, n_test=10000, seed=0, device="cpu", anchor_rate=0.10):
    """Extreme MCAR p=0.80 missingness with 10% complete anchor rows and anchor-assisted query panel."""
    problem = misgan.Problem("mcar_p80", n_train, n_test, seed, device)
    raw_train = problem.raw(problem.x_train_full)
    raw_test = problem.raw(problem.x_test)
    raw_ref = problem.raw(problem.x_ref)

    rng_anchor = torch.Generator(device=problem.m_train.device).manual_seed(int(seed) + 888)
    anchor_train = torch.rand(len(problem.m_train), generator=rng_anchor, device=problem.m_train.device) < anchor_rate
    m_train = problem.m_train.clone()
    m_train[anchor_train] = 1.0

    # Anchor-assisted MCAR query panel: 10% complete anchors + MCAR queries with >=2 observed coordinates
    anchor_test = torch.rand(len(problem.m_test), generator=rng_anchor, device=problem.m_test.device) < anchor_rate
    m_test = problem.m_test.clone()
    m_test[anchor_test] = 1.0
    underdetermined = (m_test.sum(1) < 2).nonzero(as_tuple=False).squeeze(1)
    if len(underdetermined) > 0:
        scores = torch.rand(len(underdetermined), misgan.DIM, generator=rng_anchor, device=m_test.device)
        top2 = scores.topk(2, dim=1).indices
        m_test[underdetermined] = 0.0
        m_test[underdetermined.unsqueeze(1), top2] = 1.0

    m_train_d = m_train.double().cpu()
    count = m_train_d.sum(0).clamp_min(1.0)
    problem.mean = (raw_train.cpu() * m_train_d).sum(0) / count
    problem.std = (((raw_train.cpu() - problem.mean) ** 2 * m_train_d).sum(0) / count).sqrt()

    f32 = dict(device=problem.device, dtype=torch.float32)
    problem.x_train_full = problem.standardize(raw_train.cpu()).to(**f32)
    problem.m_train = m_train.to(**f32)
    problem.x_train = problem.x_train_full * problem.m_train
    problem.x_test = problem.standardize(raw_test.cpu()).to(**f32)
    problem.m_test = m_test.to(**f32)
    problem.x_ref = problem.standardize(raw_ref.cpu()).to(**f32)

    base_probs = misgan.pattern_probs("mcar_p80") * (1.0 - anchor_rate)
    base_probs[-1] += anchor_rate
    problem.probs = base_probs
    problem.missing_rate = (problem.probs[:, None] * (1 - (torch.arange(2 ** misgan.DIM)[:, None] // misgan.BITS) % 2)).sum(0)
    return problem


@torch.no_grad()
def score_misgan_mcar_p80_anchored(x, m=None, predict_fn=None, problem=None, eval_seed=99123):
    if problem is None:
        problem = build_mcar_p80_anchored_problem(20000, 10000, 0, "cpu")

    if isinstance(x, tuple):
        if len(x) == 3:
            x, m, predict_fn = x
        elif len(x) == 2:
            x, m = x

    if not torch.all(torch.isfinite(x)) or (m is not None and not torch.all(torch.isfinite(m))):
        metrics = dict(
            modes=0,
            hq=0.0,
            mass_tv=1.0,
            off_plane=float("inf"),
            imp_acc=0.0,
            imp_off_plane=float("inf"),
            impute_rmse=float("inf"),
        )
        failures = ["modes", "hq", "mass_tv", "off_plane", "imp_acc", "imp_off_plane", "impute_rmse", "nonfinite_output"]
        return observation(metrics, failures, [])

    rng = _rng(eval_seed, device=problem.x_test.device)
    if predict_fn is not None:
        draws = predict_fn(problem.x_test, problem.m_test, rng) if callable(predict_fn) else predict_fn
        if draws.ndim == 2:
            draws = draws.unsqueeze(0)
    else:
        k = misgan_ppost_indices(x, problem.x_test, problem.m_test, sigma=0.005, draws=4, generator=_rng(eval_seed))
        imp_list = []
        for draw_idx in range(k.shape[0]):
            x_k = x[k[draw_idx]].to(device=problem.x_test.device, dtype=problem.x_test.dtype)
            imp_list.append(problem.m_test * problem.x_test + (1.0 - problem.m_test) * x_k)
        draws = torch.stack(imp_list, dim=0)

    if not torch.all(torch.isfinite(draws)):
        metrics = dict(
            modes=0,
            hq=0.0,
            mass_tv=1.0,
            off_plane=float("inf"),
            imp_acc=0.0,
            imp_off_plane=float("inf"),
            impute_rmse=float("inf"),
        )
        failures = ["modes", "hq", "mass_tv", "off_plane", "imp_acc", "imp_off_plane", "impute_rmse", "nonfinite_output"]
        return observation(metrics, failures, [])

    draws = draws.to(device=problem.x_test.device, dtype=problem.x_test.dtype)
    off_plane = problem.off_plane(x)
    x2 = problem.to2d(x)
    centers = misgan.grid_centers(device=x.device, dtype=torch.float32)
    dists = torch.cdist(x2, centers)
    d, nearest = dists.min(1)
    hq = float((d <= 0.09).float().mean())
    hq_counts = torch.bincount(nearest[d <= 0.09], minlength=100)
    modes = int((hq_counts >= 10).sum())
    all_counts = torch.bincount(nearest, minlength=100).float()
    mass_tv = float(0.5 * (all_counts / len(x) - 0.01).abs().sum())

    centers_eval = centers.to(device=draws.device, dtype=draws.dtype)
    y_test_eval = problem.y_test.to(device=draws.device)
    flat_draws = draws.reshape(-1, 8)
    imp_x2_all = problem.to2d(flat_draws)
    imp_nearest = torch.cdist(imp_x2_all, centers_eval).argmin(1).reshape(len(draws), len(problem.x_test))
    imp_acc = float((imp_nearest == y_test_eval[None]).float().mean())
    imp_off_plane = problem.off_plane(flat_draws)

    imp_mean = draws.mean(0)
    miss = (1.0 - problem.m_test).clamp_min(0.0)
    row_mse = ((imp_mean - problem.x_test).pow(2) * miss).sum(1) / miss.sum(1).clamp_min(1.0)
    impute_rmse = float(row_mse.sqrt().mean())

    metrics = dict(
        modes=modes,
        hq=hq,
        mass_tv=mass_tv,
        off_plane=off_plane,
        imp_acc=imp_acc,
        imp_off_plane=imp_off_plane,
        impute_rmse=impute_rmse,
    )

    limits = [
        ("modes", 80, 100),
        ("hq", 0.80, None),
        ("mass_tv", None, 0.15),
        ("off_plane", None, 0.06),
        ("imp_acc", 0.65, None),
        ("imp_off_plane", None, 0.06),
        ("impute_rmse", None, 0.18),
    ]
    failures = [item for name, lo_val, hi_val in limits for item in _bound(metrics, name, lo_val, hi_val)]
    views = [
        view("Generated 8D data projected to 2D grid", centers, x2[:1024], "scatter"),
        view("Anchor-assisted neural imputer projected to 2D", centers, imp_x2_all[:1024], "scatter"),
    ]
    return observation(metrics, failures, views)


class MisGANBlockRecoveryTask:
    """Exact incomplete-data law for the 2-pair block recovery gate."""
    name = "misgan-block-recovery"
    paired = False

    def __init__(self):
        self.problem = misgan.Problem("block", 20000, 10000, 0, "cpu")

    def exact_law(self, n, rng):
        idx = torch.randint(len(self.problem.m_train), (n,), generator=rng)
        return self.problem.x_train_full[idx], self.problem.m_train[idx]

    def evaluate(self, predict, n, seed):
        x, m = predict(n, _rng(seed))
        return score_misgan_recovery(x, m, self.problem, eval_seed=seed, use_ppost=True)


class MisGANBlockNeuralImputerTask:
    """Bayes imputation witness for the 3-pair neural-imputer gate."""
    name = "misgan-block-neural-imputer"
    paired = False

    def __init__(self):
        self.problem = misgan.Problem("block", 20000, 10000, 0, "cpu")

    def exact_law(self, x, m, rng):
        _, samples = misgan.bayes_posterior(self.problem, x, m, draws=16, generator=rng)
        return samples

    def evaluate(self, predict, n, seed):
        # The declared scorer uses the fixed held-out test panel, same as the fixture.
        return score_misgan_neural_imputer(predict, self.problem, eval_seed=seed)


class MisGANMCARP50RecoveryTask(MisGANBlockRecoveryTask):
    """Exact incomplete-data law for the 2-pair MCAR p=0.5 recovery gate."""
    name = "misgan-mcar-p50-recovery"
    paired = False

    def __init__(self):
        self.problem = build_mcar_p50_problem(20000, 10000, 0, "cpu")

    def evaluate(self, predict, n, seed):
        x, m = predict(n, _rng(seed))
        return score_misgan_mcar_p50_recovery(x, m, self.problem, eval_seed=seed, use_ppost=True)


class MisGANMCARP80AnchoredTask(MisGANBlockRecoveryTask):
    """Exact incomplete-data law + Bayes imputer witness for the 3-pair MCAR p=0.80 anchored gate."""
    name = "misgan-mcar-p80-anchored"
    paired = False

    def __init__(self):
        self.problem = build_mcar_p80_anchored_problem(20000, 10000, 0, "cpu")

    def exact_law(self, n, rng):
        idx = torch.randint(len(self.problem.m_train), (n,), generator=rng)
        x = self.problem.x_train_full[idx]
        m = self.problem.m_train[idx]

        def predict_fn(x_q, m_q, r):
            _, samples = misgan.bayes_posterior(self.problem, x_q, m_q, draws=8, generator=r)
            return samples

        return x, m, predict_fn

    def evaluate(self, predict, n, seed):
        out = predict(n, _rng(seed))
        return score_misgan_mcar_p80_anchored(out, problem=self.problem, eval_seed=seed)


class _ModuleBundle(nn.Module):
    """One public module over the pair networks. Parameters stay the originals."""

    def __init__(self, **modules):
        super().__init__()
        for name, module in modules.items():
            self.add_module(name, module)


def _count_critic_steps(optimizer):
    """Record each real critic Adam step. These hosts keep that Adam update."""
    from particlegan.grad_regularizers import CriticStepRecord
    optimizer.record = CriticStepRecord()

    def _after(opt, args, kwargs):
        opt.record.calls += 1
        opt.record.record_step(opt)

    optimizer.register_step_post_hook(_after)


class _FixtureRNGStreams:
    """Isolated deterministic RNG stream forks for constructor, data, latent, and noise."""

    def __init__(self, seed: int, device: torch.device | str = "cpu"):
        self.seed = int(seed)
        self.device = torch.device(device)
        self._counters: dict[str, int] = {}

    def fork_seed(self, name: str) -> int:
        idx = self._counters.get(name, 0)
        self._counters[name] = idx + 1
        offset = sum((i + 1) * ord(ch) for i, ch in enumerate(name)) + 1009 * (idx + 1)
        return self.seed + offset

    def fork(self, name: str) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(self.fork_seed(name))


class TemperatureSigmoid(nn.Module):
    """Sigmoid with temperature scaling tau_m: sigmoid(x / tau_m)."""

    def __init__(self, temperature: float = 0.66):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x / self.temperature)


class FourierMLPDiscriminator(nn.Module):
    """MLP discriminator with coordinate Fourier features and optional GAIN partial-hint conditioning."""

    def __init__(
        self,
        in_dim: int = 8,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        fourier: int = 2,
        hint_dim: int = 0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.fourier = int(fourier)
        self.hint_dim = int(hint_dim)
        dim = self.in_dim + (2 * self.fourier * self.in_dim if self.fourier > 0 else 0) + self.hint_dim
        if self.fourier > 0:
            freqs = torch.pi * (2.0 ** torch.arange(self.fourier, dtype=torch.float32))
            self.register_buffer("freqs", freqs)
        else:
            self.freqs = None
        layers = []
        for i in range(n_hidden):
            layers.append(nn.Linear(dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, hint: torch.Tensor | None = None) -> torch.Tensor:
        if self.hint_dim > 0:
            if hint is None and x.shape[1] == self.in_dim + self.hint_dim:
                x, hint = x[:, : self.in_dim], x[:, self.in_dim :]
            elif hint is None:
                hint = torch.full((x.shape[0], self.hint_dim), 0.5, device=x.device, dtype=x.dtype)
        if self.fourier > 0 and self.freqs is not None:
            xf = x.unsqueeze(-1) * self.freqs
            h = torch.cat([x, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)
        else:
            h = x
        if self.hint_dim > 0 and hint is not None:
            h = torch.cat([h, hint], dim=1)
        return self.net(h)


def _misgan_update(method):
    """Own stochastic operations and reject checkpoints of interrupted updates."""
    @wraps(method)
    def step(self):
        if self.completed_steps >= self.limit:
            raise RuntimeError("declared execution cap reached")
        if not self._checkpoint_ready:
            raise RuntimeError("interrupted MisGAN update requires a complete checkpoint restore")
        devices = [self.device.index] if self.device.type == "cuda" else []
        self._checkpoint_ready = False
        with torch.random.fork_rng(devices=devices), torch.autograd.set_multithreading_enabled(False):
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self.model_rng.get_state(), self.device)
            else:
                torch.set_rng_state(self.model_rng.get_state())
            try:
                result = method(self)
            finally:
                state = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else torch.get_rng_state()
                self.model_rng.set_state(state)
        self._checkpoint_ready = True
        return result
    return step


class _MisGANBaseFixture:
    api_components = (
        "Recipe.make_prior",
        "Recipe.make_optimizers",
        "Recipe.make_loss",
        "Recipe.make_critic_penalty",
        "Recipe.prior_regularization",
        "scale_learning_rates",
    )
    hint_rate: float = 0.50

    def _sampling_witness(self, scored, *, posterior_resampling):
        result = dict(scored)
        result["observed_sampling"] = dict(
            contract=MISGAN_VERSION, scoring_weights="live",
            sampling_law="public_prior_without_output_noise", eval_output_noise="clean",
            predictor="G_x + ppost" if posterior_resampling else "G_i",
            posterior_resampling=posterior_resampling)
        return result

    def _checkpoint_modules(self):
        names = ("G_x", "G_m", "G_i", "D_x", "D_m", "D_i",
                 "prior_x", "prior_m", "prior_i", "ema_G_x", "ema_G_m", "ema_G_i",
                 "ema_prior_x", "ema_prior_m", "ema_prior_i",
                 "regularizer_x", "regularizer_m", "regularizer_i")
        return {name: getattr(self, name) for name in names if getattr(self, name, None) is not None}

    def _checkpoint_state(self, *, allow_interrupted=False):
        if not allow_interrupted and not self._checkpoint_ready:
            raise RuntimeError("MisGAN checkpoints require a completed update boundary")
        modules = self._checkpoint_modules()
        penalties = {}
        for role in self.critic_optimizers:
            penalty = getattr(self, "penalty_" + role)
            cooler = getattr(penalty.regularizer, "cooler", None)
            penalties[role] = dict(record=penalty.optimizer.record.state_dict(),
                                   cooler=None if cooler is None else cooler.state_dict(),
                                   last_stats=penalty.last_stats)
        return deepcopy(dict(
            version=MISGAN_VERSION, case_id=self.case["id"], case=self.case, seed=self.seed,
            limit=self.limit, recipe=self.recipe.to_dict(), device=str(self.device),
            completed_steps=self.completed_steps, base_rates=self.base_rates,
            requires_grad={name: {key: p.requires_grad for key, p in module.named_parameters()}
                           for name, module in modules.items()},
            models={name: module.state_dict() for name, module in modules.items()},
            optimizers={"g": self.opt_g.state_dict(),
                        **{role: optimizer.state_dict() for role, optimizer in self.critic_optimizers.items()}},
            penalties=penalties,
            streams={name: getattr(self, name).get_state() for name in
                     ("data_rng", "latent_rng", "noise_rng", "model_rng")},
            constructor_counters=self._rng_streams._counters,
            last_losses=self.last_losses))

    def _restore_checkpoint(self, state):
        from particlegan.policy import _validate_optimizer_state
        expected = self._checkpoint_state(allow_interrupted=True)
        if not isinstance(state, dict) or state.get("version") != MISGAN_VERSION:
            raise ValueError("incomplete historical MisGAN checkpoint; resume with its original source")
        # Optimizer loaders may retain tensor references. Restoring must not
        # let subsequent training mutate the caller's saved snapshot.
        state = deepcopy(state)
        if state.keys() != expected.keys():
            raise ValueError("invalid MisGAN checkpoint fields")
        for key in ("case_id", "case", "seed", "limit", "recipe", "device", "requires_grad", "base_rates",
                    "constructor_counters"):
            if state[key] != expected[key]:
                raise ValueError(f"MisGAN checkpoint {key} differs from the declared host")
        steps = state["completed_steps"]
        if type(steps) is not int or not 0 <= steps <= self.limit:
            raise ValueError("invalid MisGAN checkpoint update count")
        modules = self._checkpoint_modules()
        optimizers = {"g": self.opt_g, **self.critic_optimizers}
        for key in ("models", "optimizers", "penalties", "streams"):
            if not isinstance(state[key], dict) or state[key].keys() != expected[key].keys():
                raise ValueError(f"invalid MisGAN checkpoint {key}")
        # Validate every object before changing a live model, optimizer or RNG.
        try:
            for name, module in modules.items():
                current, saved = module.state_dict(), state["models"][name]
                if not isinstance(saved, dict) or saved.keys() != current.keys():
                    raise ValueError("model fields differ")
                for key, value in current.items():
                    other = saved[key]
                    if isinstance(value, torch.Tensor):
                        if (not isinstance(other, torch.Tensor) or other.shape != value.shape
                                or other.dtype != value.dtype or not torch.isfinite(other).all()):
                            raise ValueError("model tensor differs")
                    elif other != value:
                        raise ValueError("model metadata differs")
                deepcopy(module).load_state_dict(saved)
                if name.startswith("regularizer_"):
                    stochastic = (saved if self.recipe.prior_regularizer_checkpoint(module) else None)
                    self.recipe.restore_prior_regularizer(module, stochastic, validate_only=True)
            for name, optimizer in optimizers.items():
                _validate_optimizer_state(optimizer, state["optimizers"][name])
                deepcopy(optimizer).load_state_dict(state["optimizers"][name])
            for role, values in state["penalties"].items():
                if not isinstance(values, dict) or values.keys() != expected["penalties"][role].keys():
                    raise ValueError("penalty fields differ")
                penalty = getattr(self, "penalty_" + role)
                optimizer_record = state["optimizers"][role].get("regularizer", {}).get("record")
                if optimizer_record != values["record"]:
                    raise ValueError("optimizer and penalty copies of critic history contradict")
                if values["record"].get("observed_steps") != steps:
                    raise ValueError("critic clock differs from the completed update count")
                deepcopy(penalty.optimizer.record).load_state_dict(values["record"])
                cooler = getattr(penalty.regularizer, "cooler", None)
                if cooler is None:
                    if values["cooler"] is not None:
                        raise ValueError("unexpected cooler history")
                else:
                    saved_cooler = values["cooler"]
                    current_cooler = expected["penalties"][role]["cooler"]
                    if not isinstance(saved_cooler, dict) or saved_cooler.keys() != current_cooler.keys():
                        raise ValueError("incomplete cooler state")
                    for key in current_cooler.keys() - {"ema_residual", "initialized"}:
                        if saved_cooler[key] != current_cooler[key]:
                            raise ValueError("cooler declaration differs")
                    if (type(saved_cooler["initialized"]) is not bool
                            or type(saved_cooler["ema_residual"]) not in (int, float)
                            or not math.isfinite(saved_cooler["ema_residual"])
                            or saved_cooler["ema_residual"] < 0):
                        raise ValueError("invalid cooler history")
                    deepcopy(cooler).load_state_dict(saved_cooler)
                if not isinstance(values["last_stats"], dict):
                    raise ValueError("invalid penalty diagnostics")
            for name, values in state["streams"].items():
                torch.Generator(device=getattr(self, name).device).set_state(values.cpu())
            if (not isinstance(state["last_losses"], dict)
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in state["last_losses"].values())):
                raise ValueError("invalid loss diagnostics")
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid complete MisGAN checkpoint state") from error
        for name, module in modules.items():
            module.load_state_dict(state["models"][name])
            if name.startswith("regularizer_"):
                stochastic = (state["models"][name] if self.recipe.prior_regularizer_checkpoint(module) else None)
                self.recipe.restore_prior_regularizer(module, stochastic)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        for role, values in state["penalties"].items():
            penalty = getattr(self, "penalty_" + role)
            penalty.optimizer.record.load_state_dict(values["record"])
            if values["cooler"] is not None:
                penalty.regularizer.cooler.load_state_dict(values["cooler"])
            penalty.last_stats = deepcopy(values["last_stats"])
            if hasattr(penalty.optimizer.record, "restore_device_state"):
                penalty.optimizer.record.restore_device_state(self.device)
        for name, values in state["streams"].items():
            getattr(self, name).set_state(values.cpu())
        self.completed_steps = steps
        self.last_losses = deepcopy(state["last_losses"])
        self._checkpoint_ready = True

    def _sample_hint(self, m: torch.Tensor) -> torch.Tensor:
        rate = float(getattr(self, "hint_rate", 0.50))
        b = (torch.rand(m.shape, generator=self.noise_rng, device=m.device) < rate).float()
        return b * m + 0.5 * (1.0 - b)

    def _sample_prior_obj(self, prior, batch_size: int, generator=None):
        if prior is None:
            return None, None
        gen = self.latent_rng if generator is None else generator
        if isinstance(prior, (MoGParticlePrior, ManipulatedPrior)) or hasattr(prior, "sigma"):
            return prior.sample(batch_size, generator=gen, noise_generator=self.noise_rng)
        return prior.sample(batch_size, generator=gen)

    def _sample_prior(self, batch_size: int):
        return self._sample_prior_obj(self.prior_x, batch_size)

    def _make_default_recipe(self, recipe_name):
        eff_name = "bispf" if recipe_name is None else recipe_name
        base = get_recipe(eff_name)
        kw = dict(batch_size=self.batch_size, num_particles=self.particles, z_dim=self.z_dim)
        if base.continuous_policy is None:
            kw["total_steps"] = self.case["default_steps"]
        return get_recipe(eff_name, **kw)

    def _schedule_recipe(self):
        return self.recipe

    @staticmethod
    def _validate_supported_recipe(recipe):
        # Caller-owned multi-pair hosts have no UpdatePolicy owner. Reject its
        # mechanisms before any optimizer is built rather than dropping them.
        unsupported = []
        for name in ("continuous_policy", "row_evidence_gate", "particle_birth_death", "serve_average",
                     "input_noise_std", "output_noise_std", "latent_damping_max_rate",
                     "kinetic_transport_weight", "kinetic_transport_local_weight"):
            if getattr(recipe, name):
                unsupported.append(name)
        if recipe.lr_control == "stationarity":
            unsupported.append("lr_control=stationarity")
        if recipe.generator_gradient_response != "none":
            unsupported.append("generator_gradient_response")
        if recipe.prior_code_law != "table":
            unsupported.append("prior_code_law")
        for name, supported in (("model", "gan"), ("training_precision", "float32")):
            if getattr(recipe, name) != supported:
                unsupported.append(name)
        if unsupported:
            from .api_contract import UnsupportedRecipeOverrides
            raise UnsupportedRecipeOverrides("MisGAN host cannot execute the intact recipe mechanisms: "
                             + ", ".join(unsupported) + "; use a separately declared supported candidate")

    def _prior_role_reg_loss(self, prior, regularizer, idx: torch.Tensor) -> torch.Tensor:
        if prior is None or idx is None:
            return torch.zeros((), device=self.device)
        base = unwrap_prior(prior)
        if hasattr(self.opt_g, "set_sampled_rows") and hasattr(base, "z") and base.z.requires_grad:
            self.opt_g.set_sampled_rows(base.z, idx)
        loss_p = torch.zeros((), device=self.device)
        if hasattr(base, "z") and regularizer is not None:
            loss_p = loss_p + self.recipe.prior_regularization(
                base.z, regularizer=regularizer, prior=prior
            )
        return loss_p

    def _prior_reg_loss(self, idx_g: torch.Tensor) -> torch.Tensor:
        return self._prior_role_reg_loss(self.prior_x, self.regularizer_x, idx_g)

    @torch.no_grad()
    def _update_ema(self, average, current, decay: float):
        for target, source in zip(average.parameters(), current.parameters()):
            target.lerp_(source, 1.0 - decay)
        for target, source in zip(average.buffers(), current.buffers()):
            target.copy_(source)

    def _prior_law(self, recipe):
        law = self.case.get("initial_prior")
        required = {"kind", "sigma", "standardize", "init_std", "initializer", "sampling", "transform"}
        if not isinstance(law, dict) or law.keys() != required:
            raise ValueError("MisGAN requires a complete task-owned initial_prior declaration")
        if (law["kind"] not in ("particle_cloud", "mog")
                or type(law["standardize"]) is not bool
                or law["initializer"] != "deterministic_orthogonal"
                or law["sampling"] != "uniform_with_replacement"):
            raise ValueError("unsupported MisGAN task prior law")
        for name in ("sigma", "init_std"):
            if type(law[name]) not in (int, float) or not math.isfinite(law[name]) or law[name] < 0:
                raise ValueError("invalid MisGAN task prior scale")
        kind = "mog" if law["kind"] == "mog" else "particles"
        # Recipe.standardize is inactive for particle storage; the cloud's
        # actual law remains raw locations, as the task explicitly declares.
        if recipe.prior_kind != kind or (kind == "mog" and recipe.standardize != law["standardize"]):
            raise ValueError("MisGAN recipe/task prior storage or standardization conflict")
        if law["kind"] == "particle_cloud" and (law["sigma"] != 0 or law["standardize"]):
            raise ValueError("a particle cloud has sigma=0 and standardize=false")
        if law["kind"] == "mog" and law["sigma"] <= 0:
            raise ValueError("this MoG task requires positive storage sigma")
        transform = law["transform"]
        if transform is not None:
            if (law["kind"] != "particle_cloud" or not isinstance(transform, dict)
                    or transform.keys() != {"kind", "sigma", "kappa_radial"}
                    or transform["kind"] != "normal_noise" or transform["kappa_radial"] != 1.
                    or type(transform["sigma"]) not in (int, float)
                    or not math.isfinite(transform["sigma"]) or transform["sigma"] <= 0):
                raise ValueError("unsupported MisGAN task sampling transform")
            if recipe.kappa_radial != transform["kappa_radial"]:
                raise ValueError("MisGAN recipe/task radial sampling law conflict")
        return law

    def _build_task_prior(self, recipe, role, *, streams=None):
        law = self._prior_law(recipe)
        count, dimension, component = {
            "x": (self.particles, self.z_dim, "constructor"),
            "m": (64, self.z_dim, "constructor_m"),
            "i": (64, 4, "constructor_i"),
        }[role]
        streams = self._rng_streams if streams is None else streams
        options = dict(sigma=law["sigma"], sigma_rel=0., init_std=law["init_std"],
                       standardize=law["standardize"], device=self.device,
                       generator=streams.fork(component))
        transform = law["transform"]
        if transform is not None:
            options["sigma"] = transform["sigma"]
        prior_recipe = recipe.replace(prior_update="learned", num_particles=count, z_dim=dimension)
        prior = prior_recipe.make_prior(**options)
        init.deterministic_orthogonal_(unwrap_prior(prior), seed=streams.fork_seed(component))
        return prior

    def _validate_task_prior(self, prior, recipe, role):
        from .api_contract import UnsupportedRecipeOverrides
        if isinstance(prior, ManipulatedPrior) and prior.has_regularizer:
            raise UnsupportedRecipeOverrides("MisGAN host lacks complete manipulator regularization context")
        law = self._prior_law(recipe)
        base = unwrap_prior(prior)
        expected_class = MoGParticlePrior if law["kind"] == "mog" else ParticlePrior
        count, dimension = {"x": (self.particles, self.z_dim), "m": (64, self.z_dim), "i": (64, 4)}[role]
        if type(base) is not expected_class or base.z.shape != (count, dimension):
            raise ValueError("MisGAN supplied prior storage kind, shape or capacity differs from the task")
        if not math.isclose(float(base.init_std), law["init_std"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("MisGAN supplied prior initialization scale differs from the task")
        if bool(getattr(base, "standardize", False)) != law["standardize"]:
            raise ValueError("MisGAN supplied prior standardization differs from the task")
        sigma = float(base.sigma) if isinstance(base, MoGParticlePrior) else 0.
        if not math.isclose(sigma, law["sigma"], rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("MisGAN supplied prior sigma differs from the task")
        manipulators = list(prior.manipulators) if isinstance(prior, ManipulatedPrior) else []
        transform = law["transform"]
        if transform is None:
            if manipulators:
                raise ValueError("MisGAN supplied prior adds an undeclared sampling transform")
        elif (len(manipulators) != 1 or type(manipulators[0]) is not NormalNoiseManipulator
              or not math.isclose(manipulators[0].sigma, transform["sigma"], rel_tol=0, abs_tol=1e-12)
              or manipulators[0].kappa_radial != transform["kappa_radial"]):
            raise ValueError("MisGAN supplied prior sampling transform differs from the task")
        reference = self._build_task_prior(recipe, role, streams=_FixtureRNGStreams(self.seed, self.device))
        if (base.z.dtype != unwrap_prior(reference).z.dtype
                or not torch.equal(base.z.detach().cpu(), unwrap_prior(reference).z.detach().cpu())):
            raise ValueError("MisGAN supplied prior locations/dtype differ from the task's deterministic initializer")

    def rebind_trainer(self, recipe, prior=None):
        if self.completed_steps:
            raise ValueError("rebinding a trained MisGAN fixture requires a new execution identity")
        self._validate_supported_recipe(recipe)
        if (recipe.batch_size != self.batch_size or recipe.num_particles != self.particles
                or recipe.z_dim != self.z_dim):
            raise ValueError("MisGAN recipe resources must match the explicitly declared host resources")
        self._prior_law(recipe)
        supplied = prior if prior is not None else getattr(self, "prior_x", None)
        if supplied is not None:
            self._validate_task_prior(supplied, recipe, "x")
        self.recipe = recipe
        if supplied is None:
            self.prior_x = self._build_task_prior(recipe, "x")
        else:
            self.prior_x = supplied.to(self.device)
            if "constructor" not in self._rng_streams._counters:
                # Validate supplied and constructed priors under the same named
                # constructor binding, preserving subsequent stream identity.
                self._rng_streams.fork_seed("constructor")
                self._rng_streams.fork_seed("constructor")
        self._validate_task_prior(self.prior_x, recipe, "x")
        self.recipe.apply_prior_policy(self.prior_x)
        self.prior = self.prior_x
        for role in ("m", "i"):
            if role == "i" and not hasattr(self, "G_i"):
                self.prior_i = None
                continue
            bound = getattr(self, "prior_" + role, None)
            if bound is None:
                bound = self._build_task_prior(recipe, role)
            self._validate_task_prior(bound, recipe, role)
            self.recipe.apply_prior_policy(bound)
            setattr(self, "prior_" + role, bound)
        self.priors = _ModuleBundle(**{"prior_" + role: getattr(self, "prior_" + role)
                                     for role in ("x", "m", "i") if getattr(self, "prior_" + role, None) is not None})

        self.loss = self.recipe.make_loss()
        self.loss_x = self.loss
        self.loss_m = self.recipe.make_loss()
        self.loss_i = self.recipe.make_loss() if hasattr(self, "D_i") else None
        self.regularizer_x = self.recipe.make_prior_regularizer(prior=self.prior_x)
        self.regularizer_m = self.recipe.make_prior_regularizer(prior=self.prior_m)
        self.regularizer = self.regularizer_x
        if self.prior_i is not None:
            self.regularizer_i = self.recipe.make_prior_regularizer(prior=self.prior_i)
        else:
            self.regularizer_i = None

        normalized = self.recipe.optimizer_family not in ("adam", "formulation")
        groups = [
            {"params": [p for p in self.G_x.parameters() if p.requires_grad], "lr": self.recipe.lr, "role": "generator"},
            {"params": [p for p in self.G_m.parameters() if p.requires_grad], "lr": self.recipe.lr, "role": "generator"},
        ]
        if hasattr(self, "G_i") and self.G_i is not None:
            groups.append({"params": [p for p in self.G_i.parameters() if p.requires_grad], "lr": self.recipe.lr, "role": "generator"})

        for prior_obj in (self.prior_x, self.prior_m, getattr(self, "prior_i", None)):
            if prior_obj is None:
                continue
            base_p = unwrap_prior(prior_obj)
            base_params = [p for p in base_p.parameters() if p.requires_grad]
            base_ids = {id(p) for p in base_params}
            router_params = [p for p in prior_obj.parameters() if p.requires_grad and id(p) not in base_ids]
            if base_params:
                groups.append({
                    "params": base_params,
                    "lr": self.recipe.lr * self.recipe.prior_lr_mult,
                    "betas": self.recipe.prior_betas if self.recipe.prior_betas is not None else self.recipe.betas,
                    **({"role": "prior"} if normalized else {}),
                    **({"eps": self.recipe.prior_eps} if self.recipe.prior_eps is not None else {}),
                })
            if router_params:
                groups.append({
                    "params": router_params,
                    "lr": self.recipe.lr,
                    "betas": self.recipe.betas,
                    **({"role": "router"} if normalized else {}),
                })

        from particlegan.capabilities import prior_mechanisms
        mechanisms = prior_mechanisms(
            self.prior_x,
            latent_damping_max_rate=self.recipe.latent_damping_max_rate if self.recipe.prior_update == "learned" else 0.0,
            prior_beta1=(self.recipe.prior_betas or self.recipe.betas)[0],
        )
        base_prior_x = unwrap_prior(self.prior_x)
        latent_table = base_prior_x.z if (base_prior_x is not None and hasattr(base_prior_x, "z") and mechanisms["prior"]["a2_eligible"]) else None
        self.opt_g = self.recipe.make_generator_optimizer(groups, latent_table=latent_table, modules=[self.G], foreach=False, fused=False)
        self.opt_g.prior_mechanisms = mechanisms

        # Each adversarial pair owns the complete public critic state. Sharing
        # private record copies left the later pairs' clocks and thermal/anchor
        # histories detached from their optimizer updates.
        self.critic_optimizers = {}
        for role in ("x", "m", "i"):
            critic = getattr(self, "D_" + role, None)
            if critic is None:
                continue
            ema_critic = deepcopy(critic) if self.recipe.effective_critic_formulation != "bispf" else None
            optimizer = self.recipe.make_critic_optimizer(critic, ema_critic=ema_critic, foreach=False, fused=False)
            if not hasattr(optimizer, "record"):
                _count_critic_steps(optimizer)
            self.critic_optimizers[role] = optimizer
            penalty = self.recipe.make_critic_penalty(optimizer, collect_stats=True)
            setattr(self, "penalty_" + role, penalty)
            if self.recipe.effective_critic_formulation == "bispf":
                from particlegan.bispf import bind_thermal_source
                bind_thermal_source(optimizer.record, getattr(self, "loss_" + role),
                                    getattr(self, "regularizer_" + role))
        self.opt_d = self.critic_optimizers["x"]
        self.penalty_i = getattr(self, "penalty_i", None)
        self.base_rates = [[group["lr"] for group in opt.param_groups]
                           for opt in (self.opt_g, *self.critic_optimizers.values())]

        self.ema_G_x = deepcopy(self.G_x).eval().requires_grad_(False)
        self.ema_G_m = deepcopy(self.G_m).eval().requires_grad_(False)
        self.ema_prior_x = deepcopy(self.prior_x).eval().requires_grad_(False)
        self.ema_prior_m = deepcopy(self.prior_m).eval().requires_grad_(False)
        if hasattr(self, "G_i") and self.G_i is not None:
            self.ema_G_i = deepcopy(self.G_i).eval().requires_grad_(False)
        if getattr(self, "prior_i", None) is not None:
            self.ema_prior_i = deepcopy(self.prior_i).eval().requires_grad_(False)


class MisGANRecoveryFixture(_MisGANBaseFixture):
    def __init__(
        self,
        case=None,
        seed=DEFAULT_SEED,
        recipe_name="bispf",
        max_steps=None,
        device="cpu",
        *,
        recipe=None,
        prior=None,
        steps=None,
        batch_size=None,
        particles=256,
        z_dim=8,
    ):
        if case is None:
            case = deepcopy(CASES["api-misgan-block-recovery"])
        self.case = deepcopy(case)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.completed_steps = 0
        eff_max_steps = steps if steps is not None else max_steps
        self.limit = case["default_steps"] if eff_max_steps is None else eff_max_steps
        if type(self.limit) is not int or not 0 < self.limit <= case["default_steps"]:
            raise ValueError("max_steps must be in the declared budget")

        self.batch_size = int(batch_size if batch_size is not None else case["batch_size"])
        self.particles = int(particles)
        self.z_dim = int(z_dim)
        self._rng_streams = _FixtureRNGStreams(self.seed, device=self.device)

        if case["id"] == "api-misgan-mcar-p50-recovery":
            self.problem = build_mcar_p50_problem(20000, 10000, 0, device=self.device)
            self._scorer = score_misgan_mcar_p50_recovery
        else:
            self.problem = misgan.Problem("block", 20000, 10000, 0, device=self.device)
            self._scorer = score_misgan_recovery

        self.data_rng = _rng(self.seed + 10, device=self.device)
        self.latent_rng = _rng(self.seed + 11, device=self.device)
        self.noise_rng = _rng(self.seed + 12, device=self.device)
        self.model_rng = _rng(self.seed + 13, device=self.device)

        with construction_rng(self.seed, "cpu"):
            self.G_x = nn.Sequential(
                nn.Linear(self.z_dim, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 8),
            ).to(self.device)
            self.G_m = nn.Sequential(
                nn.Linear(self.z_dim, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 8),
                TemperatureSigmoid(0.66),
            ).to(self.device)
            self.D_x = FourierMLPDiscriminator(
                in_dim=8, hidden_dim=64, n_hidden=2, fourier=2, hint_dim=8
            ).to(self.device)
            self.D_m = nn.Sequential(
                nn.Linear(8, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 1),
            ).to(self.device)

        init.deterministic_orthogonal_(self.G_x, seed=self.seed + 1)
        init.deterministic_orthogonal_(self.G_m, seed=self.seed + 2)
        init.deterministic_orthogonal_(self.D_x, seed=self.seed + 3)
        init.deterministic_orthogonal_(self.D_m, seed=self.seed + 4)

        self.G = _ModuleBundle(G_x=self.G_x, G_m=self.G_m)
        self.D = _ModuleBundle(D_x=self.D_x, D_m=self.D_m)

        if recipe is None:
            recipe = self._make_default_recipe(recipe_name)
        self.prior = None
        self.rebind_trainer(recipe, prior=prior)
        self.last_losses = {}
        self._checkpoint_ready = True

    @_misgan_update
    def step(self):
        if self.completed_steps >= self.limit:
            raise RuntimeError("declared execution cap reached")
        scale_learning_rates(
            self.completed_steps,
            self._schedule_recipe(),
            [self.opt_g, *self.critic_optimizers.values()],
            self.base_rates,
            self.priors,
        )

        self.G_x.train(); self.G_m.train(); self.D_x.train(); self.D_m.train()
        bs = int(self.recipe.batch_size or self.batch_size)
        x_real, m_real, _ = self.problem.train_batch(bs, self.data_rng)

        z_x, idx_x = self._sample_prior_obj(self.prior_x, bs)
        z_m, idx_m = self._sample_prior_obj(self.prior_m, bs)
        x_fake = self.G_x(z_x)
        if isinstance(self.prior_x, ManipulatedPrior) and self.prior_x.postprocesses_generated:
            x_fake = self.prior_x.postprocess_generated(x_fake)
        m_fake = self.G_m(z_m)

        m_hard = (m_fake.detach() > 0.5).float()
        hint_real_dx = self._sample_hint(m_real)
        hint_fake_dx = self._sample_hint(m_hard)
        real_dx_in = torch.cat([x_real * m_real, hint_real_dx], dim=1)
        fake_dx_in = torch.cat([x_fake.detach() * m_hard, hint_fake_dx], dim=1)

        loss_dx = self.loss_x.d_loss(self.D_x(real_dx_in), self.D_x(fake_dx_in))
        loss_dm = self.loss_m.d_loss(self.D_m(m_real), self.D_m(m_fake.detach()))

        reg_dx = self.penalty_x(self.D_x, real_dx_in, fake_dx_in)
        reg_dm = self.penalty_m(self.D_m, m_real, m_fake.detach())
        loss_d = loss_dx + loss_dm + reg_dx + reg_dm

        for optimizer in self.critic_optimizers.values():
            optimizer.zero_grad()
        loss_d.backward()
        for optimizer in self.critic_optimizers.values():
            optimizer.step()

        fake_gx_in = torch.cat([x_fake * m_hard, hint_fake_dx], dim=1)

        with torch.no_grad():
            real_dx_logits = self.D_x(real_dx_in)
            real_dm_logits = self.D_m(m_real)
        loss_gx = self.loss_x.g_loss(self.D_x(fake_gx_in), real_dx_logits)
        loss_gm = self.loss_m.g_loss(self.D_m(m_fake), real_dm_logits)
        loss_adv = loss_gx + loss_gm

        loss_prior_x = self._prior_role_reg_loss(self.prior_x, self.regularizer_x, idx_x)
        loss_prior_m = self._prior_role_reg_loss(self.prior_m, self.regularizer_m, idx_m)
        loss_prior = loss_prior_x + loss_prior_m
        loss_g = loss_adv + loss_prior

        self.opt_g.zero_grad()
        if self.recipe.constraint_geometry_mode != "none":
            constraint_geometry_backward(loss_g, self.opt_g, (loss_adv,))
        else:
            loss_g.backward()
        self.opt_g.step()

        decay = float(self.recipe.ema_decay)
        if decay > 0.0:
            self._update_ema(self.ema_G_x, self.G_x, decay)
            self._update_ema(self.ema_G_m, self.G_m, decay)
            if hasattr(self, "ema_prior_x") and self.prior_x is not None:
                self._update_ema(self.ema_prior_x, self.prior_x, decay)
            if hasattr(self, "ema_prior_m") and self.prior_m is not None:
                self._update_ema(self.ema_prior_m, self.prior_m, decay)

        self.completed_steps += 1
        self.last_losses = dict(
            loss_g=float(loss_g.detach()),
            loss_d=float(loss_d.detach()),
            loss_gx=float(loss_gx.detach()),
            loss_gm=float(loss_gm.detach()),
            critic_penalty=float((reg_dx + reg_dm).detach()),
            loss_prior=float(loss_prior.detach()),
        )
        if not all(math.isfinite(v) for v in self.last_losses.values()):
            raise FloatingPointError("nonfinite actual GAN update")
        return dict(step=self.completed_steps, **self.last_losses)

    @torch.no_grad()
    def observe(self, n=4096, seed=99123):
        models = [self.G_x, self.G_m, self.ema_G_x, self.ema_G_m]
        flags = {m: m.training for model in models for m in model.modules()}
        try:
            for model in models:
                model.eval()
            gen = _rng(seed, device=self.device)
            noise_gen = _rng(seed + 1, device=self.device)
            prior_eval = self.prior_x
            prior_m_eval = self.prior_m
            if isinstance(prior_eval, (MoGParticlePrior, ManipulatedPrior)) or hasattr(prior_eval, "sigma"):
                z_x, _ = prior_eval.sample(n, generator=gen, noise_generator=noise_gen)
            else:
                z_x, _ = prior_eval.sample(n, generator=gen)
            if isinstance(prior_m_eval, (MoGParticlePrior, ManipulatedPrior)) or hasattr(prior_m_eval, "sigma"):
                z_m, _ = prior_m_eval.sample(n, generator=gen, noise_generator=noise_gen)
            else:
                z_m, _ = prior_m_eval.sample(n, generator=gen)
            x = self.G_x(z_x)
            if isinstance(prior_eval, ManipulatedPrior) and prior_eval.postprocesses_generated:
                x = prior_eval.postprocess_generated(x)
            m = self.G_m(z_m)
            return self._sampling_witness(
                self._scorer(x, m, self.problem, eval_seed=seed, use_ppost=True),
                posterior_resampling=True)
        finally:
            for mod, flag in flags.items():
                mod.training = flag

    def state_dict(self):
        return self._checkpoint_state()

    def load_state_dict(self, state):
        self._restore_checkpoint(state)


class MisGANNeuralImputerFixture(_MisGANBaseFixture):
    def __init__(
        self,
        case=None,
        seed=DEFAULT_SEED,
        recipe_name="bispf",
        max_steps=None,
        device="cpu",
        *,
        recipe=None,
        prior=None,
        steps=None,
        batch_size=None,
        particles=256,
        z_dim=8,
    ):
        if case is None:
            case = deepcopy(CASES["api-misgan-block-neural-imputer"])
        self.case = deepcopy(case)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.completed_steps = 0
        eff_max_steps = steps if steps is not None else max_steps
        self.limit = case["default_steps"] if eff_max_steps is None else eff_max_steps
        if type(self.limit) is not int or not 0 < self.limit <= case["default_steps"]:
            raise ValueError("max_steps must be in the declared budget")

        self.batch_size = int(batch_size if batch_size is not None else case["batch_size"])
        self.particles = int(particles)
        self.z_dim = int(z_dim)
        self._rng_streams = _FixtureRNGStreams(self.seed, device=self.device)
        self.is_anchored_mcar = (case["id"] == "api-misgan-mcar-p80-anchored")
        self.hint_rate = 0.50

        if self.is_anchored_mcar:
            self.problem = build_mcar_p80_anchored_problem(20000, 10000, 0, device=self.device)
        else:
            self.problem = misgan.Problem("block", 20000, 10000, 0, device=self.device)

        self.data_rng = _rng(self.seed + 10, device=self.device)
        self.latent_rng = _rng(self.seed + 11, device=self.device)
        self.noise_rng = _rng(self.seed + 12, device=self.device)
        self.model_rng = _rng(self.seed + 13, device=self.device)

        with construction_rng(self.seed, "cpu"):
            self.G_x = nn.Sequential(
                nn.Linear(self.z_dim, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 8),
            ).to(self.device)
            self.G_m = nn.Sequential(
                nn.Linear(self.z_dim, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 8),
                TemperatureSigmoid(0.66),
            ).to(self.device)
            self.G_i = nn.Sequential(
                nn.Linear(20, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 8),
            ).to(self.device)

            self.D_x = FourierMLPDiscriminator(
                in_dim=8, hidden_dim=64, n_hidden=2, fourier=2, hint_dim=8
            ).to(self.device)
            self.D_m = nn.Sequential(
                nn.Linear(8, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 1),
            ).to(self.device)
            self.D_i = nn.Sequential(
                nn.Linear(16, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 1),
            ).to(self.device)

        init.deterministic_orthogonal_(self.G_x, seed=self.seed + 1)
        init.deterministic_orthogonal_(self.G_m, seed=self.seed + 2)
        init.deterministic_orthogonal_(self.G_i, seed=self.seed + 3)
        init.deterministic_orthogonal_(self.D_x, seed=self.seed + 4)
        init.deterministic_orthogonal_(self.D_m, seed=self.seed + 5)
        init.deterministic_orthogonal_(self.D_i, seed=self.seed + 6)

        self.G = _ModuleBundle(G_x=self.G_x, G_m=self.G_m, G_i=self.G_i)
        self.D = _ModuleBundle(D_x=self.D_x, D_m=self.D_m, D_i=self.D_i)

        if recipe is None:
            recipe = self._make_default_recipe(recipe_name)
        self.prior = None
        self.rebind_trainer(recipe, prior=prior)
        self.last_losses = {}
        self._checkpoint_ready = True

    @_misgan_update
    def step(self):
        if self.completed_steps >= self.limit:
            raise RuntimeError("declared execution cap reached")
        scale_learning_rates(
            self.completed_steps,
            self._schedule_recipe(),
            [self.opt_g, *self.critic_optimizers.values()],
            self.base_rates,
            self.priors,
        )

        self.G_x.train(); self.G_m.train(); self.G_i.train()
        self.D_x.train(); self.D_m.train(); self.D_i.train()
        bs = int(self.recipe.batch_size or self.batch_size)
        x_real, m_real, _ = self.problem.train_batch(bs, self.data_rng)

        z_x, idx_x = self._sample_prior_obj(self.prior_x, bs)
        z_m, idx_m = self._sample_prior_obj(self.prior_m, bs)
        z_i, idx_i = self._sample_prior_obj(self.prior_i, bs)

        x_fake = self.G_x(z_x)
        if isinstance(self.prior_x, ManipulatedPrior) and self.prior_x.postprocesses_generated:
            x_fake = self.prior_x.postprocess_generated(x_fake)
        m_fake = self.G_m(z_m)

        cond = torch.cat([x_real * m_real, m_real, z_i], dim=1)
        raw_imp = self.G_i(cond)
        completed = m_real * x_real + (1.0 - m_real) * raw_imp

        m_hard = (m_fake.detach() > 0.5).float()
        hint_real_dx = self._sample_hint(m_real)
        hint_fake_dx = self._sample_hint(m_hard)
        real_dx_in = torch.cat([x_real * m_real, hint_real_dx], dim=1)
        fake_dx_in = torch.cat([x_fake.detach() * m_hard, hint_fake_dx], dim=1)

        loss_dx = self.loss_x.d_loss(self.D_x(real_dx_in), self.D_x(fake_dx_in))
        loss_dm = self.loss_m.d_loss(self.D_m(m_real), self.D_m(m_fake.detach()))

        if self.is_anchored_mcar:
            hint_fake = self._sample_hint(m_real)
            hint_real = self._sample_hint(torch.ones_like(m_real))
            real_di_in = torch.cat([x_fake.detach(), hint_real], dim=1)
            fake_di_in = torch.cat([completed.detach(), hint_fake], dim=1)
            cond_fake_gi = torch.cat([completed, hint_fake], dim=1)
        else:
            real_di_in = torch.cat([x_fake.detach(), m_real], dim=1)
            fake_di_in = torch.cat([completed.detach(), m_real], dim=1)
            cond_fake_gi = torch.cat([completed, m_real], dim=1)

        loss_di = self.loss_i.d_loss(self.D_i(real_di_in), self.D_i(fake_di_in))

        reg_dx = self.penalty_x(self.D_x, real_dx_in, fake_dx_in)
        reg_dm = self.penalty_m(self.D_m, m_real, m_fake.detach())
        reg_di = self.penalty_i(self.D_i, real_di_in, fake_di_in)
        loss_d = loss_dx + loss_dm + loss_di + reg_dx + reg_dm + reg_di

        for optimizer in self.critic_optimizers.values():
            optimizer.zero_grad()
        loss_d.backward()
        for optimizer in self.critic_optimizers.values():
            optimizer.step()

        fake_gx_in = torch.cat([x_fake * m_hard, hint_fake_dx], dim=1)

        with torch.no_grad():
            real_dx_logits = self.D_x(real_dx_in)
            real_dm_logits = self.D_m(m_real)
            real_di_logits = self.D_i(real_di_in)

        loss_gx = self.loss_x.g_loss(self.D_x(fake_gx_in), real_dx_logits)
        loss_gm = self.loss_m.g_loss(self.D_m(m_fake), real_dm_logits)
        loss_gi = self.loss_i.g_loss(self.D_i(cond_fake_gi), real_di_logits)

        loss_adv = loss_gx + loss_gm + loss_gi

        loss_prior_x = self._prior_role_reg_loss(self.prior_x, self.regularizer_x, idx_x)
        loss_prior_m = self._prior_role_reg_loss(self.prior_m, self.regularizer_m, idx_m)
        loss_prior_i = self._prior_role_reg_loss(self.prior_i, self.regularizer_i, idx_i)
        loss_prior = loss_prior_x + loss_prior_m + loss_prior_i
        loss_g = loss_adv + loss_prior

        self.opt_g.zero_grad()
        if self.recipe.constraint_geometry_mode != "none":
            constraint_geometry_backward(loss_g, self.opt_g, (loss_adv,))
        else:
            loss_g.backward()
        self.opt_g.step()

        decay = float(self.recipe.ema_decay)
        if decay > 0.0:
            self._update_ema(self.ema_G_x, self.G_x, decay)
            self._update_ema(self.ema_G_m, self.G_m, decay)
            self._update_ema(self.ema_G_i, self.G_i, decay)
            if hasattr(self, "ema_prior_x") and self.prior_x is not None:
                self._update_ema(self.ema_prior_x, self.prior_x, decay)
            if hasattr(self, "ema_prior_m") and self.prior_m is not None:
                self._update_ema(self.ema_prior_m, self.prior_m, decay)
            if hasattr(self, "ema_prior_i") and self.prior_i is not None:
                self._update_ema(self.ema_prior_i, self.prior_i, decay)

        self.completed_steps += 1
        self.last_losses = dict(
            loss_g=float(loss_g.detach()),
            loss_d=float(loss_d.detach()),
            loss_gx=float(loss_gx.detach()),
            loss_gm=float(loss_gm.detach()),
            loss_gi=float(loss_gi.detach()),
            critic_penalty=float((reg_dx + reg_dm + reg_di).detach()),
            loss_prior=float(loss_prior.detach()),
        )
        if not all(math.isfinite(v) for v in self.last_losses.values()):
            raise FloatingPointError("nonfinite actual GAN update")
        return dict(step=self.completed_steps, **self.last_losses)

    @torch.no_grad()
    def observe(self, n=4096, seed=99123):
        models = [self.G_x, self.G_m, self.G_i, self.ema_G_x, self.ema_G_m, self.ema_G_i]
        flags = {m: m.training for model in models for m in model.modules()}
        try:
            for model in models:
                model.eval()

            prior_i_eval = self.prior_i
            prior_eval = self.prior_x
            prior_m_eval = self.prior_m
            def predict_fn(x_q, m_q, rng, draws=16):
                x_q = x_q.to(self.device)
                m_q = m_q.to(self.device)
                out_draws = []
                for _ in range(draws):
                    if prior_i_eval is not None:
                        if isinstance(prior_i_eval, (MoGParticlePrior, ManipulatedPrior)) or hasattr(prior_i_eval, "sigma"):
                            zi, _ = prior_i_eval.sample(len(x_q), generator=rng, noise_generator=rng)
                        else:
                            zi, _ = prior_i_eval.sample(len(x_q), generator=rng)
                    else:
                        zi = torch.randn(len(x_q), 4, generator=rng, device=rng.device).to(device=self.device, dtype=x_q.dtype)
                    cond = torch.cat([x_q * m_q, m_q, zi], dim=1)
                    raw = self.G_i(cond)
                    grafted = m_q * x_q + (1.0 - m_q) * raw
                    out_draws.append(grafted)
                return torch.stack(out_draws, dim=0)

            if self.is_anchored_mcar:
                gen = _rng(seed, device=self.device)
                noise_gen = _rng(seed + 1, device=self.device)
                if isinstance(prior_eval, (MoGParticlePrior, ManipulatedPrior)) or hasattr(prior_eval, "sigma"):
                    z_x, _ = prior_eval.sample(n, generator=gen, noise_generator=noise_gen)
                else:
                    z_x, _ = prior_eval.sample(n, generator=gen)
                if isinstance(prior_m_eval, (MoGParticlePrior, ManipulatedPrior)) or hasattr(prior_m_eval, "sigma"):
                    z_m, _ = prior_m_eval.sample(n, generator=gen, noise_generator=noise_gen)
                else:
                    z_m, _ = prior_m_eval.sample(n, generator=gen)
                x = self.G_x(z_x)
                if isinstance(prior_eval, ManipulatedPrior) and prior_eval.postprocesses_generated:
                    x = prior_eval.postprocess_generated(x)
                m = self.G_m(z_m)
                return self._sampling_witness(score_misgan_mcar_p80_anchored(
                    x, m, lambda x_q, m_q, r: predict_fn(x_q, m_q, r, draws=8), self.problem, eval_seed=seed
                ), posterior_resampling=False)

            return self._sampling_witness(
                score_misgan_neural_imputer(predict_fn, self.problem, eval_seed=seed),
                posterior_resampling=False)
        finally:
            for mod, flag in flags.items():
                mod.training = flag

    def state_dict(self):
        return self._checkpoint_state()

    def load_state_dict(self, state):
        self._restore_checkpoint(state)


def build_case(case_id,*,device="cpu",seed=DEFAULT_SEED,recipe_name=None,max_steps=None):
    if case_id not in CASES:
        for mod in _iter_stretch_providers():
            if any(row["id"] == case_id for row in mod.list_cases()):
                return mod.build_case(case_id, device=device, seed=seed, recipe_name=recipe_name or "atlas", max_steps=max_steps)
        raise ValueError(f"unknown conditional API case {case_id!r}")
    if recipe_name is None:
        recipe_name = CASES[case_id]["default_recipe"]
    if case_id in ("api-misgan-block-recovery", "api-misgan-mcar-p50-recovery"):
        return MisGANRecoveryFixture(deepcopy(CASES[case_id]), seed=seed, recipe_name=recipe_name, max_steps=max_steps, device=device)
    if case_id in ("api-misgan-block-neural-imputer", "api-misgan-mcar-p80-anchored"):
        return MisGANNeuralImputerFixture(deepcopy(CASES[case_id]), seed=seed, recipe_name=recipe_name, max_steps=max_steps, device=device)
    if torch.device(device).type!="cpu": raise ValueError("This frozen caller-owned conditional protocol is CPU-only")
    return ConditionalFixture(deepcopy(CASES[case_id]),_task(case_id.removeprefix("api-")),seed,recipe_name,max_steps)
