"""Bounded learned-feature cells, categorical transport and real-anchor repair.

This is an alternative reaction law, not the kNN/Beta law in birth_death.py.
No data-space distances, evaluator imports or full-table neighbor searches.

This module owns the GANTrainer reaction controller. Its components live in
sibling modules and are re-exported here for import compatibility:

* ``cell_statistics``: constants, finite-resolution feasibility, exact
  conditional count tests and integer allocation;
* ``latent_lineage``: bounded symmetric copy lineage;
* ``latent_geometry``: bounded local DV12 latent geometry;
* ``cell_snapshot``: the fitted cell snapshot and isolation planning;
* ``cell_transport``: the ordinary mass/support/global transport planner.
"""
import math
import time

import torch

from .birth_phase import (apply_anchor_births, learned_latent_features, novel_birth_diagnostics,
                          plan_real_anchor_births, plan_residual_isolation)
from .birth_phase import plan_residual_global_copies  # noqa: F401  (re-export)
from .cell_snapshot import FeatureCellSnapshot, paired_average_geometry
from .cell_statistics import (CELL_RESOLUTION_POLICY, JITTER_CAP, JITTER_STD, LLOYD_PASSES,
                              PARENT_RANK_CAP, PARENT_RESERVOIR, POWER_PASSES, Q, REAL_ANCHORS,
                              SCALE_PRIOR, _fit_cell_count, conditional_count_pvalues,
                              population_policy, q_order_statistic)
from .cell_statistics import (_features_ok, _group_integer_allocate,  # noqa: F401  (re-export)
                              _integer_allocate)
from .cell_transport import (GlobalTransportPolicy, MassTransportPolicy, SupportTransportPolicy,
                             TransportPolicy, plan_ordinary_transport)
from .feature_reference import FrozenFeatureReferenceBirthDeath, ParticleBirthDeath
from .latent_geometry import BoundedLatentGeometry, bounded_jitter
from .latent_lineage import LatentLineage
from .mean_transport import (FIT_CHUNK, INSIDE_POLICY, MEAN_POLICY, PLANNING_POLICY,
                             PROJECTION_POLICY, check_mean_diagnostics, initial_mean_diagnostics,
                             prepare_mean_witness, run_mean_phase)
from .row_transport import prepare_copy_history

__all__ = [
    "BoundedLatentGeometry", "CELL_RESOLUTION_POLICY", "FeatureCellBirthDeath",
    "FeatureCellSnapshot", "FrozenFeatureReferenceBirthDeath", "GlobalTransportPolicy",
    "JITTER_CAP", "JITTER_STD", "LLOYD_PASSES", "LatentLineage", "MassTransportPolicy",
    "PARENT_RANK_CAP", "PARENT_RESERVOIR", "POWER_PASSES", "ParticleBirthDeath", "Q",
    "REAL_ANCHORS", "SCALE_PRIOR", "SmallPopulationReferenceBirthDeath", "SupportTransportPolicy",
    "TransportPolicy", "bounded_jitter", "conditional_count_pvalues",
    "make_feature_cell_birth_death", "paired_average_geometry", "plan_ordinary_transport",
    "population_policy",
]


class SmallPopulationReferenceBirthDeath(FrozenFeatureReferenceBirthDeath):
    """Unchanged reference reaction/sampling law with explicit route metadata."""
    def __init__(self, trainer, seed):
        super().__init__(trainer, seed)
        self.population_policy = population_policy(self.N)
        if self.population_policy["finite_resolution_feasible"]:
            raise ValueError("reference fallback requires impossible feature-cell resolution")

    def diagnostics(self):
        result = super().diagnostics()
        result.update(backend="knn", population_policy=dict(self.population_policy))
        return result

    def state_dict(self):
        result = super().state_dict()
        result["population_policy"] = dict(self.population_policy)
        return result

    def check_state(self, state):
        if (not isinstance(state, dict)
                or state.get("population_policy") != self.population_policy):
            raise ValueError("incompatible small-population backend/sampler metadata")
        super().check_state({key: value for key, value in state.items()
                             if key != "population_policy"})


def make_feature_cell_birth_death(trainer, seed):
    if population_policy(len(trainer.prior.z))["finite_resolution_feasible"]:
        return FeatureCellBirthDeath(trainer, seed)
    return SmallPopulationReferenceBirthDeath(trainer, seed)


_MEAN_COUNTERS = ("mean_evals", "mean_witness_fires", "mean_moves", "mean_forward_rows",
                  "mean_preview_rows")
_ACTION_ROWS = ("ordinary_mean_children", "ordinary_mean_parents", "ordinary_action_children",
                "ordinary_copy_parent_rows", "ordinary_action_kinds")
_PHASE_MOVES = ("ordinary_mass_moves", "ordinary_support_moves", "ordinary_global_moves",
                "ordinary_mean_moves", "ordinary_copy_moves", "ordinary_novel_birth_moves",
                "ordinary_moves", "iso_moves", "moves")
_ACTION_KIND_FIELDS = ("ordinary_mass_moves", "ordinary_support_moves", "ordinary_global_moves",
                       "ordinary_novel_birth_moves", "ordinary_mean_moves")
_PAIRED_COUNTS = ("finite_rows", "same_group_rows", "ema_eligible_rows", "coherent_rows")


class FeatureCellBirthDeath(FrozenFeatureReferenceBirthDeath):
    """Actual GANTrainer backend; keeps original FIFO and row-copy mechanics."""
    BACKEND_SCHEMA = 10

    def __init__(self, trainer, seed):
        super().__init__(trainer, seed)
        self.population_policy = population_policy(self.N)
        if not self.population_policy["finite_resolution_feasible"]:
            raise ValueError("use make_feature_cell_birth_death for an infeasible small population")
        recipe = trainer.recipe
        rank = recipe.birth_death_metric_rank
        lineage_degree = min(rank, PARENT_RESERVOIR, self.N - 1)
        self.settings = dict(
            cells=recipe.birth_death_cells, rank=rank, chunk=recipe.birth_death_chunk,
            parent_policy=recipe.birth_death_parent_policy, lloyd=LLOYD_PASSES,
            power=POWER_PASSES, parent_reservoir=PARENT_RESERVOIR, real_anchors=1,
            parent_rank=None, latent_kernel="bounded_local_dv12_lineage",
            latent_neighbors=PARENT_RESERVOIR, latent_rank=rank, lineage_degree=lineage_degree,
            lineage_policy="symmetric_copy_links_newest_first_invalidate_overwrites",
            latent_candidate_bound=PARENT_RESERVOIR + lineage_degree,
            mass_policy="joint_mass_local_global_mean_common_3K_plus_3_unique_parents_v1",
            isolation_parent_pool="supported_same_cell_without_replacement",
            count_partition="even_fit_score_order_statistic_2K",
            count_family="original_K_plus_support_2K_plus_global_2_plus_mean_1_common_Q_over_3K_plus_3",
            copy_noise_policy="shared_noise_separate_live_ema_current_geometry_v1",
            novel_birth_policy="paired_even_real_anchor_shared_3K_plus_3_v1",
            novel_birth_cells=4, novel_birth_linearizations=4,
            novel_birth_trust="per_model_prior_rms_coordinate_spread",
            paired_average_policy="ema_support_inside_current_real_group_Q_v1",
            paired_average_expiry="strictly_less_than_one_real_fifo_turnover_v1",
            resolution_policy=CELL_RESOLUTION_POLICY, mean_policy=MEAN_POLICY,
            mean_planning_policy=PLANNING_POLICY, mean_inside_policy=INSIDE_POLICY,
            mean_rank_bound=8, mean_projection_policy=PROJECTION_POLICY,
            mean_frame_fit_chunk=FIT_CHUNK, mean_pool=64)
        self.lineage = LatentLineage(self.N, self.settings["lineage_degree"], trainer.device)
        self.latent_geometry = BoundedLatentGeometry(rank=rank, neighbors=PARENT_RESERVOIR,
                                                     chunk=recipe.birth_death_chunk,
                                                     lineage=self.lineage)
        self._copy_parent_rows = None
        self._sampling_prior, self._sampling_controller = trainer.prior, trainer.controller
        self.snapshot = None
        self.snapshot_serial = 0
        self.paired_average = dict(
            schema=1, policy=self.settings["paired_average_policy"], snapshot=0, step=0,
            rows=self.N, required=self.N-math.floor(Q*self.N), cells=0, rank=0, groups=0,
            calibration_rows=0, chart_valid=False, duplicate_ok=False, finite_rows=0,
            same_group_rows=0, ema_eligible_rows=0, coherent_rows=0, eligible=False)
        self.counters.update({key: 0 for key in (
            "cell_evals", "cell_discoveries", "ordinary_moves", "feature_rebuilds",
            "feature_distance_cells", "projection_products", "count_test_terms",
            "feature_forward_rows", "invalidated_cells", "novel_birth_moves",
            "novel_birth_attempts", *_MEAN_COUNTERS)})
        self.last.update(mean_transport=initial_mean_diagnostics(), ordinary_mean_moves=0,
                         ordinary_mean_children=[], ordinary_mean_parents=[],
                         ordinary_action_children=[], ordinary_copy_parent_rows=[],
                         ordinary_action_kinds=[])

    def paired_average_eligible(self, step):
        """Empirical bounded-stale view, valid for less than one FIFO turnover.

        G/D can move between reactions. This is an anti-blur lease, not an
        every-update support or distribution certificate. Only scalar semantic
        state is needed after load; the ephemeral chart may remain absent.
        """
        stamp = self.paired_average
        return bool(stamp["eligible"] and stamp["snapshot"] == self.snapshot_serial
                    and type(step) is int and step >= stamp["step"]
                    and self.fill == self.N and 0 <= self.rows_since_eval < self.N)

    @torch.no_grad()
    def _record_paired_average(self, trainer, snapshot):
        """One extra chunked EMA forward after every reaction action."""
        roots = (trainer.ema_G, trainer.D)
        modes = [(m, m.training) for root in roots for m in root.modules()]
        # Supported custom eval forwards can draw global randomness or update
        # registered buffers. The measurement is observational: preserve both
        # without consuming any dedicated trainer/reaction stream.
        buffers = [(m, dict(m._buffers),
                    {name: None if value is None else value.detach().clone()
                     for name, value in m._buffers.items()},
                    set(m._non_persistent_buffers_set))
                   for root in roots for m in root.modules()]
        streams = [self.stream, *[getattr(trainer, name) for name in getattr(trainer, "_STREAMS", ())]]
        stream_states = [(stream, stream.get_state()) for stream in streams]
        gradients = [(p, p.grad, None if p.grad is None else p.grad.detach().clone())
                     for root in roots for p in root.parameters()]
        devices = [trainer.device.index] if trainer.device.type == "cuda" else []
        features, finite_output = [], []
        try:
            with torch.random.fork_rng(devices=devices):
                trainer.ema_G.eval()
                trainer.D.eval()
                for block in trainer.ema_prior.z.detach().split(self.settings["chunk"]):
                    raw = trainer.ema_G(block)
                    if tuple(raw.shape[1:]) != self.sample_shape:
                        raise ValueError("EMA generator output shape does not match the real FIFO")
                    finite_output.append(torch.isfinite(raw).flatten(1).all(1))
                    features.append(self._features(trainer, raw, chunk=self.settings["chunk"]))
        finally:
            for stream, before in stream_states:
                stream.set_state(before)
            for parameter, original, before in gradients:
                parameter.grad = original
                if original is not None:
                    original.copy_(before)
            for module, original, before, nonpersistent in buffers:
                module._buffers.clear()
                module._buffers.update(original)
                module._non_persistent_buffers_set.clear()
                module._non_persistent_buffers_set.update(nonpersistent)
                for name, value in original.items():
                    if value is not None:
                        value.copy_(before[name])
            for module, mode in modes:
                module.training = mode
        finite = (torch.isfinite(trainer.prior.z.detach()).all(1)
                  & torch.isfinite(trainer.ema_prior.z.detach()).all(1)
                  & torch.cat(finite_output))
        self.paired_average = paired_average_geometry(
            snapshot, torch.cat(features), step=trainer.completed_steps+1,
            snapshot_serial=self.snapshot_serial, finite_coordinates=finite)

    @torch.no_grad()
    def _jitter(self, trainer, latent, prior, noise):
        return self.latent_geometry.displacement(latent, prior, trainer.controller.latent_bandwidth,
                                                 noise, rows=self._copy_parent_rows)

    @torch.no_grad()
    def _move(self, trainer, child, parent):
        history = prepare_copy_history(trainer.prior.z, trainer.opt_g, child, parent)
        self._copy_parent_rows = parent
        try:
            prior, ema = trainer.prior, trainer.ema_prior
            zp, ema_zp = prior.z.detach()[parent], ema.z.detach()[parent]
            noise = torch.randn(zp.shape, device=zp.device, dtype=zp.dtype, generator=self.stream)
            delta = self._jitter(trainer, zp, prior, noise)
            # A shared noise draw preserves coupling and RNG accounting. Each
            # table applies that draw within its own current local geometry.
            ema_delta = self.latent_geometry.displacement(
                ema_zp, ema, trainer.controller.latent_bandwidth, noise, rows=parent)
            prior.z[child] = zp+delta
            ema.z[child] = ema_zp+ema_delta
            for value, source in history:
                value[child] = source
        finally:
            self._copy_parent_rows = None
        self.lineage.register_copies(child, parent)

    def perturb_latent(self, latent, stream, controller=None, record=False, *, prior=None,
                       rows=None):
        controller = self._sampling_controller if controller is None else controller
        prior = self._sampling_prior if prior is None else prior
        noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=stream)
        delta = self.latent_geometry.displacement(
            latent, prior, controller.latent_bandwidth, noise,
            record=controller.latent_applications if record else None, rows=rows)
        return latent+delta

    @torch.no_grad()
    def _capture_generated(self, trainer, latent, *, sigma=0., jitter=False, rows=None):
        features = []
        chunk = self.settings["chunk"]
        for start in range(0, len(latent), chunk):
            block = latent[start:start+chunk]
            if jitter:
                noise = torch.randn(block.shape, device=block.device, dtype=block.dtype,
                                    generator=self.stream)
                block = block+self.latent_geometry.displacement(
                    block, trainer.prior, trainer.controller.latent_bandwidth, noise,
                    rows=None if rows is None else rows[start:start+len(block)])
            raw = trainer.G(block)
            if tuple(raw.shape[1:]) != self.sample_shape:
                raise ValueError("generator output shape does not match the real FIFO")
            if sigma:
                raw = raw+float(sigma)*torch.randn(raw.shape, device=raw.device, dtype=raw.dtype,
                                                   generator=self.stream)
            features.append(self._features(trainer, raw, chunk=chunk))
            self.counters["feature_forward_rows"] += len(block)
        return torch.cat(features)

    def _reset_rows(self, rows):
        """Rows whose generated output changed start a new evidence lineage."""
        self.S[rows] = 0.
        self.W[rows] = 0.
        self.n[rows] = 0

    @torch.no_grad()
    def maybe_apply(self, trainer, sigma_out):
        self.moved_rows = None
        if self.N <= self.settings["cells"]:
            return None
        if not self.ready():
            return None
        sigma = 0. if sigma_out is None else float(sigma_out)
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("output sigma must be finite and nonnegative")
        # The success path keeps its original order (the body reads the new
        # serial). A failed apply restores the accumulated evidence, the evaluation
        # clock, every counter and the serial together (so the checkpoint invariant
        # mean_evals == snapshot_serial still holds), so the evidence is not lost.
        # Rows the failed body already moved (any table row that changed) start a
        # new lineage with empty evidence. Stream draws, the moved rows themselves
        # and the snapshot are not rolled back, and the restored counters do not
        # count those partial moves.
        evidence = (self.S, self.W, self.n, self.pending)
        saved = ([t.clone() for t in evidence], self.rows_since_eval, dict(self.counters),
                 self.snapshot_serial)
        table_before = trainer.prior.z.detach().clone()
        self.rows_since_eval = 0
        self.counters["evals"] += 1
        self.counters["cell_evals"] += 1
        self.snapshot_serial += 1
        for tensor in evidence:
            tensor.zero_()
        modes = [(module, module.training) for root in (trainer.G, trainer.D)
                 for module in root.modules()]
        begin = time.perf_counter()
        try:
            return self._apply_evaluation(trainer, sigma, begin)
        except BaseException:
            values, self.rows_since_eval, counters, self.snapshot_serial = saved
            for target, value in zip(evidence, values):
                target.copy_(value)
            self.counters.clear()
            self.counters.update(counters)
            moved = torch.ne(trainer.prior.z.detach(), table_before).flatten(1).any(1)
            if self.moved_rows is not None:
                moved[self.moved_rows] = True
            if bool(moved.any()):  # rows already transported start a new lineage
                self._reset_rows(moved)
            raise
        finally:
            for module, mode in modes:
                module.training = mode

    @torch.no_grad()
    def _apply_evaluation(self, trainer, sigma, begin):
        trainer.G.eval()
        trainer.D.eval()
        chunk = self.settings["chunk"]
        z = trainer.prior.z.detach()
        q = self._capture_generated(trainer, z)
        R = self._features(trainer, self.reservoir, chunk=chunk)
        self.counters["feature_forward_rows"] += self.N
        snapshot = FeatureCellSnapshot.fit(R, generator=self.stream, cells=self.settings["cells"],
                                           rank=self.settings["rank"], chunk=chunk)
        self.snapshot = snapshot
        # Freeze the scalar witness before any odd-count-driven action.
        partial_recovery = getattr(trainer, "mean_partial_recovery", False)
        mean_fixed, mean_stamp = prepare_mean_witness(self, trainer, snapshot, R,
                                                      allow_partial_groups=partial_recovery)
        pick = torch.randint(self.N, (self.N,), device=z.device, generator=self.stream)
        F = self._capture_generated(trainer, z[pick], sigma=sigma, jitter=True, rows=pick)
        flags, pvalues, _ = snapshot.support(q)
        if not self.isolation:
            flags.zero_()
        snapshot.cache_queries(q)
        comparison = snapshot.cell_comparison(F)
        current_features = learned_latent_features(self, trainer, trainer.G)
        average_features = learned_latent_features(self, trainer, trainer.ema_G)

        def birth_planner(copy_children, copy_parents, supported, budget):
            return plan_real_anchor_births(
                snapshot, q, flags, pvalues, comparison, z, current_features,
                ema_latents=trainer.ema_prior.z.detach(), ema_feature_of_latent=average_features,
                previous_children=copy_children, previous_copy_parents=copy_parents,
                supported_counts=supported, max_moves=budget)

        child, parent, ordinary = snapshot.ordinary_transport(
            q, flags, comparison, generator=self.stream, pvalues=pvalues,
            birth_planner=birth_planner)
        birth_plan = ordinary["novel_birth_plan"]
        newborn = birth_plan["children"]
        iso_child, iso_parent, anchored = plan_residual_isolation(
            snapshot, q, flags, pvalues, birth_plan, generator=self.stream, copy_children=child,
            copy_parents=parent, supported_counts=ordinary["planned_supported_counts"])
        c = self.counters
        c["feature_rebuilds"] += 1
        c["cell_discoveries"] += ordinary["discoveries"]
        c["discoveries"] += ordinary["discoveries"]
        c["realised_deaths"] += len(child)+len(newborn)
        c["realised_births"] += len(parent)+len(newborn)
        if self.isolation:
            c["iso_evals"] += 1
            c["iso_flagged"] += int(flags.sum())
            guard = bool(anchored["guard_passed"])
            c["iso_acted" if guard else "iso_skipped"] += bool(flags.any())
            c["iso_dup_skips"] += snapshot.duplicate_fraction > Q
            self.iso_log = (self.iso_log+[[trainer.completed_steps+1, int(flags.sum()), int(guard)]])[-30:]
        last = dict(
            step=trainer.completed_steps + 1, backend="feature_cells", snapshot=self.snapshot_serial,
            cells=snapshot.cells, metric_rank=snapshot.rank, k=None, d_R=None, d_F=None,
            eligible=int((pvalues > Q).sum()), discoveries=ordinary["discoveries"],
            ordinary_discoveries=ordinary["discoveries"],
            ordinary_excess_cells=ordinary["excess_cells"],
            ordinary_deficit_cells=ordinary["deficit_cells"], ordinary_budget=ordinary["budget"],
            ordinary_inaccessible_birth_quota=ordinary["inaccessible_birth_quota"],
            ordinary_moves=0 if self.dry_run else ordinary["moves"], iso_flagged=int(flags.sum()),
            iso_moves=0 if self.dry_run else len(iso_child), moves=0,
            anchored_inaccessible_deficit=anchored["inaccessible_deficit"],
            calibration_rows=snapshot.calibration_rows,
            pvalue_floor=1 / (snapshot.calibration_rows + 1),
            minimum_bh_flag_fraction=1 / (Q * (snapshot.calibration_rows + 1)),
            duplicate_fraction=snapshot.duplicate_fraction,
            statistic="conditional_categorical_transport", density_ratio=None,
            statistical_limit="conditional iid cell test; no repeated adaptive guarantee",
            mass_topology=dict(snapshot.mass_topology),
            ordinary_within_group_moves=ordinary["within_group_moves"],
            ordinary_between_group_moves=ordinary["between_group_moves"],
            unique_reaction_parents=len(torch.unique(torch.cat((parent, iso_parent)))),
            isolation_unfilled=anchored.get("unfilled_repairs", int(flags.sum())),
            count_partition=dict(snapshot.count_partition),
            count_boundary=float(snapshot.count_boundary),
            count_categories=comparison["categories"],
            count_multiplicity=comparison["multiplicity"], count_cutoff=comparison["cutoff"],
            ordinary_mass_moves=ordinary["mass_moves"],
            ordinary_support_moves=ordinary["support_moves"],
            ordinary_global_moves=ordinary["global_moves"],
            ordinary_death_policy=ordinary["death_policy"])
        novel_diagnostics = novel_birth_diagnostics(birth_plan)
        last.update(ordinary_copy_moves=0 if self.dry_run else len(child),
                    ordinary_novel_birth_moves=0 if self.dry_run else len(newborn),
                    novel_birth_attempts=birth_plan["attempted_cells"],
                    novel_birth_target_cells=novel_diagnostics["new_latent_target_cells"],
                    novel_birth_children=novel_diagnostics["child_rows"],
                    novel_birth_seed_rows=novel_diagnostics["source_seed_rows"],
                    novel_birth=novel_diagnostics,
                    novel_birth_work_bound=dict(birth_plan["work_bound"]))
        if not snapshot.valid_metric:
            last["skip"] = "no varying learned features"
        if not self.dry_run:
            if len(child):
                self._move(trainer, child, parent)
            if len(iso_child):
                self._move(trainer, iso_child, iso_parent)
            apply_anchor_births(trainer, self, birth_plan)
            moved = torch.cat((child, iso_child, newborn))
            if len(moved):
                self.moved_rows = moved
                refreshed = self._capture_generated(trainer, trainer.prior.z.detach()[moved])
                invalidated, affected = snapshot.refresh_rows(moved, refreshed)
                self._reset_rows(invalidated)
                c["stale_resets"] += int(invalidated.sum())
                c["invalidated_cells"] += int(affected.sum())
                last.update(stale_resets=int(invalidated.sum()), invalidated_cells=int(affected.sum()))
            c["ordinary_moves"] += len(child)+len(newborn)
            c["moves"] += len(child)+len(newborn)
            c["novel_birth_moves"] += len(newborn)
            c["novel_birth_attempts"] += birth_plan["attempted_cells"]
            c["matched"] += len(child)  # Copy matches; novel births have their own counter.
            if self.isolation:
                c["iso_moves"] += len(iso_child)
            last["moves"] = len(child)+len(iso_child)+len(newborn)
        # Fourth ordinary copy phase sees actual post-prefix tables/counts;
        # the pre-prefix witness and ranking directions stay fixed.
        reserved = torch.cat((child, parent, iso_child, iso_parent, newborn,
                              birth_plan["source_seed_rows"]))
        mean_phase = run_mean_phase(self, trainer, snapshot, mean_fixed, mean_stamp,
                                    earlier_ordinary=len(child)+len(newborn), reserved_rows=reserved,
                                    allow_partial_groups=partial_recovery)
        mean_child, mean_parent = mean_phase["children"], mean_phase["parents"]
        if len(mean_child):
            invalidated, affected = mean_phase["invalidated"], mean_phase["affected"]
            self._reset_rows(invalidated)
            c["stale_resets"] += int(invalidated.sum())
            c["invalidated_cells"] += int(affected.sum())
            last["stale_resets"] = last.get("stale_resets", 0)+int(invalidated.sum())
            last["invalidated_cells"] = last.get("invalidated_cells", 0)+int(affected.sum())
            self.moved_rows = torch.cat((child, iso_child, newborn, mean_child))
            c["mean_moves"] += len(mean_child)
            c["realised_deaths"] += len(mean_child)
            c["realised_births"] += len(mean_child)
            c["ordinary_moves"] += len(mean_child)
            c["moves"] += len(mean_child)
            c["matched"] += len(mean_child)
        mean_count = len(mean_child)
        if self.dry_run:
            action_children, copy_parents, action_kinds = [], [], []
        else:
            action_children = torch.cat((ordinary["action_children"], mean_child)).cpu().tolist()
            copy_parents = torch.cat((parent, mean_parent)).cpu().tolist()
            action_kinds = torch.cat((ordinary["action_kinds"],
                                      torch.full_like(mean_child, 4))).cpu().tolist()
        last.update(mean_transport=dict(mean_phase["diagnostics"]), ordinary_mean_moves=mean_count,
                    ordinary_mean_children=mean_child.cpu().tolist(),
                    ordinary_mean_parents=mean_parent.cpu().tolist(),
                    ordinary_copy_moves=last["ordinary_copy_moves"]+mean_count,
                    ordinary_moves=last["ordinary_moves"]+mean_count,
                    moves=last["moves"]+mean_count,
                    ordinary_within_group_moves=last["ordinary_within_group_moves"]+mean_count,
                    unique_reaction_parents=len(torch.unique(torch.cat((parent, iso_parent,
                                                                        mean_parent)))),
                    ordinary_action_children=action_children,
                    ordinary_copy_parent_rows=copy_parents,
                    ordinary_action_kinds=action_kinds)
        if self.dry_run:
            for key in ("ordinary_mass_moves", "ordinary_support_moves", "ordinary_global_moves"):
                last[key] = 0
        # All old/mean copies and new latents have been applied and cached.
        # Keep the original fresh final EMA query: no feature-cache reuse.
        self._record_paired_average(trainer, snapshot)
        last["paired_average"] = dict(self.paired_average)
        last["paired_average_forward_rows"] = self.N
        c["feature_distance_cells"] += snapshot.work["distance_cells"]
        c["projection_products"] += snapshot.work["projection_products"]
        c["count_test_terms"] += snapshot.work["count_test_terms"]
        last.update(work=dict(snapshot.work), eval_seconds=time.perf_counter()-begin)
        self.last = last
        return last

    def diagnostics(self):
        result = super().diagnostics()
        result.update(backend="feature_cells", settings=dict(self.settings),
                      snapshot_serial=self.snapshot_serial,
                      snapshot_work=None if self.snapshot is None else dict(self.snapshot.work),
                      population_policy=dict(self.population_policy),
                      paired_average=dict(self.paired_average),
                      paired_average_age_real_rows=self.rows_since_eval,
                      lineage_edges=int((self.lineage.neighbors >= 0).sum())//2)
        return result

    def state_dict(self):
        result = super().state_dict()
        # Own the new mutable axes metadata at the checkpoint boundary.
        mean = dict(result["last"]["mean_transport"])
        mean["selected_axes"] = list(mean["selected_axes"])
        result["last"]["mean_transport"] = mean
        result.update(backend="feature_cells", backend_schema=self.BACKEND_SCHEMA,
                      settings=dict(self.settings), sample_shape=self.sample_shape,
                      snapshot_serial=self.snapshot_serial,
                      paired_average=dict(self.paired_average),
                      population_policy=dict(self.population_policy),
                      lineage_neighbors=self.lineage.neighbors.clone())
        return result

    # Checkpoint validation ----------------------------------------------------

    def _base_state_keys(self):
        return set(self._TENSORS) | {"fill", "cursor", "rows_since_eval", "counters", "last", "stream"}

    def check_state(self, state):
        extra = {"backend", "backend_schema", "settings", "sample_shape", "snapshot_serial",
                 "population_policy", "lineage_neighbors", "paired_average"}
        base = self._base_state_keys()
        if (not isinstance(state, dict) or set(state) != base | extra
                or state.get("backend") != "feature_cells"
                or type(state.get("backend_schema")) is not int
                or state.get("backend_schema") != self.BACKEND_SCHEMA
                or not isinstance(state.get("settings"), dict) or state.get("settings") != self.settings
                or any(type(state["settings"][key]) is not type(value)
                       for key, value in self.settings.items())
                or state.get("population_policy") != self.population_policy):
            raise ValueError("invalid or incompatible feature-cell checkpoint backend/settings")
        super().check_state({key: state[key] for key in base})
        self.lineage.validate(state["lineage_neighbors"])
        for key in ("fill", "cursor", "rows_since_eval", "snapshot_serial"):
            value = state[key]
            if type(value) is not int or value < 0:
                raise ValueError("invalid feature-cell checkpoint counters")
        if state["fill"] > self.N or state["cursor"] >= self.N:
            raise ValueError("invalid feature-cell FIFO state")
        if (not isinstance(state["counters"], dict) or set(state["counters"]) != set(self.counters)
                or any(type(value) is not int or value < 0 for value in state["counters"].values())
                or not isinstance(state["last"], dict)):
            raise ValueError("invalid feature-cell diagnostics state")
        self._check_paired_average_state(state)
        check_mean_diagnostics(state["last"].get("mean_transport"), last=state["last"],
                               paired=state["paired_average"], n=self.N,
                               snapshot_serial=state["snapshot_serial"], dry_run=self.dry_run,
                               sample_shape=state["sample_shape"])
        counts = state["counters"]
        if (counts["mean_evals"] != state["snapshot_serial"]
                or not 0 <= counts["mean_witness_fires"] <= counts["mean_evals"]
                or counts["mean_moves"] < state["last"]["mean_transport"]["moves"]
                or (state["snapshot_serial"] == 0 and any(counts[k] for k in _MEAN_COUNTERS))):
            raise ValueError("mean cumulative counters disagree with their reaction ledger")
        self._check_mean_actions(state)
        for key in ("S", "W", "n", "anchor", "radius", "pending"):
            value, expected = state[key], getattr(self, key)
            if (value is None or value.shape != expected.shape or value.dtype != expected.dtype
                    or (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
                raise ValueError("invalid feature-cell row state")
        shape = state["sample_shape"]
        if shape is not None and (not isinstance(shape, (tuple, list)) or not shape
                                  or any(type(value) is not int or value <= 0 for value in shape)):
            raise ValueError("invalid feature-cell sample shape")
        reservoir = state["reservoir"]
        if (reservoir is None) != (shape is None):
            raise ValueError("inconsistent feature-cell reservoir/sample shape")
        if reservoir is None and (state["fill"] or state["cursor"] or state["rows_since_eval"]):
            raise ValueError("feature-cell FIFO counters require a reservoir")
        if reservoir is not None and (reservoir.ndim != 2 or reservoir.shape[1] != math.prod(shape)
                                      or reservoir.dtype != self.radius.dtype
                                      or not bool(torch.isfinite(reservoir).all())):
            raise ValueError("invalid feature-cell reservoir")

    def _check_mean_actions(self, state):
        """Actual fourth-phase copy bookkeeping and bounded row identities."""
        last = state["last"]
        if any(not isinstance(last.get(k), list) for k in _ACTION_ROWS):
            raise ValueError("missing mean/ordinary action row diagnostics")
        for key in _ACTION_ROWS[:-1]:
            rows = last[key]
            if (any(type(row) is not int or not 0 <= row < self.N for row in rows)
                    or len(rows) != len(set(rows))):
                raise ValueError("invalid mean/ordinary action row IDs")
        kinds = last["ordinary_action_kinds"]
        if any(type(kind) is not int or kind not in (0, 1, 2, 3, 4) for kind in kinds):
            raise ValueError("invalid ordinary action kinds")
        child, parent = last["ordinary_mean_children"], last["ordinary_mean_parents"]
        if len(child) != len(parent) or set(child) & set(parent):
            raise ValueError("mean parents and children must be distinct paired incarnations")
        if state["snapshot_serial"] == 0:
            if (any(last[k] for k in _ACTION_ROWS)
                    or type(last.get("ordinary_mean_moves")) is not int
                    or last["ordinary_mean_moves"] != 0):
                raise ValueError("initial mean action metadata must be empty")
            return
        if any(type(last.get(k)) is not int or last[k] < 0 for k in _PHASE_MOVES):
            raise ValueError("invalid fourth-phase move counts")
        action_children = last["ordinary_action_children"]
        copy_parents = last["ordinary_copy_parent_rows"]
        if (last["ordinary_mean_moves"] != len(child)
                or last["ordinary_copy_moves"] != sum(last[k] for k in _PHASE_MOVES[:4])
                or last["ordinary_moves"] != last["ordinary_copy_moves"]+last["ordinary_novel_birth_moves"]
                or last["ordinary_moves"] > math.floor(Q*self.N)
                or last["moves"] != last["ordinary_moves"]+last["iso_moves"]
                or len(action_children) != last["ordinary_moves"]
                or len(kinds) != last["ordinary_moves"]
                or len(copy_parents) != last["ordinary_copy_moves"]
                or kinds.count(4) != len(child)
                or [r for r, k in zip(action_children, kinds) if k == 4] != child
                or not set(parent) <= set(copy_parents)
                or set(action_children) & set(copy_parents)
                or (parent and copy_parents[-len(parent):] != parent)
                or any(kinds.count(kind) != last[field]
                       for kind, field in enumerate(_ACTION_KIND_FIELDS))):
            raise ValueError("fourth-phase actions disagree with the shared ledger")
        novel = set(last.get("novel_birth_children", ())) | set(last.get("novel_birth_seed_rows", ()))
        if (set(child) | set(parent)) & novel:
            raise ValueError("mean actions cannot overwrite or reuse novel sources/children")

    def _check_paired_average_state(self, state):
        stamp = state["paired_average"]
        if not isinstance(stamp, dict) or set(stamp) != set(self.paired_average):
            raise ValueError("invalid paired-average semantic state")
        if (type(stamp["schema"]) is not int or stamp["schema"] != 1
                or stamp["policy"] != self.settings["paired_average_policy"]):
            raise ValueError("incompatible paired-average geometry law")
        booleans = ("chart_valid", "duplicate_ok", "eligible")
        integers = ("snapshot", "step", "rows", "required", "cells", "rank", "groups",
                    "calibration_rows", *_PAIRED_COUNTS)
        if (any(type(stamp[k]) is not bool for k in booleans)
                or any(type(stamp[k]) is not int or stamp[k] < 0 for k in integers)):
            raise ValueError("invalid paired-average typed fields")
        n = self.N
        if (stamp["snapshot"] != state["snapshot_serial"] or stamp["rows"] != n
                or stamp["required"] != n-math.floor(Q*n)
                or any(stamp[k] > n for k in _PAIRED_COUNTS[:3])
                or stamp["coherent_rows"] > min(stamp[k] for k in _PAIRED_COUNTS[:3])
                or (stamp["finite_rows"] == n
                    and stamp["coherent_rows"] < max(0, stamp["same_group_rows"]
                                                     + stamp["ema_eligible_rows"]-n))
                or (stamp["finite_rows"] < n
                    and any(stamp[k] for k in ("groups", *_PAIRED_COUNTS[1:])))
                or stamp["groups"] > stamp["cells"]
                or stamp["rank"] > min(self.settings["rank"], (n+1)//2-1)
                or stamp["cells"] > min(self.settings["cells"], (n+1)//2)):
            raise ValueError("inconsistent paired-average counts/chart")
        expected = (stamp["chart_valid"] and stamp["duplicate_ok"] and stamp["finite_rows"] == n
                    and stamp["coherent_rows"] >= stamp["required"])
        if stamp["eligible"] != expected:
            raise ValueError("paired-average eligibility disagrees with its counts")
        if stamp["chart_valid"] != (stamp["rank"] > 0):
            raise ValueError("paired-average chart validity disagrees with its rank")
        if stamp["snapshot"] == 0:
            if (stamp["step"] != 0 or stamp["cells"] or stamp["rank"] or stamp["groups"]
                    or stamp["calibration_rows"] or any(stamp[k] for k in booleans)
                    or any(stamp[k] for k in _PAIRED_COUNTS)):
                raise ValueError("invalid initial paired-average state")
        elif (stamp["step"] <= 0 or stamp["cells"] <= 0 or stamp["calibration_rows"] != n//2
                or (stamp["chart_valid"] and stamp["rank"] <= 0)
                or (stamp["finite_rows"] == n and stamp["groups"] <= 0)
                or state["last"].get("step") != stamp["step"]
                or state["last"].get("snapshot") != stamp["snapshot"]
                or state["last"].get("paired_average") != stamp):
            raise ValueError("inconsistent paired-average reaction stamp")
        if stamp["snapshot"]:
            self._check_reaction_chart(state["last"], stamp)

    def _check_reaction_chart(self, last, stamp):
        """The saved reaction's count/resolution diagnostics match its chart."""
        n = self.N
        fitted_rows = (n+1)//2
        actual_cells = _fit_cell_count(self.settings["cells"], fitted_rows, stamp["rank"])
        partition = dict(rule="even_fit_score_order_statistic", q=Q, fitted_rows=fitted_rows,
                         ordinal=q_order_statistic(fitted_rows), ties="inside",
                         categories=2*actual_cells)
        saved_partition = last.get("count_partition")
        if (stamp["cells"] != actual_cells
                or type(last.get("count_categories")) is not int
                or last["count_categories"] != 2*actual_cells
                or type(last.get("count_multiplicity")) is not int
                or last["count_multiplicity"] != 3*actual_cells+3
                or type(last.get("count_cutoff")) is not float
                or last["count_cutoff"] != Q/(3*actual_cells+3)
                or not isinstance(saved_partition, dict)
                or saved_partition.keys() != partition.keys()
                or any(type(saved_partition[key]) is not int
                       for key in ("fitted_rows", "ordinal", "categories"))
                or type(saved_partition["q"]) is not float
                or any(type(saved_partition[key]) is not str for key in ("rule", "ties"))
                or saved_partition != partition):
            raise ValueError("resolution/count diagnostics disagree with the actual even-fit chart")
        if any(type(last.get(key)) is not int or last[key] != stamp[field]
               for key, field in (("cells", "cells"), ("metric_rank", "rank"),
                                  ("calibration_rows", "calibration_rows"))):
            raise ValueError("paired-average stamp disagrees with its reaction chart")
        duplicate = last.get("duplicate_fraction")
        if (type(duplicate) not in (float, int) or not math.isfinite(duplicate)
                or not 0 <= duplicate <= 1 or stamp["duplicate_ok"] != (duplicate <= Q)):
            raise ValueError("paired-average duplicate guard disagrees with its reaction")
        topology = last.get("mass_topology")
        if (stamp["finite_rows"] == n
                and (not isinstance(topology, dict) or type(topology.get("groups")) is not int
                     or topology["groups"] != stamp["groups"])):
            raise ValueError("paired-average stamp disagrees with its real topology")

    def check_paired_average_step(self, state, completed_steps):
        """Trainer validates the reaction step before any load mutation."""
        if state["paired_average"]["step"] > completed_steps:
            raise ValueError("paired-average stamp is from a future trainer step")

    def load_state_dict(self, state):
        self.check_state(state)
        # Base load calls self.check_state, so copy the validated base fields
        # directly while preserving the backend-specific schema validation.
        for key in self._TENSORS:
            value = state[key]
            setattr(self, key, None if value is None else value.clone().to(self.S.device))
        self.fill, self.cursor, self.rows_since_eval = (state["fill"], state["cursor"],
                                                        state["rows_since_eval"])
        self.counters, self.last = dict(state["counters"]), dict(state["last"])
        mean = dict(self.last["mean_transport"])
        mean["selected_axes"] = list(mean["selected_axes"])
        self.last["mean_transport"] = mean
        self.stream.set_state(state["stream"].cpu())
        self.sample_shape = None if state["sample_shape"] is None else tuple(state["sample_shape"])
        self.snapshot_serial = state["snapshot_serial"]
        self.paired_average = dict(state["paired_average"])
        self.snapshot = None
        self.moved_rows = None
        self._copy_parent_rows = None
        self.lineage.neighbors = state["lineage_neighbors"].clone().to(self.S.device)
        self.latent_geometry.clear()
        self._heads = None
