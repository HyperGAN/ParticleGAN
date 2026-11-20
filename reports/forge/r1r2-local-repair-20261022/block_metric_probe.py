"""One fixed saved-state block-Kronecker prerequisite; never trains a model.

Run from the repository root. The learned-prior row blocks are exact; generator
affine blocks factor the mean output-Jacobian metric into activation/derivative
second moments. This is a new Euclidean primal metric, not saved-AMS exact GN.
"""
import argparse
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPOSITORY = Path.cwd()
REPORTS = REPOSITORY / 'reports/forge/r1r2-local-repair-20261022'
CORE = REPORTS.parent / 'r1r2-failure-diagnosis-20261022'
sys.path[:0] = [str(REPORTS), str(CORE)]
from saved_force_probe import ROOTS, inspect, mlp, sha, shape_drift, tangent
from gn_batch_probe import empirical_mmd_gradient, check_mmd_gradient
from mmd_projection_probe import WIDTHS
from replay_training_batches import recover

RIDGE_FRACTION = .01
RATE = .0085


def kron_solve(a, b, gradient):
    """Solve B D A + lambda D = -lr gradient, row-major affine parameters."""
    a, b, gradient = (v.detach() for v in (a, b, gradient))
    assert all(torch.isfinite(v).all() for v in (a, b, gradient))
    ea, qa = torch.linalg.eigh(a)
    eb, qb = torch.linalg.eigh(b)
    assert float(ea.min()) > -1e-10 * max(float(a.norm()), 1.)
    assert float(eb.min()) > -1e-10 * max(float(b.norm()), 1.)
    ea, eb = ea.clamp_min(0), eb.clamp_min(0)
    mean_diagonal = float(a.trace() * b.trace()) / gradient.numel()
    ridge = RIDGE_FRACTION * mean_diagonal
    if ridge == 0:
        assert torch.count_nonzero(gradient) == 0
        return torch.zeros_like(gradient), {'ridge': 0., 'mean_diagonal': 0., 'relative_residual': 0.}
    denominator = eb[:, None] * ea[None, :] + ridge
    delta = -RATE * qb @ ((qb.T @ gradient @ qa) / denominator) @ qa.T
    residual = b @ delta @ a + ridge * delta + RATE * gradient
    relative = float(residual.norm()) / max(float((RATE * gradient).norm()), 1e-30)
    assert relative < 1e-9 and torch.isfinite(delta).all()
    return delta, {'ridge': ridge, 'mean_diagonal': mean_diagonal, 'relative_residual': relative}


def row_solve(blocks, gradient):
    """Exact independent learned-prior row solves; unseen rows stay exactly zero."""
    blocks, gradient = blocks.detach(), gradient.detach()
    ridge = RIDGE_FRACTION * blocks.diagonal(dim1=1, dim2=2).mean(1)
    active = ridge > 0
    assert torch.count_nonzero(gradient[~active]) == 0
    delta = torch.zeros_like(gradient)
    matrix = blocks[active] + ridge[active, None, None] * torch.eye(blocks.shape[1], dtype=blocks.dtype)
    delta[active] = torch.linalg.solve(matrix, -RATE * gradient[active, :, None]).squeeze(-1)
    residual = (blocks @ delta[:, :, None]).squeeze(-1) + ridge[:, None] * delta + RATE * gradient
    relative = float(residual.norm()) / max(float((RATE * gradient).norm()), 1e-30)
    assert relative < 1e-9 and torch.isfinite(delta).all()
    return delta, {'ridge_min_active': float(ridge[active].min()), 'ridge_max': float(ridge.max()),
        'active_rows': int(active.sum()), 'relative_residual': relative}


def correctness_oracle():
    # Cartesian activation/derivative fixtures make the Kronecker factorization
    # exact. No RNG is consumed and no training fixture is substituted.
    activations = torch.tensor([[1., -2., 1.], [2., 1., 1.], [-1., .5, 1.]], dtype=torch.float64)
    derivatives = torch.arange(16, dtype=torch.float64).reshape(4, 2, 2) / 13 - .4
    jacobians = torch.stack([torch.einsum('ci,j->cij', b, a).reshape(2, 6)
        for a in activations for b in derivatives])
    exact = torch.einsum('nci,ncj->ij', jacobians, jacobians) / len(jacobians)
    a = activations.T @ activations / len(activations)
    b = torch.einsum('nci,ncj->ij', derivatives, derivatives) / len(derivatives)
    factor_error = float((exact - torch.kron(b, a)).abs().max())
    gradient = torch.arange(6, dtype=torch.float64).reshape(2, 3) / 7 - .3
    delta, receipt = kron_solve(a, b, gradient)
    ridge = RIDGE_FRACTION * float(exact.trace()) / len(exact)
    direct = torch.linalg.solve(exact + ridge * torch.eye(6, dtype=exact.dtype), -RATE * gradient.flatten()).reshape_as(delta)
    solve_error = float((delta - direct).abs().max())
    # Damping must resolve a nonzero rank-deficient block, with exact joint
    # eigenvalue damping rather than separately damped factors.
    rank_a = torch.tensor([[1., 1.], [1., 1.]], dtype=torch.float64)
    rank_b = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64)
    rank_delta, _ = kron_solve(rank_a, rank_b, gradient[:, :2])
    rank_matrix = torch.kron(rank_b, rank_a)
    rank_ridge = RIDGE_FRACTION * float(rank_matrix.trace()) / 4
    rank_direct = torch.linalg.solve(rank_matrix + rank_ridge * torch.eye(4, dtype=rank_matrix.dtype), -RATE * gradient[:, :2].flatten())
    rank_error = float((rank_delta.flatten() - rank_direct).abs().max())
    zeros, zero_receipt = kron_solve(torch.zeros_like(rank_a), rank_b, torch.zeros(2, 2, dtype=torch.float64))
    assert torch.count_nonzero(zeros) == 0 and zero_receipt['ridge'] == 0
    row_blocks = torch.stack((rank_a, torch.zeros_like(rank_a), 2 * torch.eye(2, dtype=torch.float64)))
    row_gradient = torch.tensor([[1., 2.], [0., 0.], [-1., .5]], dtype=torch.float64)
    row_delta, row_receipt = row_solve(row_blocks, row_gradient)
    assert torch.count_nonzero(row_delta[1]) == 0
    assert max(factor_error, solve_error, rank_error) < 1e-10
    return {'factorized_metric_max_abs_error': factor_error, 'dense_solve_max_abs_error': solve_error,
        'rank_deficient_solve_max_abs_error': rank_error, 'relative_residual': receipt['relative_residual'],
        'prior_row_relative_residual': row_receipt['relative_residual'], 'inactive_exact_zero': True}


def objective(x, real, dp):
    loss = F.softplus(-mlp(x, dp, critic=True)).mean()
    for width in WIDTHS:
        xx = torch.exp(-(x[:, None] - x[None, :]).square().sum(2) / (2 * width**2))
        xy = torch.exp(-(x[:, None] - real[None, :]).square().sum(2) / (2 * width**2))
        yy = torch.exp(-(real[:, None] - real[None, :]).square().sum(2) / (2 * width**2))
        loss = loss + ((xx.sum() - xx.diag().sum()) / (len(x) * (len(x)-1))
            - 2 * xy.mean() + (yy.sum() - yy.diag().sum()) / (len(real) * (len(real)-1))) / len(WIDTHS)
    return loss


def block_direction(z, gp, prior, indices, gradient):
    n = len(z)
    with torch.no_grad():
        a0 = F.linear(z, gp['net.0.weight'], gp['net.0.bias'])
        h0 = F.leaky_relu(a0, .2)
        a2 = F.linear(h0, gp['net.2.weight'], gp['net.2.bias'])
        h2 = F.leaky_relu(a2, .2)
        d0 = torch.where(a0 > 0, torch.ones_like(a0), torch.full_like(a0, .2))
        d2 = torch.where(a2 > 0, torch.ones_like(a2), torch.full_like(a2, .2))
        b4 = torch.eye(2, dtype=z.dtype).expand(n, -1, -1)
        b2 = gp['net.4.weight'][None] * d2[:, None]
        b0 = torch.einsum('nci,ij->ncj', b2, gp['net.2.weight']) * d0[:, None]
        jp = b0 @ gp['net.0.weight']
        deltas, receipts = [], []
        for i, inputs, derivatives in ((0, z, b0), (2, h0, b2), (4, h2, b4)):
            augmented = torch.cat((inputs, torch.ones(n, 1, dtype=z.dtype)), 1)
            a = augmented.T @ augmented / n
            b = torch.einsum('nci,ncj->ij', derivatives, derivatives) / n
            g = torch.cat((gradient[2 * (i//2)], gradient[2 * (i//2)+1][:, None]), 1)
            delta, receipt = kron_solve(a, b, g)
            deltas.extend((delta[:, :-1], delta[:, -1]))
            # Exact affine trace differs from the product-of-means approximation.
            exact_trace = float((augmented.square().sum(1) * derivatives.square().sum((1, 2))).mean())
            receipts.append({'layer': i, 'exact_block_trace': exact_trace,
                'factorized_block_trace': float(a.trace() * b.trace()), **receipt})
        blocks = torch.zeros(len(prior), prior.shape[1], prior.shape[1], dtype=z.dtype)
        blocks.index_add_(0, indices, torch.einsum('nci,ncj->nij', jp, jp) / n)
        delta, prior_receipt = row_solve(blocks, gradient[-1])
        deltas.append(delta)
    return deltas, receipts, prior_receipt, jp, blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    py, nr, tr = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all()
    oracle = correctness_oracle()
    root = ROOTS['ns_amsgrad']
    binding = inspect('ns_amsgrad', root, 3)
    batches, replay = recover(root, keep=16)
    def forbidden(*a, **kw):
        raise AssertionError('No draws, models or optimizers after historical replay')
    for name in ('rand', 'randn', 'randint', 'randperm', 'normal', 'multinomial'):
        setattr(torch, name, forbidden)
    torch.nn.Module.__init__ = forbidden
    torch.optim.Optimizer.__init__ = forbidden
    state = torch.load(root / 'provenance/provenance-state.pt', map_location='cpu', weights_only=True)
    models = state['trainer']['models']
    gp = {k: v.double().clone().requires_grad_() for k, v in models['G'].items()}
    dp = {k: v.double() for k, v in models['D'].items()}
    prior = models['prior']['z'].double().clone().requires_grad_()
    params = list(gp.values()) + [prior]
    assert len(gp) == 6 and [tuple(p.shape) for p in params] == [(64, 4), (64,), (64, 64), (64,), (2, 64), (2,), (256, 4)]
    assert all(g['lr'] == RATE for g in state['trainer']['optimizers'][0]['param_groups'])
    assert state['recipe']['prior_reg'] == 0
    total = [torch.zeros_like(p) for p in params]
    rows, timings = [], []
    first_checks = {}
    for number, batch in enumerate(batches):
        print('Historical batch ' + str(batch['step']), file=sys.stderr, flush=True)
        iteration_started = time.monotonic()
        indices = batch['indices']
        z = prior[indices] + float(models['prior']['sigma']) * batch['noise'].double()
        x = mlp(z, gp)
        loss = objective(x, batch['real'].double(), dp)
        gradient = torch.autograd.grad(loss, params, retain_graph=True)
        delta, layers, prior_receipt, jp, blocks = block_direction(z, gp, prior, indices, gradient)
        timings.append(time.monotonic() - iteration_started)
        for out, value in zip(total, delta):
            out += value / len(batches)
        if number == 0:
            adv, = torch.autograd.grad(F.softplus(-mlp(x, dp, critic=True)).mean(), x, retain_graph=True)
            output_grad, = torch.autograd.grad(loss, x, retain_graph=True)
            first_checks['full_objective_output_gradient_error'] = float((output_grad - adv - empirical_mmd_gradient(x.detach(), batch['real'].double())).abs().max())
            row = int(indices[0])
            exact_prior = torch.zeros_like(blocks[row])
            for i in (indices == row).nonzero().flatten():
                for c in (0, 1):
                    grad, = torch.autograd.grad(x[int(i), c], prior, retain_graph=True)
                    exact_prior += torch.outer(grad[row], grad[row]) / len(x)
            first_checks['prior_block_autograd_max_abs_error'] = float((exact_prior - blocks[row]).abs().max())
            length = sum(float(d.square().sum()) for d in delta)**.5
            unit = [d / length for d in delta]
            epsilon = 1e-5
            losses = []
            for sign in (-1, 1):
                pg = {k: (v.detach() + sign * epsilon * d) for (k, v), d in zip(gp.items(), unit[:-1])}
                pp = prior.detach() + sign * epsilon * unit[-1]
                xx = mlp(pp[indices] + float(models['prior']['sigma']) * batch['noise'].double(), pg)
                losses.append(float(objective(xx, batch['real'].double(), dp)))
            observed = (losses[1] - losses[0]) / (2 * epsilon)
            predicted = sum(float((g * d).sum()) for g, d in zip(gradient, unit))
            first_checks['full_parameter_objective_finite_difference_abs_error'] = abs(observed - predicted)
            assert max(first_checks.values()) < 1e-7
        rows.append({'step': batch['step'], 'layers': layers, 'prior': prior_receipt,
            'full_objective': float(loss.detach()), 'parameter_delta_norm': sum(float(d.square().sum()) for d in delta)**.5})
    request = json.loads((root / 'request.json').read_text())['request']
    means = torch.tensor(request['tasks']['ring16_acquisition']['execution']['host_definition']['means'], dtype=torch.float64)
    panels = {}
    for name, offsets in [('axes', torch.cat((2 * torch.eye(4), -2 * torch.eye(4))).double()),
        ('corners', torch.tensor(list(itertools.product((-1., 1.), repeat=4)), dtype=torch.float64))]:
        z = (prior[:, None] + float(models['prior']['sigma']) * offsets[None]).flatten(0, 1)
        x = mlp(z, gp)
        dz = total[-1][:, None].expand(len(prior), len(offsets), 4).reshape_as(z)
        gv = tangent(z.detach(), gp, torch.zeros_like(z), dict(zip(gp, total[:-1]))).detach()
        pv = tangent(z.detach(), gp, dz, {k: torch.zeros_like(v) for k, v in gp.items()}).detach()
        weights = torch.full((len(x),), 1 / len(x), dtype=torch.float64)
        directions = {key: shape_drift(x, value, weights, means) for key, value in [('G', gv), ('prior', pv), ('combined', gv + pv)]}
        metrics = directions['combined']['mean_over_modes']
        signs = {'covariance_decreases': metrics['original_covariance_error_drift'] < 0,
            'radial_variance_increases': metrics['radial_variance_drift'] > 0,
            'tangent_variance_nonincreases': metrics['tangent_variance_drift'] <= 0,
            'center_squared_error_nonincreases': metrics['center_squared_error_drift'] <= 0}
        panels[name] = {'points': len(x), 'directions': directions, 'prerequisite_signs': signs, 'all_signs_pass': all(signs.values())}
    assert random.getstate() == py
    assert np.random.get_state()[0] == nr[0] and np.array_equal(np.random.get_state()[1], nr[1]) and np.random.get_state()[2:] == nr[2:]
    assert torch.equal(torch.get_rng_state(), tr)
    assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
    accepted = all(panel['all_signs_pass'] for panel in panels.values())
    helpers = [REPORTS / 'gn_batch_probe.py', REPORTS / 'replay_training_batches.py', CORE / 'saved_force_probe.py', CORE / 'mmd_projection_probe.py']
    result = {'schema_version': 1, 'scope': 'frozen_endpoint_block_Kronecker_actual128batch_prerequisite',
        'qualification_input': False, 'training_updates_added': 0, 'new_independent_random_streams': 0,
        'ambient_rng_unchanged': True, 'historical_stream_replay': replay,
        'immutable_binding': binding['immutable_binding'], 'checkpoint_sha256': binding['checkpoint_sha256'],
        'probe_sha256': sha(__file__), 'helper_sha256': {str(p.relative_to(REPOSITORY)): sha(p) for p in helpers},
        'analysis_checkout_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'metric': 'Euclidean primal output-Jacobian block approximation; not saved-AMS-whitened exact GN',
        'equation': '(B_layer tensor_product A_layer + lambda_layer I) delta_layer = -lr full_gradient_layer',
        'A': 'mean of augmented-input outer products; bias in same affine block',
        'B': 'mean of output-Jacobian-to-preactivation Gram matrices, summed output coordinates',
        'prior_metric': 'exact per-row mean_n Jz(n)^T Jz(n) accumulated only where row selected; no generator/prior cross blocks',
        'damping': '.01 * trace(each block) / parameter coordinates in that block; exact joint eigenvalue damping',
        'rate': RATE, 'relative_damping': RIDGE_FRACTION, 'objective': 'full scalar non-saturating generator loss + unbiased empirical seven-width MMD U at coefficient1',
        'MMD_widths': WIDTHS, 'MMD_coefficient': 1, 'correctness_oracle': oracle,
        'empirical_MMD_gradient_max_abs_error': check_mmd_gradient(), 'saved_graph_checks': first_checks,
        'batches': rows, 'panels': panels, 'prerequisite_passed': accepted,
        'decision': 'saved prerequisite passes; no trained or runtime qualification' if accepted else 'REJECT fixed block approximation; stop without tuning or training',
        'cost': {'device': 'CPU', 'dtype': 'float64', 'threads': 1, 'batch_gradient_metric_solve_seconds': timings,
            'batch_mean_seconds': sum(timings) / len(timings), 'batch_max_seconds': max(timings),
            'diagnostic_wall_seconds': time.monotonic() - started},
        'limits': ['One archived NS/AMS endpoint; no historical intermediate model reconstruction.',
            'Original16 suffix real/latent draws evaluated independently at same frozen endpoint; arithmetic mean direction.',
            'Generator Kronecker independence approximation and block separation discard within-layer correlations and all cross-block couplings.',
            'Damping policy is per-primal-block trace normalization, distinct from exactGN dual-output trace normalization.',
            'No saved AMS metric, momentum or future moment update is included; this is an explicitly new hypothetical optimizer.',
            'Target means used only in two fixed scoring panels, never in objective or metric.',
            'No word all-host feasibility claim and no paid experiment, trained gate, leaderboard or default change.']}
    def compact(v):
        if isinstance(v, dict):
            return {k: compact(x) for k, x in v.items() if k != 'per_mode'}
        if isinstance(v, list):
            return [compact(x) for x in v]
        return v
    args.output.write_text(json.dumps(compact(result), indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'decision': result['decision'], 'panels': {k: v['directions']['combined']['mean_over_modes'] for k, v in panels.items()}, 'cost': result['cost']}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
