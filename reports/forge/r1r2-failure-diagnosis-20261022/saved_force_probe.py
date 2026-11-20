"""Read-only frozen Ring16 force/conditioning probe; never advances a trainer.

Uses certified final tensors, the original target law, and deterministic Gaussian
quadrature. Quadrature is a diagnostic integration rule, not replacement evidence
for any original sampling law or gate. No model/optimizer is constructed.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F


ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge')
ARCHIVE_COMMIT = '2cf7b0a4c796e29ace6cca12d2e553ad67211b5c'
PUBLICATIONS = {
    'ns_amsgrad': 'reports/forge/r1r2-energy-distance-repair/publication/results.json',
    'energy_amsgrad': 'reports/forge/r1r2-energy-distance-repair/publication/results.json',
    'energy_frozen_prior': 'reports/forge/r1r2-energy-prior-freeze-repair/publication/results.json',
    'scheduled_selected': 'reports/forge/recipe-prior-refactor/publication/results.json',
}
ROOTS = {
    'ns_amsgrad': ARCHIVE / 'r1r2-tier1-search-20261012/queue/r1r2-energy-distance-repair-v1/3e1e55b0f89d4a8c8cedd706dbd4e021',
    'energy_amsgrad': ARCHIVE / 'r1r2-tier1-search-20261012/queue/r1r2-energy-distance-repair-v1/219c852ff8c345768d5cd16407b04236',
    'energy_frozen_prior': ARCHIVE / 'r1r2-tier1-search-20261012/queue/r1r2-energy-prior-freeze-repair-v1/c168f1c417b04e1a9938d68b04b0556c',
    'scheduled_selected': ARCHIVE / 'recipe-owned-priors-20261010/queue/recipe-prior-refactor-selected-leaders-v1/a54bc96e82f34c9e98e7fa1f1d268a33',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mlp(x, p, *, critic=False):
    if critic:
        phase = x.unsqueeze(-1) * p['freqs']
        x = torch.cat((x, phase.sin().flatten(1), phase.cos().flatten(1)), 1)
    for i in (0, 2, 4):
        x = F.linear(x, p[f'net.{i}.weight'], p[f'net.{i}.bias'])
        if i != 4:
            x = F.leaky_relu(x, .2)
    return x.squeeze(-1) if critic else x


def tangent(x, p, dx, dp):
    for i in (0, 2, 4):
        w, b = p[f'net.{i}.weight'], p[f'net.{i}.bias']
        a = F.linear(x, w, b)
        da = F.linear(dx, w) + F.linear(x, dp[f'net.{i}.weight'], dp[f'net.{i}.bias'])
        if i != 4:
            da = da * torch.where(a > 0, torch.ones_like(a), torch.full_like(a, .2))
            a = F.leaky_relu(a, .2)
        x, dx = a, da
    return dx


def quadrature(d, order):
    q, w = np.polynomial.hermite.hermgauss(order)
    indices = list(itertools.product(range(order), repeat=d))
    points = torch.tensor([[float(q[i] * np.sqrt(2)) for i in row] for row in indices], dtype=torch.float64)
    weights = torch.tensor([float(np.prod([w[i] / np.sqrt(np.pi) for i in row])) for row in indices], dtype=torch.float64)
    assert abs(float(weights.sum()) - 1) < 1e-12
    return points, weights


def stats(t):
    t = t.detach().double().flatten()
    return {'min': float(t.min()), 'median': float(t.median()), 'max': float(t.max()), 'rms': float(t.square().mean().sqrt())}


def shape_drift(x, velocity, weight, means):
    assigned = torch.cdist(x.detach(), means).argmin(1)
    rows = []
    for k, mu in enumerate(means):
        mask = assigned == k
        xx, vv, ww = x[mask].detach(), velocity[mask].detach(), weight[mask]
        ww = ww / ww.sum()
        center, speed = (ww[:, None] * xx).sum(0), (ww[:, None] * vv).sum(0)
        centered, flow = xx - center, vv - speed
        cov = torch.einsum('b,bi,bj->ij', ww, centered, centered)
        drift = torch.einsum('b,bi,bj->ij', ww, centered, flow) + torch.einsum('b,bi,bj->ij', ww, flow, centered)
        radial = mu / mu.norm()
        tangential = torch.stack((-radial[1], radial[0]))
        residual = cov / .01 - torch.eye(2)
        original_error = residual.norm() / np.sqrt(2)
        original_drift = (residual * (drift / .01)).sum() / (residual.norm() * np.sqrt(2))
        rows.append({'mode': k, 'mass': float(weight[mask].sum()),
            'radial_variance_ratio': float(radial @ cov @ radial / .01),
            'tangent_variance_ratio': float(tangential @ cov @ tangential / .01),
            'radial_variance_drift': float(radial @ drift @ radial / .01),
            'tangent_variance_drift': float(tangential @ drift @ tangential / .01),
            'covariance_squared_error_drift': float(2 * ((cov / .01 - torch.eye(2)) * drift / .01).sum()),
            'original_covariance_error': float(original_error),
            'original_covariance_error_drift': float(original_drift),
            'center_squared_error_drift': float(2 * ((center - mu) * speed).sum() / .01)})
    keys = ['radial_variance_ratio', 'tangent_variance_ratio', 'radial_variance_drift',
            'tangent_variance_drift', 'covariance_squared_error_drift', 'original_covariance_error',
            'original_covariance_error_drift', 'center_squared_error_drift']
    return {'per_mode': rows, 'mean_over_modes': {k: sum(r[k] for r in rows) / 16 for k in keys},
        'radial_expanding_modes': sum(r['radial_variance_drift'] > 0 for r in rows),
        'tangent_expanding_modes': sum(r['tangent_variance_drift'] > 0 for r in rows),
        'covariance_improving_modes': sum(r['covariance_squared_error_drift'] < 0 for r in rows)}


def critic_probe(x, p, means, weights=None):
    x = x.detach().requires_grad_()
    score = mlp(x, p, critic=True)
    gradient, = torch.autograd.grad(score.sum(), x)
    labels = torch.cdist(x.detach(), means).argmin(1)
    per_mode = []
    for k, mu in enumerate(means):
        mask = labels == k
        delta, g = x[mask].detach() - mu, gradient[mask]
        ww = torch.ones(len(g), dtype=torch.float64) if weights is None else weights[mask]
        ww = ww / ww.sum()
        radial = mu / mu.norm()
        tangent_axis = torch.stack((-radial[1], radial[0]))
        per_mode.append({'mode': k, 'score_mean': float((ww * score[mask].detach()).sum()),
            'gradient_norm_mean': float((ww * g.norm(dim=1)).sum()),
            'gradient_radial_rms': float((ww * (g @ radial).square()).sum().sqrt()),
            'gradient_tangent_rms': float((ww * (g @ tangent_axis).square()).sum().sqrt()),
            'residual_score_gradient_inner_product': float((ww * (delta * g).sum(1)).sum())})
    return {'per_mode': per_mode, 'gradient_norm': stats(gradient.norm(dim=1)), 'score': stats(score),
        'hypothetical_ns_multiplier': stats(torch.sigmoid(-score))}


def inspect(role, root, order):
    state_path = root / 'provenance/provenance-state.pt'
    panel_path = root / 'observed-samples.pt'
    request = json.loads((root / 'request.json').read_text())['request']
    raw = json.loads((root / 'raw-result.json').read_text())
    publication_path = PUBLICATIONS[role]
    publication_bytes = subprocess.check_output(['git', 'show', ARCHIVE_COMMIT + ':' + publication_path])
    publication = json.loads(publication_bytes)
    certified = next(r for r in publication['certified_results'] if r['attempt_id'] == root.name)
    exact = publication['exact_certificate_hashes'][root.name]
    assert sha(root / 'request.json') == exact['request']
    assert sha(root / 'result.json') == exact['result']
    immutable_result = json.loads((root / 'result.json').read_text())
    assert immutable_result['raw']['result'] == raw
    if 'checkpoint' in certified:
        assert sha(state_path) == certified['checkpoint']['sha256']
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    panels = torch.load(panel_path, map_location='cpu', weights_only=True)
    # Original checkpoint and panel bytes are checked against their archived receipts.
    assert sha(panel_path) == raw['evidence']['saved_observer_outputs']['sha256']
    assert sha(state_path) == raw['evidence']['provenance_checkpoint']['sha256']
    source = request['source']
    snapshot = Path(source['snapshot_path'])
    for name, expected in source['files'].items():
        assert sha(snapshot / name) == expected, name
    t = state['trainer']
    canon = lambda obj: json.loads(json.dumps(obj))
    assert canon(t['recipe']) == canon(state['recipe']) == raw['recipe']
    for key, value in t['recipe'].items():
        if key in request['candidate']['recipe_overrides']:
            assert canon(value) == canon(request['candidate']['recipe_overrides'][key]), key
    assert t['completed_steps'] == panels[-1]['step'] == 1600
    models = t['models']
    parameter_names = [f'net.{i}.{name}' for i in (0, 2, 4) for name in ('weight', 'bias')]
    assert list(models['G']) == parameter_names
    assert list(models['D']) == ['freqs'] + parameter_names
    shapes = [[64, 4], [64], [64, 64], [64], [2, 64], [2]]
    assert [list(v.shape) for v in models['G'].values()] == shapes
    gp = {k: v.double().clone().requires_grad_() for k, v in models['G'].items()}
    dp = {k: v.double() for k, v in models['D'].items()}
    prior = models['prior']['z'].double().clone().requires_grad_()
    means = torch.tensor(request['tasks']['ring16_acquisition']['execution']['host_definition']['means'], dtype=torch.float64)
    task = request['tasks']['ring16_acquisition']
    spec = task['execution']['host_definition']
    assert spec['z_dim'] == 4 and spec['particles'] == 256 and spec['hidden'] == 64 and spec['layers'] == 2 and spec['fourier'] == 2
    assert torch.allclose(means.norm(dim=1), torch.full((16,), 3., dtype=torch.float64), atol=1e-12)
    assert spec['covariances'] == [[[.01, 0.], [0., .01]]] * 16 and spec['masses'] == [.0625] * 16
    assert state['prior_contract'] == task['execution']['prior_contract'] == 'recipe_owned_v1'
    assert {k: v for k, v in state['prior'].items() if k != 'learnable'} == task['execution']['prior'] == {'kind': 'mog', 'sigma': .1, 'standardize': False}
    assert state['prior']['learnable'] == (request['candidate']['recipe_overrides']['prior_update'] == 'learned')
    assert models['prior']['z'].shape == (256, 4) and models['prior']['sigma'] == torch.tensor(.1)
    points, qw = quadrature(4, order)
    z = (prior[:, None, :] + float(models['prior']['sigma']) * points[None, :, :]).flatten(0, 1)
    weight = qw.repeat(len(prior)) / len(prior)
    x = mlp(z, gp)
    score = mlp(x, dp, critic=True)
    rq, rw = quadrature(2, max(order, 7))
    real = (means[:, None, :] + .1 * rq[None, :, :]).flatten(0, 1)
    real_weights = rw.repeat(16) / 16
    rs = mlp(real, dp, critic=True)
    if t['recipe'].get('loss', 'relativistic') == 'non_saturating':
        multiplier = torch.sigmoid(-score.detach())
        loss = (weight * F.softplus(-score)).sum()
    else:
        # Expected paired relativistic generator loss under the frozen real law.
        # Detaching its multiplier preserves the exact first derivative and
        # avoids retaining the full target-by-prior quadrature computation graph.
        with torch.no_grad():
            multiplier = sum((torch.sigmoid(s - score.detach()) * w for s, w in zip(rs, real_weights)))
        loss = -(weight * multiplier * score).sum()
    output_gradient, = torch.autograd.grad(loss, x, retain_graph=True)
    free_output_velocity = -output_gradient.detach() / weight[:, None]
    named = list(gp) + ['prior.z']
    params = list(gp.values()) + [prior]
    grads = list(torch.autograd.grad(loss, params))
    genopt = t['optimizers'][0]
    saved = genopt['state']
    learned = state['prior']['learnable']
    expected_groups = [list(range(6)), [6]] if learned else [list(range(6))]
    assert [g['params'] for g in genopt['param_groups']] == expected_groups
    denominator, momentum, learning_rates, summaries = [], [], [], {}
    for group in genopt['param_groups']:
        ratios, denoms, moments = [], [], []
        for j in group['params']:
            s = saved[j]
            assert all(s[key].shape == params[j].shape for key in ('exp_avg', 'exp_avg_sq'))
            if group['amsgrad']:
                assert s['max_exp_avg_sq'].shape == params[j].shape
            n = float(s['step'])
            beta1, beta2 = group['betas']
            second = s.get('max_exp_avg_sq', s['exp_avg_sq']).double()
            den = (second / (1 - beta2 ** n)).sqrt() + group['eps']
            denominator.append(den)
            momentum.append(s['exp_avg'].double() / (1 - beta1 ** n))
            learning_rates.append(group['lr'])
            ratios.append((((s['exp_avg_sq'].double() / (1 - beta2 ** n)).sqrt() + group['eps']) / den).flatten())
            denoms.append(den.flatten()); moments.append(momentum[-1].flatten())
        summaries['prior' if len(group['params']) == 1 else 'generator'] = {
            'lr': group['lr'], 'betas': list(group['betas']), 'amsgrad': group['amsgrad'],
            'denominator': stats(torch.cat(denoms)), 'saved_momentum': stats(torch.cat(moments)),
            'amsgrad_speed_vs_current_second_moment': stats(torch.cat(ratios))}
    if not learned:
        denominator.append(torch.ones_like(prior)); momentum.append(torch.zeros_like(prior)); learning_rates.append(0.)
    assert len(denominator) == len(params) == len(grads)
    velocities = {}
    for method in ('current_sgd', 'current_frozen_denominator', 'saved_momentum'):
        directions = []
        for g, den, m, lr in zip(grads, denominator, momentum, learning_rates):
            directions.append(-lr * (g if method == 'current_sgd' else (m if method == 'saved_momentum' else g) / den))
        gd = dict(zip(gp, directions[:-1]))
        pdir = directions[-1][:, None, :].expand_as(z.reshape(len(prior), -1, 4)).reshape_as(z)
        zero_g = {k: torch.zeros_like(v) for k, v in gp.items()}
        gv = tangent(z.detach(), gp, torch.zeros_like(z), gd).detach()
        pv = tangent(z.detach(), gp, pdir, zero_g).detach()
        velocities[method] = {'G': shape_drift(x, gv, weight, means), 'prior': shape_drift(x, pv, weight, means),
            'combined': shape_drift(x, gv + pv, weight, means),
            'G_velocity': stats(gv), 'prior_velocity': stats(pv)}
    # Cross-check G-only, prior-only and combined JVPs by finite differences.
    eps = 1e-6
    gd = dict(zip(gp, [-g for g in grads[:-1]]))
    pd = -grads[-1][:, None, :].expand(len(prior), len(points), 4).reshape_as(z)
    jvp_errors = {}
    for name, dz, dg in [('G', torch.zeros_like(z), gd), ('prior', pd, {k: torch.zeros_like(v) for k,v in gp.items()}), ('combined', pd, gd)]:
        exact_jvp = tangent(z.detach(), gp, dz, dg).detach()
        plus = mlp(z.detach() + eps * dz, {k: v.detach() + eps * dg[k] for k, v in gp.items()})
        minus = mlp(z.detach() - eps * dz, {k: v.detach() - eps * dg[k] for k, v in gp.items()})
        jvp_errors[name] = float(((plus - minus) / (2 * eps) - exact_jvp).abs().max())
        assert jvp_errors[name] < 1e-7
    return {'artifact_root': str(root), 'checkpoint_sha256': sha(state_path), 'saved_panel_sha256': sha(panel_path),
        'request_sha256': sha(root / 'request.json'), 'executed_source': source['origin_commit'],
        'immutable_binding': {'commit': ARCHIVE_COMMIT, 'publication_path': publication_path,
            'publication_sha256': hashlib.sha256(publication_bytes).hexdigest(),
            'publication_git_blob': subprocess.check_output(['git', 'rev-parse', ARCHIVE_COMMIT + ':' + publication_path], text=True).strip(),
            'request_sha256': exact['request'], 'result_sha256': exact['result'], 'raw_matches_pinned_result': True,
            'recipe_prior_target_architecture_optimizer_bindings_checked': True},
        'source_digest': source['digest'], 'snapshot_files_verified': len(source['files']),
        'model_source_pins': {p: source['files'][p] for p in ('lib/toy_models.py', 'benchmarks/locked_shared/mlp.py')},
        'recipe': t['recipe'], 'original_recorded_metrics': raw['evidence']['live'],
        'learned_prior': learned, 'critic_on_saved_fake': critic_probe(panels[-1]['samples'].double(), dp, means),
        'actual_loss_generator_multiplier': stats(multiplier),
        'critic_on_target_quadrature': critic_probe(real, dp, means, real_weights),
        'optimizer_roles': summaries, 'adversarial_projected_drift': velocities,
        'unprojected_adversarial_drift': shape_drift(x, free_output_velocity, weight, means),
        'manual_jvp_max_abs_error': jvp_errors,
        'diagnostic_prior_points': len(x), 'target_quadrature_points': len(real)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--roles', nargs='+', choices=ROOTS, default=list(ROOTS))
    parser.add_argument('--compact-output', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    py, nr, tr = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def forbidden(*a, **kw):
        raise AssertionError('This saved-state diagnostic forbids random draws, models and optimizers')
    for name in ('rand', 'randn', 'randint', 'randperm', 'normal', 'multinomial'):
        setattr(torch, name, forbidden)
    torch.nn.Module.__init__ = forbidden
    torch.optim.Optimizer.__init__ = forbidden
    results = {}
    for role in args.roles:
        print('Inspecting ' + role, flush=True, file=__import__('sys').stderr)
        results[role] = inspect(role, ROOTS[role], args.order)
    assert random.getstate() == py and np.array_equal(np.random.get_state()[1], nr[1])
    assert np.random.get_state()[0] == nr[0] and np.random.get_state()[2:] == nr[2:]
    assert torch.equal(tr, torch.get_rng_state())
    result = {'schema_version': 1, 'scope': 'saved_endpoint_force_conditioning_diagnostic',
        'qualification_input': False, 'training_updates_added': 0, 'random_draws_added': 0,
        'ambient_rng_unchanged': True, 'torch_version': torch.__version__,
        'probe_sha256': sha(__file__), 'quadrature_order': args.order,
        'diagnostic_wall_seconds': time.monotonic() - started,
        'limits': ['Frozen endpoint only; no historical minibatches or critic refitting.',
            'Current-force projection excludes energy auxiliary and momentum; saved-momentum projection includes stored training history.',
            'Frozen Adam denominator is a local diagnostic, not a simulated optimizer step.',
            'Saved momentum is previous-update history projected at the final model, not a next Adam update.',
            'Squared covariance error drift is an analysis proxy; original covariance error drift uses the original relative-Frobenius formula with fixed labels.',
            'Tensor evaluation uses float64 CPU; trained states and original grades retain their original runtime.',
            'Nearest-target labels and quadrature are analysis-only; component boundaries held fixed for directional derivatives.'],
        'cohorts': results}
    if args.compact_output:
        def compact(value):
            if isinstance(value, dict):
                return {k: compact(v) for k, v in value.items() if k not in ('per_mode', 'recipe', 'original_recorded_metrics')}
            return value
        summary = compact(result)
        summary['raw_stdout_archive'] = str(ARCHIVE / 'r1r2-failure-diagnosis-20261022' / f'saved-force-order{args.order}.stdout.json')
        args.compact_output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
