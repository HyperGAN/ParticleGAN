"""Analysis-only fixed archived MMD force projected through saved NS G/AMS.

The finite panel uses all 256 saved prior rows and eight deterministic axial
sigma points per row. It matches Gaussian first/second moments but is not the
training/evaluation sampling law and supplies no gate or trained result.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from saved_force_probe import ROOTS, inspect, mlp, sha, shape_drift, stats, tangent

WIDTHS = (.0625, .125, .25, .5, 1., 2., 4.)


def mmd_force(x, means):
    """Exact derivative for empirical fake U and integrated original real law."""
    n = len(x)
    out = torch.zeros_like(x)
    for first in range(0, n, 128):
        last = first + 128
        xx = x[first:last]
        delta = xx[:, None, :] - x[None, :, :]
        real_delta = xx[:, None, :] - means[None, :, :]
        dist2, real_dist2 = delta.square().sum(2), real_delta.square().sum(2)
        for width in WIDTHS:
            s2 = width ** 2
            fake = torch.exp(-dist2 / (2 * s2))
            real = s2 / (s2 + .01) * torch.exp(-real_dist2 / (2 * (s2 + .01)))
            out[first:last] += (-2 / (n * (n - 1) * s2) * (fake[:, :, None] * delta).sum(1)
                + 2 / (n * (s2 + .01)) * (real[:, :, None] * real_delta).mean(1)) / len(WIDTHS)
    return out


def numerical_check(means):
    x = (torch.arange(14, dtype=torch.float64).reshape(7, 2) / 7 - 1).requires_grad_()
    loss = torch.tensor(0., dtype=torch.float64)
    for width in WIDTHS:
        s2 = width ** 2
        kernel = torch.exp(-(x[:, None] - x[None, :]).square().sum(2) / (2 * s2))
        real = s2 / (s2 + .01) * torch.exp(-(x[:, None] - means[None, :]).square().sum(2) / (2 * (s2 + .01)))
        loss = loss + ((kernel.sum() - kernel.diag().sum()) / (len(x) * (len(x) - 1)) - 2 * real.mean()) / len(WIDTHS)
    g, = torch.autograd.grad(loss, x)
    error = float((g - mmd_force(x.detach(), means)).abs().max())
    assert error < 1e-10
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compact-output', type=Path)
    parser.add_argument('--panel', choices=['axes', 'corners'], default='axes')
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    py, nr, tr = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def forbidden(*a, **kw):
        raise AssertionError('Saved-state MMD probe forbids draws/models/optimizers')
    for name in ('rand', 'randn', 'randint', 'randperm', 'normal', 'multinomial'):
        setattr(torch, name, forbidden)
    torch.nn.Module.__init__ = forbidden
    torch.optim.Optimizer.__init__ = forbidden
    root = ROOTS['ns_amsgrad']
    binding = inspect('ns_amsgrad', root, 3)
    state_path = root / 'provenance/provenance-state.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    model = state['trainer']['models']
    gp = {k: v.double().clone().requires_grad_() for k, v in model['G'].items()}
    dp = {k: v.double() for k, v in model['D'].items()}
    prior = model['prior']['z'].double().clone().requires_grad_()
    request = json.loads((root / 'request.json').read_text())['request']
    means = torch.tensor(request['tasks']['ring16_acquisition']['execution']['host_definition']['means'], dtype=torch.float64)
    axes = (torch.cat((2 * torch.eye(4), -2 * torch.eye(4))).double() if args.panel == 'axes'
            else torch.tensor(list(itertools.product((-1., 1.), repeat=4)), dtype=torch.float64))
    z = (prior[:, None, :] + float(model['prior']['sigma']) * axes[None, :, :]).flatten(0, 1)
    x = mlp(z, gp)
    weights = torch.full((len(x),), 1 / len(x), dtype=torch.float64)
    adversarial = F.softplus(-mlp(x, dp, critic=True)).mean()
    mmd_gradient = mmd_force(x.detach(), means)
    mmd_pseudo = (x * mmd_gradient).sum()
    params = list(gp.values()) + [prior]
    opt = state['trainer']['optimizers'][0]
    denom, rates = [], []
    for group in opt['param_groups']:
        for j in group['params']:
            s = opt['state'][j]
            denom.append((s['max_exp_avg_sq'].double() / (1 - group['betas'][1] ** float(s['step']))).sqrt() + group['eps'])
            rates.append(group['lr'])
    results = {}
    for name, loss in [('adversarial', adversarial), ('mmd', mmd_pseudo), ('combined', adversarial + mmd_pseudo)]:
        grads = torch.autograd.grad(loss, params, retain_graph=True)
        directions = [-lr * g / d for g, d, lr in zip(grads, denom, rates)]
        gd = dict(zip(gp, directions[:-1]))
        pd = directions[-1][:, None, :].expand(len(prior), len(axes), 4).reshape_as(z)
        gv = tangent(z.detach(), gp, torch.zeros_like(z), gd).detach()
        pv = tangent(z.detach(), gp, pd, {k: torch.zeros_like(v) for k,v in gp.items()}).detach()
        force, = torch.autograd.grad(loss, x, retain_graph=True)
        results[name] = {'unprojected': shape_drift(x, -force / weights[:, None], weights, means),
            'G': shape_drift(x, gv, weights, means), 'prior': shape_drift(x, pv, weights, means),
            'combined': shape_drift(x, gv + pv, weights, means),
            'G_velocity': stats(gv), 'prior_velocity': stats(pv)}
    assert random.getstate() == py and np.array_equal(np.random.get_state()[1], nr[1])
    assert np.random.get_state()[0] == nr[0] and np.random.get_state()[2:] == nr[2:]
    assert torch.equal(tr, torch.get_rng_state())
    result = {'schema_version': 1, 'scope': 'saved_endpoint_analysis_only_finite_panel_mmd_projection',
        'qualification_input': False, 'training_updates_added': 0, 'random_draws_added': 0,
        'ambient_rng_unchanged': True, 'probe_sha256': sha(__file__),
        'core_probe_sha256': sha(Path(__file__).with_name('saved_force_probe.py')),
        'immutable_binding': binding['immutable_binding'], 'checkpoint_sha256': binding['checkpoint_sha256'],
        'widths': WIDTHS, 'auxiliary_coefficient': 1, 'finite_panel': {'points': len(x), 'rule': args.panel,
            'law': ('all256 saved rows times8 equal-weight antithetic axial points at +/-2sigma along each of4 axes'
                if args.panel == 'axes' else 'all256 saved rows times16 equal-weight +/-1sigma hypercube corners in4 dimensions'),
            'original_gaussian_law_replaced': False, 'diagnostic_approximation_only': True},
        'analytic_force_max_abs_error': numerical_check(means), 'directions': results,
        'diagnostic_wall_seconds': time.monotonic() - started,
        'limits': ['Finite deterministic panel; not a minibatch force or Gaussian-integrated generator law.',
            'No momentum/denominator updates or future critic refitting; fixed endpoint Jacobian.',
            'Signed fake U statistic and exact uniform16-Gaussian target kernel expectation; no center-only target.',
            'No trained result, qualification, API mechanism or paid experiment follows from directional signs.']}
    if args.compact_output:
        def compact(v):
            if isinstance(v, dict):
                return {k: compact(x) for k,x in v.items() if k != 'per_mode'}
            return v
        args.compact_output.write_text(json.dumps(compact(result), indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
