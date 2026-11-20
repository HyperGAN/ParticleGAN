"""Saved-only prerequisite for a data-local density repair, not a trained gate."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import itertools

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'r1r2-failure-diagnosis-20261022'))
from saved_force_probe import ROOTS, inspect, mlp, sha, shape_drift, stats, tangent


def conditional_moments(fake, real):
    x, y = fake.flatten(1), real.detach().flatten(1)
    n, dim = x.shape
    distances = torch.cdist(y, y).square()
    other = distances.clone()
    other.fill_diagonal_(float('inf'))
    scale = (y - y.mean(0)).square().mean()
    if float(scale) == 0:
        return x.sum() * 0
    floor = torch.finfo(x.dtype).eps * scale
    width = other.kthvalue(min(4, n-1), dim=0).values.clamp_min(floor)
    losses = []
    for multiplier in (1., 2., 4.):
        p = torch.exp(-distances / (2 * multiplier**2 * width[None, :]))
        q = torch.exp(-torch.cdist(x, y).square() / (2 * multiplier**2 * width[None, :]))
        p = p / p.sum(0)
        qsum = q.sum(0)
        effective = qsum.square() / q.square().sum(0).clamp_min(torch.finfo(x.dtype).tiny)
        q = q / qsum.clamp_min(torch.finfo(x.dtype).tiny)
        mp, mq = p.T @ y, q.T @ x
        cp = torch.einsum('ba,bai,baj->aij', p, y[:, None, :] - mp[None, :, :], y[:, None, :] - mp[None, :, :])
        cq = torch.einsum('ba,bai,baj->aij', q, x[:, None, :] - mq[None, :, :], x[:, None, :] - mq[None, :, :])
        normp = cp.square().sum((1, 2))
        tracep = cp.diagonal(dim1=1, dim2=2).sum(1)
        valid = (tracep > floor) & (effective.detach() >= 2) & (qsum.detach() > torch.finfo(x.dtype).eps)
        residual = (mq-mp).square().sum(1) / tracep.clamp_min(floor)
        residual = residual + (cq-cp).square().sum((1, 2)) / normp.clamp_min(floor.square())
        losses.append(torch.where(valid, residual, torch.zeros_like(residual)).mean())
    return torch.stack(losses).mean()


def score_transport(fake, real):
    x, y = fake.flatten(1), real.detach().flatten(1)
    with torch.no_grad():
        rr = torch.cdist(y, y).square()
        rr.fill_diagonal_(float('inf'))
        width = rr.kthvalue(min(4, len(y)-1), dim=0).values
        scale = (y-y.mean(0)).square().mean()
        if float(scale) == 0:
            return x.sum() * 0
        floor = torch.finfo(x.dtype).eps * scale
        xy = torch.cdist(x.detach(), y).square()
        h2 = width[xy.argmin(1)].clamp_min(floor)
        xx = torch.cdist(x.detach(), x.detach()).square()
        logits = -xx / (2 * h2[:, None])
        logits.fill_diagonal_(-float('inf'))
        fake_mean = logits.softmax(1) @ x.detach()
        real_mean = (-xy / (2 * h2[:, None])).softmax(1) @ y
        force = (real_mean - fake_mean) / h2[:, None]
    return -(x * force).sum(1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', choices=('axes', 'corners'), default='axes')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mechanism', choices=('local', 'conditional', 'score'), default='score')
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    py, nr, tr = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def forbidden(*a, **kw):
        raise AssertionError('Saved probe forbids models, optimizers and draws')
    for name in ('rand', 'randn', 'randint', 'randperm', 'normal', 'multinomial'):
        setattr(torch, name, forbidden)
    torch.nn.Module.__init__ = forbidden
    torch.optim.Optimizer.__init__ = forbidden
    root = ROOTS['ns_amsgrad']
    binding = inspect('ns_amsgrad', root, 3)
    s = torch.load(root / 'provenance/provenance-state.pt', map_location='cpu', weights_only=True)
    t, models = s['trainer'], s['trainer']['models']
    gp = {k: v.double().clone().requires_grad_() for k,v in models['G'].items()}
    dp = {k: v.double() for k,v in models['D'].items()}
    prior = models['prior']['z'].double().clone().requires_grad_()
    request = json.loads((root / 'request.json').read_text())['request']
    means = torch.tensor(request['tasks']['ring16_acquisition']['execution']['host_definition']['means'], dtype=torch.float64)
    sigma = float(models['prior']['sigma'])
    offsets = (torch.cat((2 * torch.eye(4), -2 * torch.eye(4))).double() if args.panel == 'axes'
        else torch.tensor(list(itertools.product((-1., 1.), repeat=4)), dtype=torch.float64))
    z = (prior[:, None, :] + sigma * offsets[None, :, :]).flatten(0, 1)
    x = mlp(z, gp)
    weight = torch.full((len(x),), 1 / len(x), dtype=torch.float64)
    # A deterministic2D cubature fixture; mean and covariance equal original law.
    # Recipe function gets only these coordinates, never component means/labels.
    angles = torch.arange(8, dtype=torch.float64) * (torch.pi / 4) + (torch.pi / 16 if args.panel == 'corners' else 0)
    real_offsets = np.sqrt(2) * .1 * torch.stack((angles.cos(), angles.sin()), 1)
    real = (means[:, None, :] + real_offsets[None, :, :]).flatten(0, 1)
    path = HERE.parents[2] / 'particlegan/kinetic_transport.py'
    spec = importlib.util.spec_from_file_location('bound_local_transport', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mechanism = {'local': module.kinetic_transport_local_loss, 'conditional': conditional_moments, 'score': score_transport}[args.mechanism]
    local = sum(mechanism(batch, real) for batch in x.split(128)) / (len(x) / 128)
    adversarial = F.softplus(-mlp(x, dp, critic=True)).mean()
    params = list(gp.values()) + [prior]
    opt = t['optimizers'][0]
    denominator, rates = [], []
    for group in opt['param_groups']:
        for j in group['params']:
            saved = opt['state'][j]
            denominator.append((saved['max_exp_avg_sq'].double() / (1 - group['betas'][1] ** float(saved['step']))).sqrt() + group['eps'])
            rates.append(group['lr'])
    results = {}
    for name, loss in [('adversarial', adversarial), ('local', local), ('combined', adversarial + local)]:
        grads = torch.autograd.grad(loss, params, retain_graph=True)
        directions = [-lr * g / d for g,d,lr in zip(grads, denominator, rates)]
        gd = dict(zip(gp, directions[:-1]))
        pd = directions[-1][:, None, :].expand(len(prior), len(offsets), 4).reshape_as(z)
        gv = tangent(z.detach(), gp, torch.zeros_like(z), gd).detach()
        pv = tangent(z.detach(), gp, pd, {k: torch.zeros_like(v) for k,v in gp.items()}).detach()
        output_gradient, = torch.autograd.grad(loss, x, retain_graph=True)
        results[name] = {'loss': float(loss.detach()),
            'unprojected': shape_drift(x, -output_gradient / weight[:, None], weight, means),
            'G': shape_drift(x, gv, weight, means), 'prior': shape_drift(x, pv, weight, means),
            'combined': shape_drift(x, gv + pv, weight, means),
            'G_velocity': stats(gv), 'prior_velocity': stats(pv)}
    assert random.getstate() == py and np.array_equal(np.random.get_state()[1], nr[1])
    assert np.random.get_state()[0] == nr[0] and np.random.get_state()[2:] == nr[2:]
    assert torch.equal(tr, torch.get_rng_state())
    result = {'schema_version': 1, 'scope': 'saved_endpoint_local_density_prerequisite',
        'qualification_input': False, 'training_updates_added': 0, 'random_draws_added': 0,
        'ambient_rng_unchanged': True, 'probe_sha256': sha(__file__),
        'core_probe_sha256': sha(HERE.parent / 'r1r2-failure-diagnosis-20261022/saved_force_probe.py'),
        'immutable_binding': binding['immutable_binding'], 'checkpoint_sha256': binding['checkpoint_sha256'],
        'mechanism_source': {'path': 'particlegan/kinetic_transport.py', 'sha256': sha(path)},
        'recipe_delta': {'auxiliary': args.mechanism, 'weight': [0, 1]},
        'panel': {'rule': args.panel, 'fake_points': len(x), 'equal_training_batch_size': 128,
            'target': '16 original Gaussian means times8 radius-sqrt2sigma octagon points; covariance exact, higher moments approximate',
            'objective_uses_component_labels_or_means': False},
        'directions': results, 'diagnostic_wall_seconds': time.monotonic() - started,
        'limits': ['Endpoint linearization only; not a trained gate or qualification.',
            'Fixed analytical cubature fixtures replace random draws for diagnosis only.',
            'Same target128-point cubature fixture reused across deterministic fake128 blocks; actual training batches remain independent.',
            'Saved Adam denominator fixed; no future moment update, momentum or critic refitting.']}
    if args.output:
        def compact(v):
            if isinstance(v, dict):
                return {k: compact(x) for k,x in v.items() if k != 'per_mode'}
            return v
        args.output.write_text(json.dumps(compact(result), indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
