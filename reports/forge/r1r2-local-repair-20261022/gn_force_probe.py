"""Saved-only damped output Gauss-Newton prerequisite; no trainer update."""
import argparse
import itertools
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'r1r2-failure-diagnosis-20261022'))
from saved_force_probe import ROOTS, inspect, mlp, sha, shape_drift, stats, tangent
from mmd_projection_probe import mmd_force, WIDTHS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', choices=('axes', 'corners'), default='axes')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-iterations', type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    py, nr, tr = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def forbidden(*a, **kw):
        raise AssertionError('Saved diagnostic forbids draws, models and optimizers')
    for name in ('rand', 'randn', 'randint', 'randperm', 'normal', 'multinomial'):
        setattr(torch, name, forbidden)
    torch.nn.Module.__init__ = forbidden
    torch.optim.Optimizer.__init__ = forbidden
    root = ROOTS['ns_amsgrad']
    binding = inspect('ns_amsgrad', root, 3)
    s = torch.load(root / 'provenance/provenance-state.pt', map_location='cpu', weights_only=True)
    t, models = s['trainer'], s['trainer']['models']
    gp = {k:v.double().clone().requires_grad_() for k,v in models['G'].items()}
    dp = {k:v.double() for k,v in models['D'].items()}
    prior = models['prior']['z'].double().clone().requires_grad_()
    request = json.loads((root / 'request.json').read_text())['request']
    means = torch.tensor(request['tasks']['ring16_acquisition']['execution']['host_definition']['means'], dtype=torch.float64)
    offsets = (torch.cat((2 * torch.eye(4), -2 * torch.eye(4))).double() if args.panel == 'axes'
        else torch.tensor(list(itertools.product((-1., 1.), repeat=4)), dtype=torch.float64))
    z = (prior[:, None, :] + float(models['prior']['sigma']) * offsets[None, :, :]).flatten(0, 1)
    x = mlp(z, gp)
    params = list(gp.values()) + [prior]
    metric = []
    for group in t['optimizers'][0]['param_groups']:
        for j in group['params']:
            state = t['optimizers'][0]['state'][j]
            metric.append(1 / ((state['max_exp_avg_sq'].double() / (1 - group['betas'][1] ** float(state['step']))).sqrt() + group['eps']))
    pg = dict(zip(gp, metric[:-1]))
    pprior = metric[-1][:, None, :].expand(len(prior), len(offsets), 4).reshape_as(z)
    a0 = F.linear(z, gp['net.0.weight'], gp['net.0.bias'])
    h0 = F.leaky_relu(a0, .2)
    a2 = F.linear(h0, gp['net.2.weight'], gp['net.2.bias'])
    h2 = F.leaky_relu(a2, .2)
    d0 = torch.where(a0 > 0, torch.ones_like(a0), torch.full_like(a0, .2))
    d2 = torch.where(a2 > 0, torch.ones_like(a2), torch.full_like(a2, .2))
    b2 = gp['net.4.weight'][None, :, :] * d2[:, None, :]
    b0 = torch.einsum('nci,ij->ncj', b2, gp['net.2.weight']) * d0[:, None, :]
    prior_jac = b0 @ gp['net.0.weight']
    coordinates_diag = h2.square() @ pg['net.4.weight'].T + pg['net.4.bias']
    coordinates_diag = coordinates_diag + (b2.square() * (h0.square() @ pg['net.2.weight'].T + pg['net.2.bias'])[:, None, :]).sum(2)
    coordinates_diag = coordinates_diag + (b0.square() * (z.square() @ pg['net.0.weight'].T + pg['net.0.bias'])[:, None, :]).sum(2)
    coordinates_diag = (coordinates_diag + (prior_jac.square() * pprior[:, None, :]).sum(2)).detach()
    diag = coordinates_diag.sum(1)
    trace_check = []
    for i in (0, len(x)//2, len(x)-1):
        actual = 0.
        for c in (0, 1):
            grad = torch.autograd.grad(x[i,c], params, retain_graph=True)
            actual += sum(float((g.square() * p).sum()) for g,p in zip(grad, metric))
        trace_check.append(abs(actual-float(diag[i])) / max(actual, 1e-30))
    assert max(trace_check) < 1e-10
    mean_diagonal = float(diag.sum()) / x.numel()
    damping = .01 * mean_diagonal
    def transpose(u):
        return [g.detach() for g in torch.autograd.grad(x, params, grad_outputs=u, retain_graph=True)]
    def forward(v):
        gd = dict(zip(gp, v[:-1]))
        dz = v[-1][:, None, :].expand(len(prior), len(offsets), 4).reshape_as(z)
        return tangent(z.detach(), gp, dz, gd).detach()
    def kernel(u):
        return forward([p*g for p,g in zip(metric, transpose(u))])
    adversarial = F.softplus(-mlp(x, dp, critic=True)).mean()
    grad_adv, = torch.autograd.grad(adversarial, x, retain_graph=True)
    grad_mmd = mmd_force(x.detach(), means)
    # lr is the archived scalar rate; no gain or task-dependent coefficient.
    rate = t['optimizers'][0]['param_groups'][0]['lr']
    rhs = -rate * len(x) * (grad_adv.detach() + grad_mmd)
    u = torch.zeros_like(rhs)
    residual = rhs.clone()
    preconditioner = coordinates_diag + damping
    conditioned = residual / preconditioner
    direction = conditioned.clone()
    rr = float((residual * conditioned).sum())
    norm_rhs = float(rhs.norm())
    history = []
    for iteration in range(args.max_iterations):
        ad = kernel(direction) + damping * direction
        denominator = float((direction * ad).sum())
        assert denominator > 0
        alpha = rr / denominator
        u = u + alpha * direction
        residual = residual - alpha * ad
        conditioned = residual / preconditioner
        new_rr = float((residual * conditioned).sum())
        history.append(float(residual.norm()) / norm_rhs)
        if history[-1] <= 1e-4:
            break
        direction = conditioned + (new_rr / rr) * direction
        rr = new_rr
    delta = [p*g for p,g in zip(metric, transpose(u))]
    gd = dict(zip(gp, delta[:-1]))
    dz = delta[-1][:, None, :].expand(len(prior), len(offsets), 4).reshape_as(z)
    gv = tangent(z.detach(), gp, torch.zeros_like(z), gd).detach()
    pv = tangent(z.detach(), gp, dz, {k:torch.zeros_like(v)for k,v in gp.items()}).detach()
    ordinary_delta = [-rate * g * p for g,p in zip(torch.autograd.grad(adversarial + (x*grad_mmd).sum(), params, retain_graph=True), metric)]
    ordinary = forward(ordinary_delta)
    weights = torch.full((len(x),), 1/len(x), dtype=torch.float64)
    result = {'schema_version': 1, 'scope': 'saved_endpoint_damped_output_gauss_newton_prerequisite',
        'qualification_input': False, 'training_updates_added': 0, 'random_draws_added': 0,
        'probe_sha256': sha(__file__), 'core_probe_sha256': sha(HERE.parent / 'r1r2-failure-diagnosis-20261022/saved_force_probe.py'),
        'mmd_probe_sha256': sha(HERE.parent / 'r1r2-failure-diagnosis-20261022/mmd_projection_probe.py'),
        'immutable_binding': binding['immutable_binding'], 'checkpoint_sha256': binding['checkpoint_sha256'],
        'panel': args.panel, 'points': len(x), 'MMD_widths': WIDTHS, 'MMD_weight': 1,
        'metric': 'archived AMS maximum-second-moment diagonal inverse, shared G and learned prior',
        'gauss_newton': {'relative_damping': .01, 'mean_kernel_diagonal': mean_diagonal,
            'absolute_damping': damping, 'maximum_CG_iterations': args.max_iterations, 'relative_tolerance': 1e-4,
            'solver': 'diagonal-preconditioned conjugate gradient, same damped normal equation',
            'iterations': len(history), 'residual_history': history,
            'analytic_trace_relative_error_max': max(trace_check),
            'fitted_velocity_relative_error': float((gv+pv-rhs).norm()) / float(rhs.norm()),
            'linear_solve_relative_residual': float((kernel(u)+damping*u-rhs).norm()) / float(rhs.norm())},
        'directions': {'desired_free_output': shape_drift(x, rhs, weights, means),
            'ordinary_AMS': shape_drift(x, ordinary, weights, means),
            'gauss_newton_G': shape_drift(x, gv, weights, means),
            'gauss_newton_prior': shape_drift(x, pv, weights, means),
            'gauss_newton_combined': shape_drift(x, gv+pv, weights, means)},
        'parameter_delta': stats(torch.cat([d.flatten()for d in delta])),
        'diagnostic_wall_seconds': time.monotonic()-started,
        'limits': ['Frozen endpoint metric and Jacobian; future AMS moments/critic refits not simulated.',
            'Deterministic analysis panels; not original stochastic128training minibatches.',
            'Hypothesis changes generator/prior update geometry and adds fixedMMD; no trained gate or global recipe pass.',
            'Original target Gaussian expectation uses means only to integrate analysis; runtime requires only real batches.']}
    assert random.getstate() == py and np.array_equal(np.random.get_state()[1], nr[1])
    assert np.random.get_state()[0] == nr[0] and np.random.get_state()[2:] == nr[2:]
    assert torch.equal(tr, torch.get_rng_state())
    result['ambient_rng_unchanged'] = True
    if args.output:
        def compact(v):
            if isinstance(v,dict):return {k:compact(x)for k,x in v.items()if k!='per_mode'}
            return v
        args.output.write_text(json.dumps(compact(result),indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    main()
