"""Frozen endpoint GN sensitivity on exact original128-example batch draws."""
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
from saved_force_probe import ROOTS, inspect, mlp, sha, shape_drift, tangent
from mmd_projection_probe import WIDTHS
from replay_training_batches import recover


def empirical_mmd_gradient(x, y):
    n = len(x)
    xx = x[:, None, :] - x[None, :, :]
    xy = x[:, None, :] - y[None, :, :]
    out = torch.zeros_like(x)
    for width in WIDTHS:
        s2 = width**2
        q = torch.exp(-xx.square().sum(2) / (2*s2))
        p = torch.exp(-xy.square().sum(2) / (2*s2))
        out += (-2 * (q[:, :, None]*xx).sum(1) / (n*(n-1)*s2)
            + 2 * (p[:, :, None]*xy).mean(1) / (n*s2)) / len(WIDTHS)
    return out


def check_mmd_gradient():
    x=(torch.arange(14,dtype=torch.float64).reshape(7,2)/7-1).requires_grad_()
    y=torch.arange(22,dtype=torch.float64).reshape(11,2)/9-.7
    loss=x.sum()*0
    for width in WIDTHS:
        k=torch.exp(-(x[:,None,:]-x[None,:,:]).square().sum(2)/(2*width**2))
        cross=torch.exp(-(x[:,None,:]-y[None,:,:]).square().sum(2)/(2*width**2))
        loss+=( (k.sum()-k.diag().sum())/(len(x)*(len(x)-1))-2*cross.mean())/len(WIDTHS)
    actual,=torch.autograd.grad(loss,x)
    error=float((actual-empirical_mmd_gradient(x.detach(),y)).abs().max())
    assert error<1e-10
    return error


def diagonal(z, gp, metric, indices):
    pg = dict(zip(gp, metric[:-1]))
    a0 = F.linear(z, gp['net.0.weight'], gp['net.0.bias'])
    h0 = F.leaky_relu(a0, .2)
    a2 = F.linear(h0, gp['net.2.weight'], gp['net.2.bias'])
    h2 = F.leaky_relu(a2, .2)
    d0 = torch.where(a0>0, torch.ones_like(a0), torch.full_like(a0,.2))
    d2 = torch.where(a2>0, torch.ones_like(a2), torch.full_like(a2,.2))
    b2 = gp['net.4.weight'][None,:,:] * d2[:,None,:]
    b0 = torch.einsum('nci,ij->ncj',b2,gp['net.2.weight']) * d0[:,None,:]
    jprior = b0 @ gp['net.0.weight']
    out = h2.square() @ pg['net.4.weight'].T + pg['net.4.bias']
    out += (b2.square() * (h0.square() @ pg['net.2.weight'].T + pg['net.2.bias'])[:,None,:]).sum(2)
    out += (b0.square() * (z.square() @ pg['net.0.weight'].T + pg['net.0.bias'])[:,None,:]).sum(2)
    return (out + (jprior.square()*metric[-1][indices,None,:]).sum(2)).detach()


def solve(x, z, gp, prior, indices, metric, output_force, max_iterations=1024):
    params = list(gp.values())+[prior]
    diag = diagonal(z,gp,metric,indices)
    actual_diagonal=0.
    for c in (0,1):
        gradients=torch.autograd.grad(x[0,c],params,retain_graph=True)
        actual_diagonal+=sum(float((v.square()*p).sum())for v,p in zip(gradients,metric))
    diagonal_error=abs(actual_diagonal-float(diag[0].sum()))/actual_diagonal
    assert diagonal_error<1e-10
    damping = .01 * float(diag.mean())
    def jt(u):
        return [v.detach()for v in torch.autograd.grad(x,params,grad_outputs=u,retain_graph=True)]
    def j(v):
        return tangent(z.detach(),gp,v[-1][indices],dict(zip(gp,v[:-1]))).detach()
    def kernel(u):
        return j([p*v for p,v in zip(metric,jt(u))])
    rhs = .0085 * output_force
    u = torch.zeros_like(rhs)
    r = rhs.clone()
    preconditioner = diag+damping
    conditioned = r/preconditioner
    direction = conditioned.clone()
    rr = float((r*conditioned).sum())
    norm_rhs = float(rhs.norm())
    for iteration in range(max_iterations):
        a = kernel(direction)+damping*direction
        denominator = float((direction*a).sum())
        assert denominator>0
        alpha = rr/denominator
        u += alpha*direction
        r -= alpha*a
        if float(r.norm())/norm_rhs<=1e-4:break
        conditioned = r/preconditioner
        new_rr = float((r*conditioned).sum())
        direction = conditioned+(new_rr/rr)*direction
        rr = new_rr
    actual_residual = float((kernel(u)+damping*u-rhs).norm())/norm_rhs
    return [p*v for p,v in zip(metric,jt(u))], {'iterations':iteration+1,
        'relative_residual':actual_residual,'numerically_accepted':actual_residual<=1e-4,
        'independent_diagonal_relative_error':diagonal_error,
        'damping':damping,'mean_kernel_diagonal':float(diag.mean())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--panel',choices=('axes','corners'),default='axes')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    started=time.monotonic()
    py,nr,tr=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    cuda_before=torch.cuda.get_rng_state_all()
    root=ROOTS['ns_amsgrad']
    binding=inspect('ns_amsgrad',root,3)
    batches,replay=recover(root,keep=16)
    def forbidden(*a,**kw):raise AssertionError('No new draws after verified historical replay')
    for name in ('rand','randn','randint','randperm','normal','multinomial'):setattr(torch,name,forbidden)
    torch.nn.Module.__init__=forbidden
    torch.optim.Optimizer.__init__=forbidden
    s=torch.load(root/'provenance/provenance-state.pt',map_location='cpu',weights_only=True)
    t,models=s['trainer'],s['trainer']['models']
    gp={k:v.double().clone().requires_grad_()for k,v in models['G'].items()}
    dp={k:v.double()for k,v in models['D'].items()}
    prior=models['prior']['z'].double().clone().requires_grad_()
    params=list(gp.values())+[prior]
    metric=[]
    for group in t['optimizers'][0]['param_groups']:
        for index in group['params']:
            state=t['optimizers'][0]['state'][index]
            metric.append(1/((state['max_exp_avg_sq'].double()/(1-group['betas'][1]**float(state['step']))).sqrt()+group['eps']))
    assert all(g['lr']==.0085 for g in t['optimizers'][0]['param_groups'])
    totals={name:[torch.zeros_like(p)for p in params]for name in ('ns_only_GN','ns_mmd_GN','ns_only_AMS','ns_mmd_AMS')}
    observations=[]
    for batch in batches:
        print('Historical batch '+str(batch['step']),file=sys.stderr,flush=True)
        indices=batch['indices']
        z=prior[indices]+float(models['prior']['sigma'])*batch['noise'].double()
        x=mlp(z,gp)
        loss=F.softplus(-mlp(x,dp,critic=True)).mean()
        adv,=torch.autograd.grad(loss,x,retain_graph=True)
        mmd=empirical_mmd_gradient(x.detach(),batch['real'].double())
        for name,grad in [('ns_only',adv.detach()),('ns_mmd',adv.detach()+mmd)]:
            force=-len(x)*grad
            delta,receipt=solve(x,z,gp,prior,indices,metric,force)
            for out,v in zip(totals[name+'_GN'],delta):out+=v/len(batches)
            raw_grads=torch.autograd.grad(x,params,grad_outputs=grad,retain_graph=True)
            for out,p,g in zip(totals[name+'_AMS'],metric,raw_grads):out-=.0085*p*g/len(batches)
            observations.append({'step':batch['step'],'objective':name,**receipt})
    request=json.loads((root/'request.json').read_text())['request']
    means=torch.tensor(request['tasks']['ring16_acquisition']['execution']['host_definition']['means'],dtype=torch.float64)
    offsets=(torch.cat((2*torch.eye(4),-2*torch.eye(4))).double()if args.panel=='axes'
        else torch.tensor(list(itertools.product((-1.,1.),repeat=4)),dtype=torch.float64))
    z=(prior[:,None,:]+float(models['prior']['sigma'])*offsets[None,:,:]).flatten(0,1)
    x=mlp(z,gp)
    weights=torch.full((len(x),),1/len(x),dtype=torch.float64)
    directions={}
    for name,delta in totals.items():
        dz=delta[-1][:,None,:].expand(len(prior),len(offsets),4).reshape_as(z)
        gv=tangent(z.detach(),gp,torch.zeros_like(z),dict(zip(gp,delta[:-1]))).detach()
        pv=tangent(z.detach(),gp,dz,{k:torch.zeros_like(v)for k,v in gp.items()}).detach()
        directions[name]={'G':shape_drift(x,gv,weights,means),'prior':shape_drift(x,pv,weights,means),
            'combined':shape_drift(x,gv+pv,weights,means)}
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],nr[1])
    assert np.random.get_state()[0]==nr[0] and np.random.get_state()[2:]==nr[2:]
    assert torch.equal(tr,torch.get_rng_state())
    assert all(torch.equal(a,b)for a,b in zip(cuda_before,torch.cuda.get_rng_state_all()))
    result={'schema_version':1,'scope':'frozen_endpoint_actual128batch_sensitivity',
        'qualification_input':False,'training_updates_added':0,'new_independent_random_streams':0,
        'ambient_rng_unchanged':True,'historical_stream_replay':replay,
        'probe_sha256':sha(__file__),'replay_probe_sha256':sha(HERE/'replay_training_batches.py'),
        'core_probe_sha256':sha(HERE.parent/'r1r2-failure-diagnosis-20261022/saved_force_probe.py'),
        'mmd_gradient_autograd_max_abs_error':check_mmd_gradient(),
        'immutable_binding':binding['immutable_binding'],'checkpoint_sha256':binding['checkpoint_sha256'],
        'panel':args.panel,'directions':directions,'numerical_solves':observations,
        'all_solves_accepted':all(r['numerically_accepted']for r in observations),
        'MMD_widths':WIDTHS,'MMD_coefficient':1,'relative_damping':.01,
        'CG_max_iterations':1024,'CG_tolerance':1e-4,'diagnostic_wall_seconds':time.monotonic()-started,
        'limits':['All16suffix draw identities evaluated at one frozen endpoint, not historical modelstates.',
            'Mean of16independent hypothetical parameter directions; no parameters or optimizerstates updated.',
            'Original empirical128real rows and original128latent indices/noise; MMD target is empirical, no analyticmeans used in force.',
            'Fixed savedAMS metric, no newmoment or futurecritic refitting; no trained gate or qualification.']}
    if args.output:
        def compact(v):
            if isinstance(v,dict):return{k:compact(x)for k,x in v.items()if k!='per_mode'}
            return v
        args.output.write_text(json.dumps(compact(result),indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
