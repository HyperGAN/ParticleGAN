"""Read-only NS DDGAN critic-field endpoint diagnosis on fixed quadrature laws."""
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

ROOT=Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-repair-20261022/queue/r1r2-repair-20261022/32a62170283a41158410c13744709fb8')
SOURCE='e481b63224fc2d8cf77b136ab54f06421f7ac0967e47509a63f95cfa42733464'
INPUT_HASHES={
    'request.json':'4e4935055a92d2b4d7ecce4935ac91421a344c5d394b105f9f4a4599825a8c00',
    'raw-result.json':'68e3e7c6637bd4ef6ceefa151f7fff135aefd4c1c824640fb1c52eaa36a9c3f5',
    'api/receipt.json':'d453723aa6bb4d956b99e83530c5dd360de923eca7d70814909a38fdcdeabc59',
    'api/final-state.pt':'6d5abe85ce06fc3216150f66d2c0809e6a28533900196568a8225fa470093485',
}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gh(d,order):
    q,w=np.polynomial.hermite.hermgauss(order)
    ii=list(itertools.product(range(order),repeat=d))
    return (torch.tensor([[q[j]*np.sqrt(2)for j in row]for row in ii],dtype=torch.float64),
        torch.tensor([np.prod([w[j]/np.sqrt(np.pi)for j in row])for row in ii],dtype=torch.float64))


def mlp(x,p):
    for i in (0,2,4):
        x=F.linear(x,p[f'net.{i}.weight'],p[f'net.{i}.bias'])
        if i!=4:x=F.leaky_relu(x,.2)
    return x


def score(x,p,xy,step):
    return mlp(torch.cat((x,x.new_tensor(xy).expand(len(x),2),x.new_full((len(x),1),step/4)),1),p).squeeze(1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--order',type=int,default=2);args=parser.parse_args()
    started=time.monotonic();torch.set_num_threads(1)
    py,nr,tr=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    def forbidden(*a,**kw):raise AssertionError('Saved field probe forbids draws/models/optimizers')
    for name in ('rand','randn','randint','randperm','normal','multinomial'):setattr(torch,name,forbidden)
    torch.nn.Module.__init__=forbidden;torch.optim.Optimizer.__init__=forbidden
    for relative,digest in INPUT_HASHES.items():assert sha(ROOT/relative)==digest
    raw=json.loads((ROOT/'raw-result.json').read_text())
    api=json.loads((ROOT/'api/receipt.json').read_text())
    request=json.loads((ROOT/'request.json').read_text())['request']
    assert request['source']['digest']==SOURCE
    for relative,digest in request['source']['files'].items():assert sha(Path(request['source']['snapshot_path'])/relative)==digest
    assert sha(ROOT/'api/receipt.json')==raw['evidence']['api_receipt_sha256']
    for name,row in api['artifacts'].items():assert sha(ROOT/'api'/name)==row['sha256']
    state=torch.load(ROOT/'api/final-state.pt',map_location='cpu',weights_only=True)
    assert state['completed_steps']==2500
    assert state['case_id']=='api-model-ddgan-gaussian-chain'
    assert json.loads(json.dumps(state['recipe']))==api['recipe']
    assert state['recipe']['loss']=='non_saturating' and state['recipe']['reg_coeff']==.1
    assert state['recipe'].get('reg_real_weight',1.)==state['recipe'].get('reg_fake_weight',1.)==1.
    expected_keys=[f'net.{i}.{part}'for i in (0,2,4)for part in ('weight','bias')]
    for model,shapes in [('G',[(32,7),(32,),(32,32),(32,),(2,32),(2,)]),
                         ('D',[(32,5),(32,),(32,32),(32,),(1,32),(1,)])]:
        assert list(state['models'][model])==expected_keys
        assert [tuple(v.shape)for v in state['models'][model].values()]==shapes
    gp={k:v.double()for k,v in state['models']['G'].items()}
    dp={k:v.double().clone().requires_grad_()for k,v in state['models']['D'].items()}
    pz=state['models']['prior']['z'].double();sigma=float(state['models']['prior']['sigma'])
    assert pz.shape==(128,4) and sigma==.25
    assert state['models']['prior']['_extra_state']=={'sigma_rel':0.,'standardize':False}
    schedule={k:v.double()for k,v in state['models']['schedule'].items()}
    mu=torch.tensor([.6,-.35],dtype=torch.float64);var=torch.tensor([.45**2,.30**2],dtype=torch.float64)
    nodes,w=gh(4,args.order);eta,ew=gh(2,args.order);real_nodes,rw=gh(2,7)
    z=(pz[:,None,:]+sigma*nodes[None,:,:]).flatten(0,1);zw=w.repeat(128)/128
    panels=[];g_adv=[torch.zeros_like(p)for p in dp.values()];g_r1=[v.clone()for v in g_adv];g_r2=[v.clone()for v in g_adv]
    def grads(loss):
        values=torch.autograd.grad(loss,list(dp.values()),retain_graph=True,allow_unused=True)
        return [torch.zeros_like(p)if v is None else v.detach()for p,v in zip(dp.values(),values)]
    for step in range(1,5):
        for index,xy in enumerate(((-.5,.25),(.5,-.25))):
            print('Panel t'+str(step)+' q'+str(index),file=__import__('sys').stderr,flush=True)
            ab=schedule['ab'][step];A=schedule['A'][step];B=schedule['B'][step];noise=schedule['posterior_var'][step]
            cleanvar=1/(1/var+ab/(1-ab));cleanmu=cleanvar*(mu/var+ab.sqrt()*mu.new_tensor(xy)/(1-ab))
            realmu=A*cleanmu+B*mu.new_tensor(xy);realvar=A.square()*cleanvar+noise
            real=(realmu[None,:]+realvar.sqrt()[None,:]*real_nodes).requires_grad_()
            # G takes (xt,z,t), unlike D's (x_prev,xt,t).
            clean=mlp(torch.cat((z.new_tensor(xy).expand(len(z),2),z,z.new_full((len(z),1),step/4)),1),gp)
            fake=(A*clean[:,None,:]+B*mu.new_tensor(xy)[None,None,:]+noise.sqrt()*eta[None,:,:]).flatten(0,1).detach().requires_grad_()
            fw=(zw[:,None]*ew[None,:]).flatten()
            dr,df=score(real,dp,xy,step),score(fake,dp,xy,step)
            gr,=torch.autograd.grad(dr.sum(),real,create_graph=True);gf,=torch.autograd.grad(df.sum(),fake,create_graph=True)
            adv=(rw*F.softplus(-dr)).sum()+(fw*F.softplus(df)).sum()
            r1=.05*(rw*gr.square().sum(1)).sum();r2=.05*(fw*gf.square().sum(1)).sum()
            for out,g in zip(g_adv,grads(adv)):out+=g/8
            for out,g in zip(g_r1,grads(r1)):out+=g/8
            for out,g in zip(g_r2,grads(r2)):out+=g/8
            field=torch.sigmoid(-df)[:,None]*gf
            mean=(fw[:,None]*fake.detach()).sum(0);correction=realmu-mean
            alignment=(fw*(field*correction[None,:]).sum(1)).sum()
            ag=grads(alignment)
            cov=torch.einsum('n,ni,nj->ij',fw,fake.detach()-mean,fake.detach()-mean)
            eigen=torch.linalg.eigvalsh(torch.diag(realvar.rsqrt())@cov@torch.diag(realvar.rsqrt()))
            panels.append({'label':f't{step}_q{index}','initial_alignment':float(alignment.detach()),
                'normalized_mean':float((correction/realvar.sqrt()).norm()),'eigenvalues':eigen.tolist(),
                'clean_variance_share':(A.square()*cleanvar/realvar).tolist(),
                'critic_real_score':float((rw*dr.detach()).sum()),'critic_fake_score':float((fw*df.detach()).sum()),
                'alignment_gradient':ag})
    opt=state['optimizers'][1];assert opt['param_groups'][0]['params']==list(range(6))
    group=opt['param_groups'][0];lr=group['lr'];fresh=[a+b+c for a,b,c in zip(g_adv,g_r1,g_r2)]
    assert group['amsgrad'] and group['betas']==(.5,.99) and lr==.0085
    r1only=[a+b for a,b in zip(g_adv,g_r1)]
    directions={name:[]for name in ('stored_momentum','fresh_gradient','half_history_fresh_filter','R1_only_fresh')}
    for i,(g,r1)in enumerate(zip(fresh,r1only)):
        saved=opt['state'][i];n=float(saved['step']);b1,b2=group['betas']
        assert n==2500
        assert all(saved[k].shape==g.shape for k in ('exp_avg','exp_avg_sq','max_exp_avg_sq'))
        den=(saved['max_exp_avg_sq'].double()/(1-b2**n)).sqrt()+group['eps']
        m=saved['exp_avg'].double()/(1-b1**n)
        for name,value in [('stored_momentum',m),('fresh_gradient',g),('half_history_fresh_filter',.5*m+.5*g),('R1_only_fresh',r1)]:directions[name].append(-lr*value/den)
    for panel in panels:
        ag=panel.pop('alignment_gradient')
        panel['alignment_derivative']={name:sum(float((a*d).sum())for a,d in zip(ag,values))for name,values in directions.items()}
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],nr[1])
    assert np.random.get_state()[0]==nr[0] and np.random.get_state()[2:]==nr[2:]
    assert torch.equal(tr,torch.get_rng_state())
    norm=lambda values:float(sum(v.square().sum()for v in values).sqrt())
    result={'schema':1,'scope':'equal-eight-fixed-context-frozen-DDGAN-critic-field-quadrature',
        'training_updates_added':0,'random_draws_added':0,'ambient_rng_unchanged':True,
        'qualification_input':False,'order':args.order,'fake_latent_points':len(z),
        'source_digest':SOURCE,'source_files_verified':len(request['source']['files']),
        'driver_sha256':sha(__file__),'request_sha256':sha(ROOT/'request.json'),
        'raw_result_sha256':sha(ROOT/'raw-result.json'),
        'api_receipt_sha256':sha(ROOT/'api/receipt.json'),'checkpoint_sha256':sha(ROOT/'api/final-state.pt'),
        'runtime':{'torch':torch.__version__,'device':'cpu','dtype':'float64','threads':1},
        'gradient_norms':{'adversarial':norm(g_adv),'R1':norm(g_r1),'R2':norm(g_r2),'combined':norm(fresh)},
        'panels':panels,'wall_seconds':time.monotonic()-started,
        'limits':['Equal fixed-panel cohort is not the training context marginal.',
            'Alignment derivatives are frozen generator/output-space diagnostics, not G parameter updates.',
            'D directions use a fixed historical AMS denominator; no new optimizer moments or training steps.',
            'Stored momentum is the previous update history; half_history_fresh_filter is a diagnostic mixture, not the exact next Adam update.',
            'Fresh conditional target moments are analysis-only Gaussian laws; no analytic training targets are proposed.']}
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
