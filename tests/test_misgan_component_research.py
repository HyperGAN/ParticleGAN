from copy import deepcopy

import pytest
import torch

from benchmarks.toy_audit.api_model_misgan import (Problem, MisGANFixture, list_cases, score, CENTERS, SIGMA)
from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.state import state_digest
from particlegan.masked_mixture import masked_mixture_nll, NeuralComponentImputer, masked_quantile_transport_loss


def oracle_fixture():
    p=Problem(); r=torch.Generator().manual_seed(913)
    centers=torch.tensor(CENTERS)
    x2=centers[torch.randint(4,(4096,),generator=r)]+SIGMA*torch.randn(4096,2,generator=r)
    generated=p.standardize(x2@p.A.T)
    from benchmarks.toy_audit.api_sources.misgan import sample_masks
    masks=sample_masks('block',4096,r)
    x,m=p.ambiguous_queries()
    raw_sum=(x[:,7]*p.std[7]+p.mean[7])*8**.5
    branches=torch.randint(2,(256,16),generator=r)
    along=(2*branches-1)*2.**.5+SIGMA*torch.randn(256,16,generator=r)
    x2a=torch.stack((raw_sum[None]/2+along/2.**.5,raw_sum[None]/2-along/2.**.5),-1)
    ambiguous=p.standardize(x2a@p.A.T)
    ambiguous=x*m+ambiguous*(1-m)
    # Genuine analytic conditional draws, including random ambiguous branches;
    # the positive control never copies a hidden complete test row.
    prediction=torch.zeros_like(p.x_test)
    for pattern in p.m_test.unique(dim=0):
        rows=(p.m_test==pattern).all(1); sensors=pattern.bool()
        raw=p.x_test[rows][:,sensors]*p.std[sensors]+p.mean[sensors]
        if int(sensors.sum())>1:
            z=torch.linalg.lstsq(p.A[sensors],raw.T).solution.T
        else:
            total=raw[:,0]*8**.5
            center_sums=centers.sum(1)
            weights=(-.5*(total[:,None]-center_sums).square()/(2*SIGMA**2)).softmax(1)
            ids=torch.multinomial(weights,1,generator=r).flatten()
            along=(centers[ids,0]-centers[ids,1])/2.**.5+SIGMA*torch.randn(len(ids),generator=r)
            z=torch.stack((total/2+along/2.**.5,total/2-along/2.**.5),1)
        prediction[rows]=p.standardize(z@p.A.T)
    prediction=p.x_test*p.m_test+prediction*(1-p.m_test)
    return p,generated,masks,prediction,ambiguous


def test_scorer_oracle_and_uncertainty_controls():
    args=oracle_fixture()
    assert score(*args)['passed']
    p,g,m,pred,amb=args
    for broken in (amb.mean(0,keepdim=True).expand_as(amb),amb[:1].expand_as(amb),amb.flip(2)):
        assert not score(p,g,m,pred,broken)['passed']
    wrong_mass=amb.clone();wrong_mass[:230]=amb[0]
    assert not score(p,g,m,pred,wrong_mass)['passed']
    assert not score(p,g[:1].expand_as(g),m,pred,amb)['passed']
    assert not score(p,g,m[:1].expand_as(m),pred,amb)['passed']


def test_observed_likelihood_ignores_missing_values_and_detaches_teacher():
    x=torch.tensor([[1.,float('nan')],[1.,999.]],requires_grad=True)
    mask=torch.tensor([[1.,0.],[1.,0.]])
    centers=torch.tensor([[0.,2.],[1.,3.]],requires_grad=True)
    loss=masked_mixture_nll(x,mask,centers,sigma=.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
    assert torch.equal(x.grad[:,1],torch.zeros(2))
    assert torch.equal(centers.grad[:,1],torch.zeros(2))
    net=NeuralComponentImputer(2,2)
    teacher=torch.randn(4,2,requires_grad=True)
    net.synthetic_loss(teacher,torch.ones(4,2),torch.tensor([0,1,0,1])).backward()
    assert teacher.grad is None
    assert any(p.grad is not None for p in net.parameters())


@pytest.mark.parametrize('candidate_name',['bispf-misgan-amortized-v1','bispf-misgan-transport-settle-v1'])
@reproducible_execution
def test_full_checkpoint_and_matched_batches(candidate_name):
    torch.set_num_threads(1)
    case=list_cases()[0]
    a=MisGANFixture(case,max_steps=3)
    b=MisGANFixture(case,max_steps=3,recipe_name=candidate_name)
    assert state_digest(a.G.state_dict())==state_digest(b.G.state_dict())
    assert state_digest(a.prior.state_dict())==state_digest(b.prior.state_dict())
    assert not hasattr(a.problem,'x_train_full')
    a.step();b.step()
    assert a.batch_sequence_sha256==b.batch_sequence_sha256
    before=state_digest(b.state_dict());b.observe(n=128)
    assert state_digest(b.state_dict())==before
    state=deepcopy(b.state_dict())
    b.step();expected=state_digest(b.state_dict())
    c=MisGANFixture(case,max_steps=3,recipe_name=candidate_name)
    c.load_state_dict(state);c.step()
    assert state_digest(c.state_dict())==expected


def test_nonconsumer_refuses_recipe_objectives():
    from particlegan import GANTrainer,get_recipe
    from torch import nn
    with pytest.raises(ValueError,match='incomplete-data host'):
        GANTrainer(get_recipe('ka2',masked_mixture_weight=1.),nn.Linear(2,2),nn.Linear(2,1))
    from benchmarks.toy_audit.api_contract import build, discover
    from benchmarks.toy_audit.api_model_ae import task_recipe
    case=discover()['api-model-ae-reconstruction']
    with pytest.raises(ValueError,match='explicit MisGAN component consumer'):
        build(case,max_steps=1,recipe=task_recipe(case).replace(masked_transport_weight=1.))


def test_masked_transport_equal_mass_and_no_missing_target_access():
    real=torch.tensor([[-1.,float('nan')],[-1.,float('nan')],[1.,float('nan')],[1.,float('nan')]])
    mask=torch.tensor([[1.,0.]]*4)
    exact=torch.tensor([[-1.,9.],[-1.,9.],[1.,9.],[1.,9.]],requires_grad=True)
    assert masked_quantile_transport_loss(exact,real,mask)==0
    wrong=torch.tensor([[-1.,99.],[-1.,99.],[-1.,99.],[1.,99.]],requires_grad=True)
    loss=masked_quantile_transport_loss(wrong,real,mask);loss.backward()
    assert loss>0 and torch.isfinite(loss)
    assert torch.equal(wrong.grad[:,1],torch.zeros(4))
    assert wrong.grad[2,0]<0  # Excess left mass is driven across the gap.
