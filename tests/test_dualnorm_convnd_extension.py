"""Ten NEW CPU cases for the explicit unreleased ConvNd layout extension.

The independent reference indexes original native kernel offsets directly and
writes its own SVD law. No model weights, CUDA, prior native tests or downloads.
"""
from copy import deepcopy
from itertools import product
import math
import pytest
import torch
from torch import nn
from particlegan.optim import dualnorm as d

CASES=[(nn.Conv1d,(3,),False),(nn.ConvTranspose1d,(3,),True),
       (nn.Conv2d,(2,3),False),(nn.ConvTranspose2d,(2,3),True),
       (nn.Conv3d,(2,1,3),False),(nn.ConvTranspose3d,(2,1,3),True)]


def layer(cls,kernel,*,groups=2):
    # Isolate constructor RNG; no initializer policy or pretrained model here.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1701)
        value=cls(4,6,kernel,groups=groups,bias=False,dtype=torch.float64)
    with torch.no_grad():value.weight.copy_(torch.linspace(-.7,.9,value.weight.numel()).reshape_as(value.weight))
    return value


def optimizer(value,*,momentum=0.):
    groups=d.convolution_parameter_groups([dict(params=[value.weight],role='critic')],[value],family='dualnorm')
    assert groups[0]['params'][0] is value.weight
    return d.NormalizedOptimizer(groups,lr=.03,family='dualnorm',momentum=momentum,smoothing=.001,
                                 convolution='per_offset',convolution_rule='polar',svd_backend='native')


def gradient(value,offset=0.):
    x=torch.arange(value.weight.numel(),dtype=torch.float64).reshape_as(value.weight)
    return torch.sin(x*.37+offset)+torch.cos(x*.19-offset)


def reference(weight,grad,value,transposed):
    """No optimizer layout helper or polar helper is used by this reference."""
    result=weight.clone();channels=value.in_channels if transposed else value.out_channels
    block=channels//value.groups
    for group in range(value.groups):
        for offset in product(*(range(size)for size in value.kernel_size)):
            index=(slice(group*block,(group+1)*block),slice(None),*offset)
            matrix=grad[index].T if transposed else grad[index]
            if matrix.norm()<1e-8:continue
            left,singular,right=torch.linalg.svd(matrix,full_matrices=False)
            cutoff=max(matrix.shape)*torch.finfo(matrix.dtype).eps*singular[0]
            spectral=singular/torch.sqrt(singular.square()+.001**2)
            spectral=torch.where(singular>cutoff,spectral,torch.zeros_like(spectral))
            delta=(left*spectral)@right
            delta=delta.T if transposed else delta
            result[index].add_(delta,alpha=-.03*math.sqrt(value.out_channels/value.in_channels)/math.prod(value.kernel_size))
    return result


@pytest.mark.parametrize('cls,kernel,transposed',CASES)
def test_native_group_offset_update_matches_independent_reference_and_storage(cls,kernel,transposed):
    value=layer(cls,kernel);original=value.weight;before=value.weight.detach().clone();grad=gradient(value)
    # One offset stays fixed, proving offsets are not flattened into channels.
    zero=(slice(None),slice(None),*(0 for _ in kernel));grad[zero]=0
    expected=reference(before,grad,value,transposed);opt=optimizer(value)
    metadata=opt.param_groups[0]['dualnorm_convolution'];dimension=len(kernel)
    assert metadata['layout']==(f'conv_transpose{dimension}d_in_out'if transposed else f'conv{dimension}d_out_in')
    assert metadata['kernel_size']==list(kernel)and metadata['groups']==2
    assert metadata['update_version']==('per_offset_polar_v1'if dimension==2 else'per_offset_polar_v1_nd_layout_v1')
    batch=d._convolution_batch(grad,metadata)
    assert torch.equal(d._convolution_storage(batch,metadata).reshape_as(grad),grad)
    independent=[];width=(value.in_channels if transposed else value.out_channels)//value.groups
    for group in range(value.groups):
        for offset in product(*(range(size)for size in kernel)):
            item=grad[(slice(group*width,(group+1)*width),slice(None),*offset)]
            independent.append(item.T if transposed else item)
    actual=list(d._convolution_matrices(grad,metadata))
    assert len(actual)==len(independent)==value.groups*math.prod(kernel)
    assert all(torch.equal(a,b)for a,b in zip(actual,independent))
    # Basic indexed matrices retain writable native parameter storage.
    view=next(d._convolution_matrices(value.weight,metadata))
    with torch.no_grad():view[0,0]=123.;assert value.weight[(0,0,*(0 for _ in kernel))]==123.;value.weight.copy_(before)
    value.weight.grad=grad;opt.step()
    assert value.weight is original
    assert torch.allclose(value.weight,expected,rtol=2e-13,atol=2e-13)
    assert torch.equal(value.weight[zero],before[zero])


def test_zero_current_gradient_preserves_weight_despite_retained_momentum():
    value=layer(nn.Conv3d,(2,1,3));opt=optimizer(value,momentum=.5)
    value.weight.grad=gradient(value);opt.step();before=value.weight.detach().clone()
    assert opt.state[value.weight]['momentum_buffer'].count_nonzero()
    value.weight.grad=torch.zeros_like(value.weight);opt.step()
    assert torch.equal(value.weight,before)and opt.state[value.weight]['step']==2


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a)is type(b)and len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def test_new_3d_exact_state_continuation_retains_parameter_identity_and_metadata():
    value=layer(nn.ConvTranspose3d,(2,1,3));opt=optimizer(value,momentum=.5)
    value.weight.grad=gradient(value);opt.step();saved_weight=value.weight.detach().clone();saved=deepcopy(opt.state_dict())
    value.weight.grad=gradient(value,.9);opt.step();expected=value.weight.detach().clone();state=deepcopy(opt.state_dict())
    restored=layer(nn.ConvTranspose3d,(2,1,3));identity=id(restored.weight)
    with torch.no_grad():restored.weight.copy_(saved_weight)
    resumed=optimizer(restored,momentum=.5);resumed.load_state_dict(saved)
    restored.weight.grad=gradient(restored,.9);resumed.step()
    assert id(restored.weight)==identity and torch.equal(restored.weight,expected);equal(state,resumed.state_dict())


def test_unchanged_2d_metadata_legacy_order_and_batched_inverse():
    # This new regression covers the generalized layout path, not a replayed
    # historical model/native proof or serialized old training cohort.
    for cls,transposed in[(nn.Conv2d,False),(nn.ConvTranspose2d,True)]:
        value=layer(cls,(2,3),groups=1);groups=d.convolution_parameter_groups([dict(params=[value.weight],role='critic')],[value],family='dualnorm',convolution_rule='quintic')
        m=groups[0]['dualnorm_convolution']
        assert m==dict(update_version='per_offset_quintic_hybrid_v1',layout='conv_transpose2d_in_out'if transposed else'conv2d_out_in',groups=1,in_channels=4,out_channels=6,kernel_size=[2,3])
        storage=value.weight.reshape(1,*(value.weight.shape));old=storage.permute(0,3,4,2,1)if transposed else storage.permute(0,3,4,1,2)
        assert torch.equal(d._convolution_batch(value.weight,m),old)
        old_inverse=old.permute(0,4,3,1,2)if transposed else old.permute(0,3,4,1,2)
        assert torch.equal(d._convolution_storage(old,m),old_inverse)


def test_explicit_nd_metadata_mismatch_is_rejected_without_parameter_mutation():
    value=layer(nn.Conv3d,(2,1,3));opt=optimizer(value);group=opt.param_groups[0];before=value.weight.detach().clone()
    for change in[dict(layout=[]),dict(layout='conv2d_out_in'),dict(update_version='per_offset_polar_v1'),dict(kernel_size=[2,3]),dict(groups=3)]:
        invalid={**group,'dualnorm_convolution':{**group['dualnorm_convolution'],**change}}
        with pytest.raises(ValueError):d._validate_convolution_group(invalid)
    assert torch.equal(value.weight,before)
