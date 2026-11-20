"""Focused current-API contracts for automatic feature-cell selection."""
import os
from pathlib import Path
from copy import deepcopy
import json
import runpy

import pytest
import torch

from particlegan.recipes import Recipe
from particlegan.training import GANTrainer
from particlegan.particle_prior import ParticlePrior
from particlegan.continuous import SequentialSettleTest
from particlegan.population_continuity import PopulationSequentialSettleTest

CONFIG = Path(os.environ.get('PARTICLEGAN_RA12_CONTRACT_CONFIG',
    str(Path(__file__).resolve().parents[1] / 'tests/fixtures/feature-auto-base.json')))
BASE = json.loads(CONFIG.read_text())
SEED = 314159


@pytest.fixture(autouse=True)
def isolated_cpu_contract():
    threads=torch.get_num_threads()
    deterministic=torch.are_deterministic_algorithms_enabled()
    warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    device=torch.get_default_device()
    dtype=torch.get_default_dtype()
    rng=torch.get_rng_state().clone()
    cuda_initialized=torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
    try:
        yield
        assert torch.cuda.is_initialized() == cuda_initialized
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        torch.set_rng_state(rng)


def make(*, backend='auto', n=1024, width=2, quarter=False, callback=False):
    # One fixed contract fixture. No quality scorer or seed search is involved.
    torch.manual_seed(SEED)
    G = torch.nn.Sequential(torch.nn.Linear(2,8), torch.nn.Tanh(), torch.nn.Linear(8,width))
    D = torch.nn.Sequential(torch.nn.Linear(width,8), torch.nn.Tanh(), torch.nn.Linear(8,1))
    prior = ParticlePrior(n,2,generator=torch.Generator().manual_seed(SEED+1))
    config = dict(BASE, z_dim=2, num_particles=n, batch_size=128, birth_death_backend=backend)
    if backend == 'knn':
        for key in ('birth_death_cells','birth_death_metric_rank','birth_death_chunk','birth_death_parent_policy'):
            config.pop(key)
    if quarter:
        config.update(lr=.0010625, prior_lr_mult=8., d_lr_mult=4.)
    trainer = GANTrainer(Recipe.from_dict(config),G,D,prior=prior,seed=SEED,serial_backward=True)
    if callback:
        trainer.policy.generation = lambda model,latent: model(latent)
    return trainer


def real(width=2):
    return torch.arange(128*width,dtype=torch.float32).reshape(128,width).sin()


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.device == b.device and a.dtype == b.dtype
        assert torch.allclose(a,b,rtol=0,atol=0,equal_nan=True)
    elif isinstance(a,dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x,y in zip(a,b): equal(x,y)
    elif isinstance(a,float) and a != a:
        assert b != b
    else:
        assert a == b


def numerical_state(trainer):
    state = trainer.state_dict()
    # Reaction wall timings are diagnostic; compare every semantic value.
    if 'eval_seconds' in state['birth_death']['last']:
        state['birth_death']['last'].pop('eval_seconds')
    return {key:state[key] for key in ('models','optimizers','controller','lr_settle',
                'birth_death','row_evidence','surprise','streams','output_noise','initial_lrs')}


def test_pending_checkpoint_and_no_update_resolution():
    a = make()
    saved = a.state_dict()
    assert saved['backend_selection']['actual_backend'] == 'pending'
    assert saved['backend_selection']['output_shape'] is None
    b = make()
    b.load_state_dict(saved)
    equal(saved,b.state_dict())
    rng = torch.get_rng_state().clone()
    streams = {name:getattr(a,name).get_state().clone() for name in a._STREAMS}
    a.policy._feature_selection.observe_shape(real())
    assert a.completed_steps == 0 and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(streams[name],getattr(a,name).get_state()) for name in streams)
    assert a.initial_lrs == [[.0010625,.0085,.0010625],[.00425]]
    assert isinstance(a.policy._table_tester(),PopulationSequentialSettleTest)


@pytest.mark.parametrize('n,width,callback,backend,reason',[
    (1024,2,False,'feature_cells','finite_resolution_and_complete_raw_moment_frame'),
    (1024,8,False,'feature_cells','finite_resolution_and_complete_raw_moment_frame'),
    (1024,9,False,'knn','raw_output_exceeds_complete_moment_frame'),
    (256,2,False,'knn','finite_resolution_infeasible'),
    (1024,2,True,'knn','caller_callbacks_own_representation'),
])
def test_structural_scope_and_preserved_base_owners(n,width,callback,backend,reason):
    a = make(n=n,width=width,callback=callback)
    a.policy._feature_selection.observe_shape(real(width))
    selection = a.state_dict()['backend_selection']
    assert selection['actual_backend'] == backend and selection['selection_reason'] == reason
    assert selection['output_shape'] == [width] and selection['raw_output_width'] == width
    assert a.initial_lrs[0][1] == .0085 and a.initial_lrs[1] == [.00425]
    assert a.initial_lrs[0][0] == (.0010625 if backend == 'feature_cells' else .00425)
    tester = a.policy._table_tester()
    assert isinstance(tester,PopulationSequentialSettleTest) == (backend == 'feature_cells')
    if backend == 'knn': assert isinstance(tester,SequentialSettleTest)


def test_explicit_feature_rates_remain_caller_owned():
    a = make(backend='feature_cells',quarter=True,width=9)
    a.policy._feature_selection.observe_shape(real(9))
    assert a.state_dict()['backend_selection']['selection_reason'] == 'explicit_feature_cells'
    assert a.state_dict()['backend_selection']['generator_noise_factor'] == 1.
    assert a.initial_lrs == [[.0010625,.0085,.0010625],[.00425]]


def test_explicit_feature_callback_fallback_keeps_the_current_reference_owner():
    from particlegan.birth_death import ParticleBirthDeath
    a=make(backend='feature_cells',callback=True)
    a.policy._feature_selection.observe_shape(real())
    selection=a.state_dict()['backend_selection']
    assert selection['actual_backend']=='knn'
    assert selection['selection_reason']=='caller_callbacks_own_representation'
    assert type(a.birth_death) is ParticleBirthDeath
    assert a.initial_lrs == [[.00425,.0085,.00425],[.00425]]


def test_auto_matches_explicit_feature_through_original_reaction():
    auto = make()
    forced = make(backend='feature_cells',quarter=True)
    for step in range(8):
        batch = real() + step * .001
        auto.step(batch)
        forced.step(batch)
    assert auto.birth_death.counters['evals'] == 1
    assert auto.birth_death.snapshot_serial == 1
    equal(numerical_state(auto),numerical_state(forced))
    assert auto.policy.surprise is not None


def test_reference_fallback_matches_current_explicit_knn():
    auto = make(width=9)
    reference = make(backend='knn',width=9)
    for step in range(2):
        auto.step(real(9))
        reference.step(real(9))
    equal(numerical_state(auto),numerical_state(reference))
    assert 'backend_selection' not in reference.state_dict()


def test_feature_checkpoint_replay_and_independent_served_law():
    a = make()
    for _ in range(8): a.step(real())
    saved = a.state_dict()
    b = make()
    b.load_state_dict(saved)
    equal(numerical_state(a),numerical_state(b))
    c = make()
    c.policy.load_state_dict(a.policy.state_dict())
    equal(a.policy.state_dict(),c.policy.state_dict())
    a.step(real()); b.step(real())
    equal(numerical_state(a),numerical_state(b))
    frozen = a.served_model()
    for noise in (False,True):
        g1=torch.Generator().manual_seed(SEED+100)
        g2=torch.Generator().manual_seed(SEED+100)
        equal(a.sample(64,generator=g1,output_noise=noise),
              frozen.sample(64,generator=g2,output_noise=noise))
    frozen_before=frozen.sample(64,generator=torch.Generator().manual_seed(SEED+101),output_noise=True)
    with torch.no_grad(): a.prior.z.add_(.2)
    a.birth_death.lineage.neighbors.fill_(-1)
    equal(frozen_before,frozen.sample(64,generator=torch.Generator().manual_seed(SEED+101),output_noise=True))
    assert frozen.backend_selection['actual_backend'] == 'feature_cells'


@pytest.mark.parametrize('mutation', ['schema','shape','factor','owner','rate','future_lease'])
def test_malformed_route_rejected_before_model_optimizer_mutation(mutation):
    a = make()
    a.step(real())
    valid = a.state_dict(); invalid=deepcopy(valid)
    if mutation == 'schema': invalid['backend_selection']['schema'] = True
    elif mutation == 'shape': invalid['backend_selection']['output_shape'] = [9]
    elif mutation == 'factor': invalid['backend_selection']['generator_noise_factor'] = .5
    elif mutation == 'owner': invalid['backend_selection']['rate_mapping'][0][1]['role'] = 'generator'
    elif mutation == 'rate': invalid['initial_lrs'][0][0] *= 2
    elif mutation == 'future_lease': invalid['birth_death']['paired_average']['step'] = 99
    with pytest.raises(ValueError): a.load_state_dict(invalid)
    equal(valid,a.state_dict())
    invalid_policy=deepcopy(a.policy.state_dict())
    invalid_policy['backend_selection']['schema'] = True
    with pytest.raises(ValueError): a.policy.load_state_dict(invalid_policy)
    equal(valid,a.state_dict())


def test_shape_invariants_are_checked_before_any_optimizer_update():
    a = make(width=9)
    saved=a.state_dict()
    models=deepcopy(saved['models'])
    with pytest.raises(ValueError,match='generator output shape'): a.step(real(2))
    equal(models,{name:getattr(a,name).state_dict() for name in models})
    assert a.completed_steps == 0 and not a.opt_g.state and not a.opt_d.state
    assert a.interrupted
    with pytest.raises(RuntimeError,match='interrupted'):a.state_dict()
    # The failed first cycle bound a real shape. Restore the unresolved
    # checkpoint into a fresh owner rather than rewriting that frozen binding.
    with pytest.raises(ValueError,match='output shape differs'):a.load_state_dict(saved)
    assert a.interrupted
    recovered=make(width=9)
    recovered.load_state_dict(saved)
    assert not recovered.interrupted
    equal(saved,recovered.state_dict())
    recovered.step(real(9))
    assert recovered.completed_steps == 1
    b=make();b.step(real())
    saved=b.state_dict()
    models=deepcopy(saved['models'])
    with pytest.raises(ValueError,match='real output shape'): b.step(real(9))
    equal(models,{name:getattr(b,name).state_dict() for name in models})
    assert b.completed_steps == 1
    assert b.interrupted
    with pytest.raises(RuntimeError,match='interrupted'):b.state_dict()
    b.load_state_dict(saved)
    assert not b.interrupted
    equal(saved,b.state_dict())


def test_separate_plain_adam_table_owner_preserves_rates_and_copy_states():
    from particlegan.policy import UpdatePolicy
    template=make()
    config=template.recipe
    G,D,prior=template.G,template.D,template.prior
    opt_g=config.make_generator_optimizer(G.parameters())
    opt_d=config.make_critic_optimizer(D,ema_critic=deepcopy(D))
    table_optimizer=torch.optim.Adam([prior.z],lr=.0085,amsgrad=True)
    p=UpdatePolicy(config,G,D,prior=prior,generator_optimizer=opt_g,
                   critic_optimizer=opt_d,table_optimizer=table_optimizer,seed=SEED)
    p._feature_selection.observe_shape(real())
    assert p.roles == [['generator','noise'],['critic'],['table']]
    assert p.initial_lrs == [[.0010625,.0010625],[.00425],[.0085]]
    prior.z.grad=torch.arange(prior.z.numel(),dtype=prior.z.dtype).reshape_as(prior.z).add_(1)
    table_optimizer.step()
    before=deepcopy(table_optimizer.state[prior.z])
    p.birth_death._move(p._feature_selection.facade,torch.tensor([1]),torch.tensor([2]))
    for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        equal(table_optimizer.state[prior.z][key][1],before[key][2])
    assert table_optimizer.state[prior.z]['step'] == before['step']
    assert p._feature_selection.facade.opt_g.latent_history is None


def test_routed_rows_keep_their_current_ownership_and_typed_replay():
    example=Path(__file__).resolve().parents[1] / 'examples/e22_routed_paired.py'
    api=runpy.run_path(str(example))
    overrides={'birth_death_backend':'auto','birth_death_cells':128,
               'birth_death_feature_scale':'std'}
    loop=api['make_loop'](recipe_overrides=overrides)
    owner=loop.policy.routed_control
    rates=deepcopy(loop.policy.initial_lrs)
    api['update'](loop)
    selection=loop.policy.state_dict()['backend_selection']
    assert selection['actual_backend'] == 'routed'
    assert selection['selection_reason'] == 'routed_rows_owns_controls'
    assert loop.policy.birth_death is owner
    assert loop.policy.initial_lrs == rates
    saved=api['checkpoint'](loop)
    resumed=api['make_loop'](recipe_overrides=overrides)
    api['restore'](resumed,saved)
    equal(saved,api['checkpoint'](resumed))
    equal(api['update'](loop),api['update'](resumed))


def test_leased_serving_bad_restore_does_not_mutate_parameters_or_versions():
    a=make()
    for step in range(8): a.step(real()+step*.001)
    # Prime a consistent semantic lease to exercise this rare serving branch.
    # The actual reaction chart, count budget and all stamp guards remain active.
    birth=a.birth_death
    stamp=deepcopy(birth.paired_average)
    assert stamp['chart_valid'] and stamp['duplicate_ok']
    stamp.update(finite_rows=1024,same_group_rows=1024,ema_eligible_rows=1024,
                 coherent_rows=1024,eligible=True)
    birth.paired_average=stamp
    birth.last['paired_average']=deepcopy(stamp)
    a._serve_apply()
    assert a._fast is not None and a._serve_settled()
    saved=a.state_dict()
    # Capture versions after the explicit state_dict swap, before rejected load.
    parameters=list(a._served_parameters())
    values=[p.detach().clone() for p in parameters]
    versions=[p._version for p in parameters]
    bad=deepcopy(saved)
    bad['backend_selection']['generator_noise_factor']=.5
    with pytest.raises(ValueError): a.load_state_dict(bad)
    assert a._fast is not None
    assert versions == [p._version for p in parameters]
    for value,parameter in zip(values,parameters): equal(value,parameter)
    policy_bad=deepcopy(a.policy.state_dict())
    values=[p.detach().clone() for p in parameters]
    versions=[p._version for p in parameters]
    policy_bad['backend_selection']['schema']=True
    with pytest.raises(ValueError): a.policy.load_state_dict(policy_bad)
    assert versions == [p._version for p in parameters]
    for value,parameter in zip(values,parameters): equal(value,parameter)
    a.birth_death.rows_since_eval=1024
    assert not a._serve_settled()


def test_reference_recipe_dictionary_matches_current_api_and_legacy_current_load():
    from particlegan.recipes import get_recipe
    reference=get_recipe('e22')
    assert all(not key.startswith('birth_death_') or key in
               ('birth_death_space','birth_death_isolation','birth_death_feature_scale')
               for key in reference.to_dict())
    # No new dictionary/schema keys appear on the unchanged reference path.
    a=make(backend='knn',n=32)
    a.step(real())
    saved=a.state_dict()
    assert 'backend_selection' not in saved
    b=make(backend='knn',n=32)
    b.load_state_dict(saved)
    equal(saved,b.state_dict())
