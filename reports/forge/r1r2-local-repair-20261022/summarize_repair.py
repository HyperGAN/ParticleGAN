"""Compact source-bound saved-state conclusions, never qualification inputs."""
import json
from pathlib import Path

from local_repair_probe import sha

HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-local-repair-20261022')
KEYS=('center_squared_error_drift','original_covariance_error_drift','radial_variance_drift','tangent_variance_drift')


def means(row):
    return{k:row['mean_over_modes'][k]for k in KEYS}


def main():
    receipts={}
    def read(name,filename,driver):
        path=RAW/filename
        d=json.loads(path.read_text())
        assert d['probe_sha256']==sha(HERE/driver),name
        assert d['training_updates_added']==0 and not d['qualification_input']
        assert d['ambient_rng_unchanged']
        receipts[name]={'path':str(path),'sha256':sha(path),'driver_sha256':d['probe_sha256'],
            'diagnostic_wall_seconds':d['diagnostic_wall_seconds']}
        return d
    negative={}
    for mechanism in ('local','conditional','score'):
        negative[mechanism]={}
        for panel in ('axes','corners'):
            d=read(mechanism+'_'+panel,'final-'+mechanism+'-'+panel+'.stdout.json','local_repair_probe.py')
            negative[mechanism][panel]=means(d['directions']['combined']['combined'])
    population={}
    for panel in ('axes','corners'):
        d=read('GN_population_'+panel,'gn-force-pcg1024-'+panel+'.stdout.json','gn_force_probe.py')
        assert d['gauss_newton']['linear_solve_relative_residual']<=1e-4
        population[panel]={'ordinary_AMS':means(d['directions']['ordinary_AMS']),
            'GN':means(d['directions']['gauss_newton_combined']),
            'iterations':d['gauss_newton']['iterations'],
            'relative_residual':d['gauss_newton']['linear_solve_relative_residual']}
    empirical={}
    replay=None
    for panel in ('axes','corners'):
        d=read('GN_actual128_'+panel,'gn-batch-'+panel+'.stdout.json','gn_batch_probe.py')
        assert d['all_solves_accepted'] and d['mmd_gradient_autograd_max_abs_error']<1e-10
        assert d['replay_probe_sha256']==sha(HERE/'replay_training_batches.py')
        for record in d['numerical_solves']:
            assert record['relative_residual']<=1e-4 and record['independent_diagonal_relative_error']<1e-10
        if replay is not None:assert replay==d['historical_stream_replay']
        replay=d['historical_stream_replay']
        empirical[panel]={'directions':{k:means(v['combined'])for k,v in d['directions'].items()},
            'iterations_min':min(r['iterations']for r in d['numerical_solves']),
            'iterations_max':max(r['iterations']for r in d['numerical_solves']),
            'maximum_relative_residual':max(r['relative_residual']for r in d['numerical_solves'])}
    result={'schema_version':1,'scope':'saved_only_repair_prerequisite_summary',
        'qualification_input':False,'training_updates_added':0,'new_independent_random_streams':0,
        'diagnostic_runtime':{'device':'cpu','floating_dtype':'torch.float64','threads':1,
            'historical_replay':'CPUtarget and CUDAoriginalpriorindex/Gaussian generator clones, not a model run'},
        'raw_receipts':receipts,'negative_auxiliaries':negative,
        'GN_population_proxy':population,'GN_actual128_empirical_sensitivity':empirical,
        'historical_draw_replay':replay,
        'source_pins':{p.name:sha(p)for p in HERE.glob('*.py')},
        'conclusion':{'rejected_saved_prerequisites':['relative local feature means weight1: centering regression',
            'conditional first/fullcovariance moments: radial contraction and panel-sensitive centering',
            'KDEtarget-minus-fake score transport: covariance/tangent regression after projection'],
            'supported_but_unqualified_hypothesis':'FixedMMD plus damped output-tangent G/prior update; ordinary critic/R1R2 preserved',
            'actual128_caveat':'MMD+GN corrects radial/center and weakly covariance but slightly expands alreadywide tangent; NS-onlyGN contracts radial',
            'runtime_caveat':'Accurate archivedfloat64 batch solves need109-134iterations; current genericfloat32CUDA solver runtime/accuracy do not establish a complete affordable global recipe',
            'next':'Do not admit a trainedGNcomparison from this proxy alone; resolve accurate bounded runtime and allhost contracts, or select a separately justified mechanism.'}}
    target=HERE/'repair-prerequisites.json'
    target.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','output':str(target),'sha256':sha(target)},sort_keys=True))


if __name__=='__main__':main()
