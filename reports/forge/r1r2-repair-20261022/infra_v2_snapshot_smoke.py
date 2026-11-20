import json, sys, time, importlib
from pathlib import Path
import torch
from experiments.forge.model_toy_adapter import run
from experiments.forge.contracts import file_hash, atomic_json
from experiments.forge.sources import verify_snapshot
from experiments.forge.views import grade_result

root=Path.cwd().resolve()
raw=Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-repair-20261022/infra-v2-smoke')
request_path=raw.parent/'infra-v2-smoke-request.json'
request=json.loads(request_path.read_text())
verify_snapshot(root,request['source'])
assert request['protocol']['seed']==0
assert request['candidate']['id']=='r1r2-repair-control-20261022'
recipe=request['candidate']['resolved_recipe']
assert recipe['loss']=='non_saturating' and recipe['amsgrad'] is True and recipe['optimizer_family']=='adam'
assert recipe['lr']==.0085 and recipe['betas']==[.5,.99] and recipe['prior_betas']==[0.,.99]
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.autograd.set_multithreading_enabled(False)
rows=[]
for name,task in sorted(request['tasks'].items()):
    if not name.startswith('model_'): continue
    print('START',name,flush=True)
    started=time.monotonic()
    result=run(request,task,raw/name,'cuda:0',execution_limit=2)
    evidence=result['evidence']; guards=evidence['guards']
    expected=dict(generator=2,encoder=0 if name.startswith('model_ddgan') else 2,prior=2,
                  discriminator=0 if name=='model_ae_reconstruction' else 2)
    assert result['scope']=='integration_demo_only' and result['cost']['completed_steps']==2
    assert guards['all_finite'] is True and guards['unintended_rng_deviations']==0 and guards['hooks_exercised'] is True
    assert guards['optimizer_updates']==expected,(name,guards['optimizer_updates'],expected)
    gif=raw/name/'api/goal.gif'; assert gif.read_bytes().startswith(b'GIF')
    grade=grade_result(task,result); assert grade['gate_status']!='PASS'
    modules={key:str(Path(module.__file__).resolve()) for key,module in sys.modules.items()
             if (key.startswith(('experiments.forge','benchmarks.toy_audit','particlegan')) and getattr(module,'__file__',None))}
    assert all(Path(path).is_relative_to(root) for path in modules.values()),modules
    row=dict(task_id=name,status='PASS',scope='integration_demo_only',qualification_input=False,
             completed_steps=2,elapsed_seconds=time.monotonic()-started,guards=guards,
             ordinary_gate_status=grade['gate_status'],initial_models=evidence['initial_models'],
             batch_sequence_sha256=evidence['batch_sequence_sha256'],
             actual_training_gif=evidence['actual_training_gif'],
             adapter_receipt_sha256=file_hash(raw/name/'adapter-receipt.json'),
             api_receipt_sha256=evidence['api_receipt_sha256'],
             provenance_checkpoint=evidence['provenance_checkpoint'],imported_source_files=modules)
    rows.append(row)
    atomic_json(raw/'smoke-results.json',dict(source_digest=request['source']['digest'],request_sha256=file_hash(request_path),rows=rows))
    print('PASS',name,round(row['elapsed_seconds'],3),expected,flush=True)
assert len(rows)==8
verify_snapshot(root,request['source'])
print('COMPLETE 8/8 integration-only; no ordinary gate qualification',flush=True)
