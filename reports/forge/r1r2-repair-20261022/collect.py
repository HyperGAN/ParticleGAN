"""Collect original Forge certificates and saved-training media; never train.

Different execution sources remain separate cohorts. Existing verdicts and
physical charge ownership are copied, never recomputed or pooled.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.sources import verify_snapshot
from experiments.forge.state import state_digest
from experiments.forge.tier1_media import render
from reports.forge.regenerate_technique_inventory import project_receipt

CAMPAIGN = 'r1r2-repair-20261022'
DEFAULT_QUEUE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-repair-20261022/queue')
ROLES = {f'r1r2-repair-{role}-20261022': role for role in ('control', 'r1only', 'r1cycle')}


def saved_publisher():
    path = ROOT/'reports/forge/bcap-develop-integration/publish.py'
    spec = importlib.util.spec_from_file_location('r1r2_saved_publication', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dictionaries(value, prefix='state'):
    if isinstance(value, dict):
        yield prefix, value
        for key, item in value.items():
            yield from dictionaries(item, f'{prefix}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from dictionaries(item, f'{prefix}.{index}')


def numerical_parts(saved):
    parts = {name: {} for name in ('models', 'role_parameters', 'optimizers', 'stream_states')}
    data_states = {}
    for path, node in dictionaries(saved):
        for name in ('models', 'role_parameters', 'optimizers'):
            if name in node:
                parts[name][f'{path}.{name}'] = state_digest(node[name])
        if isinstance(node.get('states'), dict):
            for name, value in node['states'].items():
                if not isinstance(value, torch.Tensor):
                    continue
                key = f'{path}.states.{name}'
                parts['stream_states'][key] = state_digest(value)
                try:
                    is_data = json.loads(name)[0] == 'data'
                except (ValueError, TypeError, IndexError):
                    is_data = name == 'data' or name.startswith('data:')
                if is_data:
                    data_states[key] = state_digest(value)
    return {name: value for name, value in parts.items() if value}, data_states


def differing_saved_paths(left,right,path='state'):
    if state_digest(left)==state_digest(right):
        return []
    if isinstance(left,dict) and isinstance(right,dict):
        return [part for key in sorted(left.keys()|right.keys())
                for part in differing_saved_paths(left.get(key),right.get(key),f'{path}.{key}')]
    if isinstance(left,(list,tuple)) and isinstance(right,type(left)) and len(left)==len(right):
        return [part for index,(a,b) in enumerate(zip(left,right)) for part in differing_saved_paths(a,b,f'{path}.{index}')]
    return [path]


def initial_models(row, saved):
    initial = {}
    for path, node in dictionaries(dict(row=row.get('initialization', {}),
        host=row.get('evidence', {}).get('host', {}))):
        if 'initial_state_sha256' in node:
            initial[path] = node['initial_state_sha256']
    initial.update({f'provider.{key}': value for key,value in
                    row.get('evidence', {}).get('initial_models', {}).items()})
    if saved is not None:
        for path, node in dictionaries(saved):
            if isinstance(node.get('initialization'), dict):
                for role, value in node['initialization'].items():
                    if isinstance(value, dict) and 'initial_state_sha256' in value:
                        initial[f'{path}.initialization.{role}'] = value['initial_state_sha256']
    return initial


def checkpoint(row):
    descriptor = row.get('evidence', {}).get('provenance_checkpoint')
    if not descriptor:
        return None, None
    root = Path(descriptor['artifact_root'])
    verify_artifacts(root, descriptor['artifact_manifest'])
    path = root / descriptor['path']
    assert file_hash(path) == descriptor['sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert state_digest(saved) == descriptor['state_sha256']
    parts, data_states = numerical_parts(saved)
    named = descriptor.get('named_stream_state_sha256', {})
    if named:
        assert set(saved['streams']['states']) == set(named)
        assert all(state_digest(saved['streams']['states'][name]) == digest for name,digest in named.items())
    return saved, dict(path=str(path), sha256=descriptor['sha256'],
        state_sha256=descriptor['state_sha256'], artifact_manifest_sha256=descriptor['artifact_manifest']['sha256'],
        completed_steps=descriptor['completed_steps'], purpose=descriptor.get('purpose'),
        numerical_parts=parts, consumed_data_rng_state_sha256=data_states)


def acquired_checkpoint(row):
    evidence = row.get('evidence',{})
    descriptor = evidence.get('checkpoint')
    if row['task_id']!='five_word_joint_smoke' or not descriptor:
        return None
    root=Path(evidence['artifact_root'])
    verify_artifacts(root,evidence['artifact_manifest'])
    path=(root/descriptor['path']).resolve()
    assert path.is_relative_to(root.resolve()) and file_hash(path)==descriptor['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert state_digest(saved)==descriptor['state_sha256']
    return dict(**descriptor,artifact_root=str(root),verified_checkpoint_bytes_and_state=True)


def word_curve_summary(task,row):
    if task['adapter']!='word_joint':
        return None
    evidence=row.get('evidence',{})
    primary=evidence.get('observations',[])
    confirms=evidence.get('confirmations',[])
    if not primary or not confirms:
        return None
    records=[dict(step=point['step'],kind='primary',metrics=point) for point in primary]
    records.extend(dict(step=point['step'],kind='independent_confirmation',metrics=point['metrics']) for point in confirms)
    def passes(value,op,bound):
        return type(value) in (int,float) and math.isfinite(value) and (
            value>=bound if op=='>=' else value<=bound if op=='<=' else value==bound)
    summaries={}
    for name,op,bound in task['evaluation']['thresholds']:
        values=[(record['metrics'][name],record['step'],record['kind']) for record in records]
        low=min(values);high=max(values)
        failed=[record for record in records if not passes(record['metrics'].get(name),op,bound)]
        summaries[name]=dict(minimum=low[0],minimum_step=low[1],minimum_kind=low[2],
            maximum=high[0],maximum_step=high[1],maximum_kind=high[2],
            failed_primary_or_confirmation_count=len(failed),op=op,threshold=bound)
    failed_checks=[dict(step=record['step'],kind=record['kind'],
        failed_metrics=[name for name,op,bound in task['evaluation']['thresholds']
                        if not passes(record['metrics'].get(name),op,bound)]) for record in records]
    failed_checks=sorted((record for record in failed_checks if record['failed_metrics']),key=lambda record:(record['step'],record['kind']))
    return dict(primary_checks=len(primary),independent_confirmation_checks=len(confirms),
        metric_extrema_and_failure_counts=summaries,earliest_failed_check=failed_checks[0] if failed_checks else None,
        confirmation_curve_sha256=stable_hash(confirms),
        interpretation='Descriptive extrema of the original saved scored curve; no observations, grades or qualification decisions added.')


def clock_proof(row):
    if row['task_id']!='clockfree_audit_measurement_v1':
        return None,{}
    evidence=row.get('evidence',{})
    if not evidence.get('artifact_root'):
        return None,{}
    root=Path(evidence['artifact_root'])
    verify_artifacts(root,evidence['artifact_manifest'])
    path=root/'comparisons.pt'
    proof=torch.load(path,map_location='cpu',weights_only=True)
    initial={f'clock_constructor.{role}':value['initial_state_sha256']
             for role,value in proof['initial']['initialization'].items()}
    parts,_=numerical_parts(proof)
    return dict(path=str(path),sha256=file_hash(path),
        comparison_start_after_warmup_models={role:state_digest(model) for role,model in proof['initial']['trainer']['models'].items()},
        numerical_trajectory_sha256=stable_hash(parts),
        interpretation='One digest of actual saved numerical model/optimizer/stream trajectories; no per-update state or metric stream copied.'),initial


def export_media(task, row, local, destination, request):
    evidence = row.get('evidence', {})
    actual = evidence.get('actual_training_gif')
    if actual:
        source = Path(actual['path'])
        assert file_hash(source) == actual['sha256']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        assert file_hash(destination) == actual['sha256']
        receipt = dict(schema_version=1,kind='retained_original_public_api_actual_training_gif',
            task_id=task['id'],recorded_grade=row['gate_status'],source_path=str(source),
            source_sha256=actual['sha256'],gif_sha256=file_hash(destination),
            optimizer_updates_added=0,sampling_draws_added=0,qualification_input=False)
        atomic_json(destination.with_suffix('.json'), receipt)
        return receipt
    if not evidence.get('observations') and task['adapter']!='clockfree_audit':
        return None
    sidecar = destination.with_suffix('.json')
    if destination.exists() and sidecar.exists():
        receipt = read_json(sidecar)
        assert receipt['gif_sha256'] == file_hash(destination)
        assert all(file_hash(Path(path)) == digest for path,digest in receipt.get('source_inputs', {}).items())
        return receipt
    publisher = saved_publisher()
    # This public exporter selects exact scored observations from retained
    # component-view supersets (AE hold also retains pretraining snapshots).
    # It validates saved bytes and exact metric equality; no new observations.
    entry = dict(task=task,row=row,request=request,attempt=dict(local=local))
    return publisher.render_saved(entry,destination,publisher._saved_renderer(ROOT))


def collect(options):
    torch.set_num_threads(1)
    state = read_json(options.queue / 'queue/state.json')
    submissions = {rid: value for rid,value in state['submissions'].items()
        if value['request'].get('campaign_id') == CAMPAIGN and value['request']['candidate']['id'] in ROLES}
    active = [rid for rid,sub in submissions.items() if sub['status'] in {'queued','running','paused'}]
    if not options.allow_partial:
        assert not active, f'active or paused requests: {active}'
    source_proofs = {}
    cohorts, attempts, charges = {}, {}, {}
    rows_by_cohort = {}
    for rid, sub in submissions.items():
        request = sub['request']; source = request['source']; digest = source['digest']
        if digest not in source_proofs:
            verify_snapshot(Path(source['snapshot_path']), source)
            source_proofs[digest] = dict(source_digest=digest,origin_commit=source['origin_commit'],
                snapshot_path=source['snapshot_path'],verified_files=len(source['files']),runtime=request['runtime'])
        role = ROLES[request['candidate']['id']]
        evidence_scope = request['view'].get('evidence_scope','ordinary')
        cohort_key = f'{digest}:{evidence_scope}'
        cohort = cohorts.setdefault(cohort_key, dict(source=source_proofs[digest],evidence_scope=evidence_scope,requests=[],roles=[]))
        authorized = [a for a in request['view']['assignments'] if a['qualification_tier'] <= request['through_tier']]
        authorized_names = {a['task'] for a in authorized}
        request_keys = {job['compatibility_key'] for job in request['jobs']
                        if set(job.get('task_ids',[job['task_id']])) <= authorized_names}
        cohort['requests'].append(dict(request_id=rid,role=role,study=request['study']['id'],
            candidate=request['candidate']['id'],candidate_revision=request['candidate_revision'],
            status=sub['status'],reason=sub.get('reason'),view_revision=request['view']['revision'],
            global_recipe=request['candidate']['recipe_overrides']))
        current = {}
        for job in state['jobs'].values():
            result = job.get('result')
            # Cancellation removes live subscriptions, not original evidence.
            if job['definition']['compatibility_key'] not in request_keys or result is None:
                continue
            aid = result['attempt_id']
            summary = project_receipt(ROOT, aid)
            original_dir = ROOT/'reports/forge/attempts'/aid
            envelope = read_json(original_dir/'request.json')
            certificate = read_json(original_dir/'evidence.json')
            local = Path(certificate['local_artifact_root'])
            assert read_json(original_dir/'result.json') == result == read_json(local/'result.json')
            assert read_json(local/'request.json') == envelope
            assert summary['provenance']['source_digest'] == digest
            physical_charge = [c for c in state['charges'] if c['attempt_id'] == aid]
            assert len(physical_charge) == 1
            charges[aid] = dict(**physical_charge[0],full_allowance_seconds=job['definition']['budget_seconds'],source_digest=digest,evidence_scope=evidence_scope,role=role)
            attempts[aid] = dict(attempt_id=aid,role=role,source_digest=digest,evidence_scope=evidence_scope,
                candidate_revision=request['candidate_revision'],
                provenance=summary['provenance'],original_artifact_root=str(local),
                independent_grading=result['raw'].get('grading',{}).get('grades',{}),
                original_attempt_status=result['raw']['attempt_status'])
            compact = {r['task_id']:r for r in summary['task_results']}
            for row in result['task_results']:
                name = row['task_id'];task = request['tasks'][name]
                saved, cp = checkpoint(row)
                clock,clock_initial=clock_proof(row)
                item = dict(**compact[name],attempt_id=aid,role=role,source_digest=digest,evidence_scope=evidence_scope,
                    request_id=rid,study_id=request['study']['id'],
                    task_contract_sha256=stable_hash(task),effective_recipe=row.get('recipe',row.get('applied',{}).get('recipe')),
                    initial_prior=row.get('initial_prior'),prior=row.get('prior',row.get('applied',{}).get('prior')),
                    execution_sampling={key:row.get('evidence',{}).get(key) for key in ('sampling_law','eval_output_noise','sampling_contract_version')},
                    prior_policy=row.get('evidence',{}).get('prior_policy'),
                    continuity=row.get('evidence',{}).get('continuity'),
                    acquired_checkpoint=acquired_checkpoint(row),
                    original_word_curve_summary=word_curve_summary(task,row),
                    initialization={**initial_models(row,saved),**clock_initial},checkpoint=cp,clock_audit_proof=clock,
                    batch_sequence_sha256=(row.get('evidence',{}).get('batch_sequence_sha256') or
                        (stable_hash(row['evidence']['data_sha256']) if row.get('evidence',{}).get('data_sha256') else None)),
                    consumed_batch_traces=row.get('evidence',{}).get('data_sha256'),
                    observations_sha256=stable_hash(row.get('evidence',{}).get('observations',[])),
                    training_budget=dict(steps=task['execution']['steps'],
                        original_schedule_horizon=task['execution'].get('original_schedule_horizon'),
                        timeout_seconds=task['resources']['timeout_seconds']),
                    qualification_input=False,qualification_reuse=False)
                dest=options.output/'media'/f'{digest[:12]}-{evidence_scope}-{role}-{name}-{aid[:8]}.gif'
                if options.media or dest.exists():
                    media = export_media(task,row,local,dest,request)
                    if media:
                        item['actual_training_gif']=dict(path=str(dest.relative_to(ROOT)),
                            sha256=file_hash(dest),sidecar_sha256=file_hash(dest.with_suffix('.json')))
                    else:
                        item['actual_training_gif_gap']='No retained completed training observations; no illustration fabricated.'
                current[name]=item
                rows_by_cohort.setdefault(cohort_key,{})[(role,name)] = item
        cohort['roles'].append(dict(role=role,study_id=request['study']['id'],request_id=rid,
            required_counts=dict(Counter(current[a['task']]['gate_status'] if a['task'] in current else 'UNMEASURED'
                for a in authorized if a['importance']=='required')),
            diagnostic_counts=dict(Counter(current[a['task']]['gate_status'] if a['task'] in current else 'UNMEASURED'
                for a in authorized if a['importance']=='diagnostic')),
            tasks=[dict(**a,result=current.get(a['task'])) for a in authorized]))
    comparisons=[]
    word_parent_proofs=[]
    for cohort_key, lookup in rows_by_cohort.items():
        digest,evidence_scope=cohort_key.split(':',1)
        for role in ROLES.values():
            hold=lookup.get((role,'five_word_joint_hold'))
            if not hold or not hold.get('continuity'):
                continue
            parent=lookup.get((role,'five_word_joint_smoke'))
            assert parent and parent['gate_status']=='PASS'
            continuity=hold['continuity']; acquired=parent['acquired_checkpoint']
            assert acquired and continuity['parent_checkpoint_sha256']==acquired['sha256']
            assert continuity['parent_state_sha256']==acquired['state_sha256']
            assert continuity['parent_confirmed_step']==acquired['completed_steps']==continuity['prefix_steps']
            assert continuity['parent_candidate_revision']==attempts[parent['attempt_id']]['candidate_revision']
            assert continuity['parent_compatibility_key']==parent['compatibility_key']
            assert continuity['restored_exactly'] is True and continuity['history_reset'] is False
            assert continuity['same_recipe_prior_architecture'] is True
            word_parent_proofs.append(dict(source_digest=digest,evidence_scope=evidence_scope,role=role,
                smoke_attempt_id=parent['attempt_id'],hold_attempt_id=hold['attempt_id'],
                acquired_checkpoint=acquired,continuity=continuity,
                parent_check='Verified exact own same-source/evidence-use smoke bytes and tensor state, earliest confirmed step, candidate revision and unchanged history; original evaluator remains authoritative.'))
        for task in sorted({name for role,name in lookup}):
            available = {role:lookup[(role,task)] for role in ('control','r1only','r1cycle') if (role,task) in lookup}
            initial = {role:row['initialization'] for role,row in available.items() if row['initialization']}
            batch = {role:row['batch_sequence_sha256'] for role,row in available.items() if row['batch_sequence_sha256']}
            data = {role:row['checkpoint']['consumed_data_rng_state_sha256'] for role,row in available.items()
                    if row['checkpoint'] and row['checkpoint']['consumed_data_rng_state_sha256']}
            proof=dict(source_digest=digest,evidence_scope=evidence_scope,task_id=task,roles_available=sorted(available),
                initial_model_digests=initial,initial_models_match=(len({stable_hash(v) for v in initial.values()})==1 if len(initial)>1 else None),
                actual_batch_digests=batch,actual_batches_match=(len(set(batch.values()))==1 if len(batch)>1 else None),
                consumed_data_rng_digests=data,consumed_data_rng_states_match=(len({stable_hash(v) for v in data.values()})==1 if len(data)>1 else None),
                actual_batch_trace_gap=(None if batch else 'No direct consumed-batch trace is retained by this adapter; named data-stream endpoints are separate evidence.'))
            if task=='five_word_joint_hold':
                prefixes={role:row['continuity']['prefix_steps'] for role,row in available.items() if row.get('continuity')}
                proof['own_acquisition_prefix_by_role']=prefixes
                proof['matching_continuation_start_steps']=(len(set(prefixes.values()))==1 if len(prefixes)>1 else None)
                proof['continuation_comparison_scope']='Each unchanged task restores its own earliest confirmed acquisition. Different acquired prefixes select different windows of the same declared training-data stream; do not claim identical held minibatches from matching protocol alone.'
            left,right=available.get('r1only'),available.get('r1cycle')
            if task!='five_word_joint_smoke' and left and right and left['checkpoint'] and right['checkpoint']:
                a,b=left['checkpoint']['numerical_parts'],right['checkpoint']['numerical_parts']
                proof['inactive_joint_inverse_comparison']=dict(
                    numerical_checkpoint_parts_match=a==b,parts_left=a,parts_right=b,
                    observations_match=left['observations_sha256']==right['observations_sha256'],
                    original_grades_match=left['gate_status']==right['gate_status'],
                    excludes='Recipe metadata differs by the declared inapplicable global joint coefficient; original identities and charges remain separate.')
            if task=='clockfree_audit_measurement_v1' and left and right and left['clock_audit_proof'] and right['clock_audit_proof']:
                proof['inactive_joint_inverse_clock_comparison']=dict(
                    numerical_training_proof_match=(left['clock_audit_proof']['numerical_trajectory_sha256']==right['clock_audit_proof']['numerical_trajectory_sha256']),
                    original_grades_match=left['gate_status']==right['gate_status'])
            comparisons.append(proof)
    cross_scope_acquisition_replays=[]
    for digest in source_proofs:
        ordinary=rows_by_cohort.get(f'{digest}:ordinary',{})
        diagnostic=rows_by_cohort.get(f'{digest}:research_diagnostic',{})
        for role in ROLES.values():
            left=ordinary.get((role,'five_word_joint_smoke'))
            right=diagnostic.get((role,'five_word_joint_smoke'))
            if not left or not right:
                continue
            acquired_left,acquired_right=left['acquired_checkpoint'],right['acquired_checkpoint']
            acquired_parts_match=None;acquired_differences=None
            if acquired_left and acquired_right:
                first=torch.load(Path(acquired_left['artifact_root'])/acquired_left['path'],weights_only=True,map_location='cpu')
                second=torch.load(Path(acquired_right['artifact_root'])/acquired_right['path'],weights_only=True,map_location='cpu')
                acquired_parts_match=numerical_parts(first)==numerical_parts(second)
                acquired_differences=differing_saved_paths(first,second)
            cross_scope_acquisition_replays.append(dict(source_digest=digest,role=role,
                ordinary_attempt_id=left['attempt_id'],diagnostic_attempt_id=right['attempt_id'],
                original_gate_statuses=[left['gate_status'],right['gate_status']],
                initial_models_match=left['initialization']==right['initialization'],
                observations_match=left['observations_sha256']==right['observations_sha256'],
                numerical_checkpoint_parts_match=(left['checkpoint']['numerical_parts']==right['checkpoint']['numerical_parts']
                    if left['checkpoint'] and right['checkpoint'] else None),
                acquired_state_matches=(left['acquired_checkpoint']['state_sha256']==right['acquired_checkpoint']['state_sha256']
                    if left['acquired_checkpoint'] and right['acquired_checkpoint'] else None),
                acquired_model_optimizer_named_stream_parts_match=acquired_parts_match,
                differing_acquired_checkpoint_paths=acquired_differences,
                evidence_reuse=False,physical_costs_separate=True,
                interpretation='Evidence-use-specific keys require distinct original receipts. Whole checkpoints may differ in ambient global CPU/CUDA RNG snapshots even when consumed named streams, model/optimizer tensors and observations match. Each hold restores its own complete checkpoint, including its own ambient snapshots. Replay equality grants no cross-scope credit or checkpoint substitution.'))
    campaign_charges={charge['attempt_id'] for charge in state['charges'] if charge['owner']['campaign']==CAMPAIGN}
    assert set(charges)==campaign_charges, 'every charged physical attempt needs its original certificate projection'
    output=dict(schema_version=1,scope='source_separated_original_campaign_results',campaign=CAMPAIGN,
        partial_snapshot=bool(active),active_request_ids=active,
        qualification_input=False,qualification_reuse=False,training_updates_added=0,sampling_draws_added=0,
        source_cohorts=cohorts,physical_attempts=attempts,unique_paid_costs=charges,
        total_unique_paid_seconds=sum(c['seconds'] for c in charges.values()),
        physical_attempt_count=len(charges),campaign_budget=state['campaigns'][CAMPAIGN]['definition'],
        comparisons=comparisons,word_parent_proofs=word_parent_proofs,
        cross_scope_acquisition_replays=cross_scope_acquisition_replays,generated_leaderboard=False,
        interpretation='Existing certified verdicts only. Sources, recipes, checkpoint parents and physical charges remain distinct. UNKNOWN, cancelled, BLOCKED and INCOMPLETE are not scientific FAILs.')
    options.output.mkdir(parents=True,exist_ok=True)
    atomic_json(options.output/'results.json',output)
    print(json.dumps(dict(source_cohorts=len(cohorts),physical_attempts=len(charges),
        paid_seconds=output['total_unique_paid_seconds'],output=str(options.output/'results.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,default=DEFAULT_QUEUE)
    parser.add_argument('--output',type=Path,default=HERE/'publication')
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--media',action='store_true')
    collect(parser.parse_args())
