"""Publish saved Forge results and GIFs; no training, sampling, or rendering."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, stable_hash


def export(queue):
    report=Path(__file__).resolve().parent
    rows=[]
    media=report/'media'
    media.mkdir(exist_ok=True)
    for directory in sorted(queue.glob('vaegan-coverage-round*-v1/*')):
        if not (directory/'request.json').is_file():
            continue
        envelope=json.loads((directory/'request.json').read_text())
        request=envelope['request']; task_id=envelope['job']['task_id']
        task=request['tasks'][task_id]
        raw_path=directory/'api/receipt.json'
        adapter_path=directory/'adapter-receipt.json'
        if not raw_path.is_file() or not adapter_path.is_file():
            raise ValueError(f'Incomplete attempt must be examined before publication: {directory}')
        raw=json.loads(raw_path.read_text()); adapter=json.loads(adapter_path.read_text())
        terminal_receipt=json.loads((directory/'terminal.json').read_text())
        recorded=json.loads((directory/'raw-result.json').read_text())
        if (terminal_receipt['attempt_status']!='completed'
                or terminal_receipt['grading']['raw_hash']!=stable_hash(recorded)
                or terminal_receipt['grading']['source_digest']!=request['source']['digest']):
            raise ValueError('Missing or mismatched independent frozen-source grading')
        grade=terminal_receipt['grading']['grades'][task_id]
        scoring=[v for v in raw['observations'] if v['step'] in raw['protocol']['metric_evaluation_steps'] and v['step']>0]
        terminal=scoring[-5:]
        suffix=0
        for v in reversed(scoring):
            if not v['passed']: break
            suffix+=1
        candidate=request['candidate']['id']
        name=f'{candidate}-{task_id}'
        gif=media/f'{name}.gif'
        shutil.copyfile(directory/'api/goal.gif',gif)
        checkpoints=adapter['evidence']['observation_checkpoints']
        if {v['step'] for v in scoring}-{v['step'] for v in checkpoints}:
            raise ValueError('Missing scheduled complete checkpoint')
        failed_terminal=[dict(step=v['step'],failed_bounds=v['failed_bounds']) for v in terminal if not v['passed']]
        rows.append(dict(candidate=candidate,task=task_id,attempt_id=directory.name,
            study_id=request['study']['id'],request_id=envelope['request_id'],
            source_commit=request['source']['origin_commit'],source_digest=request['source']['digest'],
            scientific_job=envelope['job']['science'],recipe=raw['recipe'],recipe_sha256=stable_hash(raw['recipe']),
            task_sha256=stable_hash({k:v for k,v in task.items() if k!='field_ownership'}),
            task_binding_sha256=stable_hash(task),verdict=grade['gate_status'],grade=grade,
            completed_updates=raw['completed_updates'],full_protocol_complete=raw['default_protocol_complete'],
            terminal_passes=sum(v['passed'] for v in terminal),stable_suffix=suffix,metric_observations=len(scoring),
            final_metrics=scoring[-1]['metrics'],terminal_metrics=[dict(step=v['step'],passed=v['passed'],metrics=v['metrics']) for v in terminal],
            failed_terminal=failed_terminal,failed_bounds=raw['failed_bounds'],
            initial_models=raw['reproducibility']['initial_models'],batch_sequence_sha256=raw['reproducibility']['batch_sequence_sha256'],
            guards=adapter['evidence']['guards'],prior_policy=adapter['evidence']['prior_policy'],
            elapsed_execution_seconds=raw['elapsed_seconds'],runtime=raw['runtime'],
            paid_worker_seconds=terminal_receipt['elapsed_seconds'],
            raw_directory=str(directory),raw_receipt_sha256=file_hash(raw_path),adapter_receipt_sha256=file_hash(adapter_path),
            terminal_receipt_sha256=file_hash(directory/'terminal.json'),
            checkpoint_count=len(checkpoints),checkpoint_index_sha256=stable_hash(checkpoints),
            terminal_checkpoints=[v for v in checkpoints if v['step'] in {r['step'] for r in terminal}],
            gif=gif.relative_to(report).as_posix(),gif_sha256=file_hash(gif),qualification_input=False))
    if not rows: raise ValueError('No complete evidence')
    rows.sort(key=lambda v:(v['candidate'],v['task']))
    # Architecture and consumed data law are shared even across the two
    # posterior variants. Recipe differences must not perturb these streams.
    assert len({v['source_digest'] for v in rows})==1, 'unmatched scientific source'
    assert len({stable_hash(v['initial_models']) for v in rows})==1, 'unmatched initial models'
    assert len({v['batch_sequence_sha256'] for v in rows})==1, 'unmatched target batches'
    for task in {v['task'] for v in rows}:
        assert len({v['task_sha256'] for v in rows if v['task']==task})==1, 'unmatched task conditions'
    result=dict(schema='vaegan_bispf_coverage_results_v1',goal='vaegan_coverage',seed=0,
        qualification_input=False,default_adoption=False,speed_ranking=False,
        protocol_sha256=file_hash(report/'protocol.json'),
        matched_initial_models=True,matched_target_batches=True,matched_scientific_source=True,
        actual_execution_seconds=sum(v['elapsed_execution_seconds'] for v in rows),
        paid_worker_seconds=sum(v['paid_worker_seconds'] for v in rows),
        reserved_ceiling_seconds=2400,runs=rows)
    atomic_json(report/'results.json',result)
    lines=['<!-- current-comparison -->','','| Global recipe | Hard VAEGAN | Categorical VAEGAN | Task passes |',
        '| --- | --- | --- | ---: |']
    groups=defaultdict(list)
    for row in rows: groups[row['candidate']].append(row)
    for candidate,arms in sorted(groups.items(),key=lambda item:(-sum(v['verdict']=='PASS' for v in item[1]),item[0])):
        by_mode={v['task']:v for v in arms}
        def cell(mode):
            v=by_mode[f'model_vae_{mode}']
            return f"[{v['verdict']}]({v['gif']}) ({v['terminal_passes']}/5 terminal; suffix {v['stable_suffix']})"
        lines.append(f"| `{candidate}` | {cell('hard')} | {cell('categorical')} | {sum(v['verdict']=='PASS' for v in arms)}/2 |")
    lines.extend(['','All measured arms complete 1,200 updates and24 checks; the last five determine',
        'PASS. This table is the single current comparison for `vaegan_coverage`.',
        'Every revision and failure remains in [compact results](results.json).',
        'GPU0 was shared under load; recorded costs supply no speed ranking.','','<!-- /current-comparison -->'])
    p=report/'README.md'; text=p.read_text()
    start='<!-- current-comparison -->';end='<!-- /current-comparison -->'
    if start in text:
        text=text[:text.index(start)]+'\n'.join(lines)+text[text.index(end)+len(end):]
    else: text+='\n'+'\n'.join(lines)+'\n'
    p.write_text(text)
    print(json.dumps(dict(runs=len(rows),outcomes={k:sum(v['verdict']==k for v in rows) for k in ('PASS','FAIL')},execution_seconds=result['actual_execution_seconds'])))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue-root',required=True,type=Path)
    export(p.parse_args().queue_root)
