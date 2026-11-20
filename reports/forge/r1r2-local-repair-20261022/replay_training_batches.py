"""Recover original batch draw identities, checking full sequence and RNG state.

No model is constructed or trained. Original CPU target and CUDA index/noise
streams are replayed independently from their receipt-bound seeds. Returned
latent row/noise identities may be evaluated only with explicitly frozen models;
they do not reconstruct intermediate historical generator or prior parameters.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

import torch


def state_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def recover(root, keep=16):
    root = Path(root)
    request = json.loads((root / 'request.json').read_text())['request']
    raw = json.loads((root / 'raw-result.json').read_text())
    state = torch.load(root / 'provenance/provenance-state.pt', map_location='cpu', weights_only=True)
    source = request['source']
    snapshot = Path(source['snapshot_path'])
    rels = ['benchmarks/transfer_suite/vector_tasks.py', 'experiments/forge/state.py', 'particlegan/training.py', 'particlegan/particle_prior.py']
    for relative in rels:
        assert hashlib.sha256((snapshot/relative).read_bytes()).hexdigest() == source['files'][relative]
    sample = extract(snapshot / rels[0], {'sample_target', 'target_scale'}, {'torch': torch, 'math': math})['sample_target']
    digest = extract(snapshot / rels[1], {'state_digest'}, {'torch': torch, 'hashlib': hashlib, 'json': json})['state_digest']
    manifest, archived_states = state['streams']['manifest']['bindings'], state['streams']['states']
    keys = {'data': '["data","target","training","cpu"]',
        'indices': '["prior","latent","indices","cuda:0"]',
        'noise': '["noise","prior","gaussian","cuda:0"]'}
    cloned = {}
    checkpoints = {}
    for name,key in keys.items():
        binding = manifest[key]
        cloned[name] = torch.Generator(device=binding['device']).manual_seed(binding['seed'])
        checkpoints[name] = {'key': key, 'seed': binding['seed'], 'device': binding['device'],
            'initial_state_sha256': state_sha(cloned[name].get_state())}
        assert checkpoints[name]['initial_state_sha256'] == binding['initial_state_sha256']
    task = request['tasks']['ring16_acquisition']
    spec = task['execution']['host_definition']
    assert state['trainer']['completed_steps'] == 1600 and state['recipe']['batch_size'] == 128
    assert state['recipe']['input_noise_std'] == state['recipe']['output_noise_std'] == state['recipe']['latent_damping_max_rate'] == 0
    sequence = hashlib.sha256(b'forge-actual-target-batches-v1\0').hexdigest()
    batches = []
    for step in range(1, 1601):
        real = sample(spec, 128, cloned['data'], step-1)
        sequence = hashlib.sha256(bytes.fromhex(sequence) + bytes.fromhex(digest(real))).hexdigest()
        for role in ('critic', 'generator'):
            indices = torch.randint(0, 256, (128,), device='cuda:0', generator=cloned['indices'])
            noise = torch.randn((128, 4), device='cuda:0', dtype=torch.float32, generator=cloned['noise'])
            if step > 1600-keep and role == 'generator':
                batches.append({'step': step, 'real': real.clone(), 'indices': indices.cpu(), 'noise': noise.cpu()})
    comparison = raw['evidence']['comparison_protocol']
    for role in ('critic', 'generator'):
        assert comparison['actual_target_batches'][role] == {'batches': 1600, 'samples': 204800, 'sha256': sequence}
    for name,key in keys.items():
        final = cloned[name].get_state().cpu()
        assert torch.equal(final, archived_states[key])
        checkpoints[name]['final_state_sha256'] = state_sha(final)
        checkpoints[name]['matches_archived_terminal_state'] = True
    return batches, {'scope': 'exact historical named-stream replay; no model/trainer replay',
        'source_digest': source['digest'], 'source_files': {r:source['files'][r]for r in rels},
        'full_actual_target_sequence_sha256': sequence, 'target_batches_verified': 1600,
        'target_rows_replayed': 204800, 'prior_index_batches_replayed': 3200,
        'prior_noise_batches_replayed': 3200, 'retained_generator_steps': [r['step']for r in batches],
        'stream_checkpoints': checkpoints, 'tensor_bundle_sha256': digest(batches),
        'new_independent_streams': 0, 'training_updates_added': 0,
        'historical_draws_replayed': True, 'historical_model_parameters_reconstructed': False}
