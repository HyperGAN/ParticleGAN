"""Read-only exploratory per-row diagnosis on saved VAEGAN terminal states.

No optimizer is constructed or updated. CPU observations use a separate local
panel; they are diagnostics, not replacements for the recorded CUDA gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from particlegan import Recipe
from benchmarks.toy_audit.api_model_vae import ANCHORS, sample_target
from experiments.forge.contracts import atomic_json, file_hash


def analyze(directory):
    request = json.loads((directory / 'request.json').read_text())
    task_id = request['job']['task_id'] if 'job' in request else ''
    rows = []
    for path in sorted((directory / 'observation-checkpoints').glob('step-*.pt')):
        step = int(path.stem.split('-')[-1])
        if step not in (1000, 1050, 1100, 1150, 1200):
            continue
        original_sha = file_hash(path)
        saved = torch.load(path, weights_only=False, map_location='cpu')
        recipe = Recipe.from_dict(saved['policy']['recipe'])
        def mlp(out):
            # Construction is isolated from the caller's global RNG below.
            return nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(.2), nn.Linear(32, out))
        with torch.random.fork_rng(devices=[]):
            encoder, generator = mlp(2), mlp(2)
            prior = recipe.make_prior(sigma=.025)
        states = saved['policy']['models']
        encoder.load_state_dict(states['encoder'])
        generator.load_state_dict(states['generator'])
        recipe.apply_prior_policy(prior)
        prior.load_state_dict(states['prior'])
        with torch.no_grad():
            rng = torch.Generator().manual_seed(10000)
            data = sample_target(4096, rng, 'cpu')
            encoded = recipe.encode(encoder(data), prior, draws=8, generator=rng)
            usage = encoded.log_probs.exp().mean(0)
            centers = prior.means()
            noise = .025 * torch.randn(16, 1024, 2, generator=rng)
            prediction = generator((centers[:, None] + noise).flatten(0, 1)).reshape(16, 1024, 2)
            distances = torch.cdist(prediction.flatten(0, 1), prediction.new_tensor(ANCHORS)).min(1).values.reshape(16, 1024)
            row_hq = (distances <= .2).float().mean(1)
            low = usage < .01
            misses = 1 - row_hq
            low_attribution = float(misses[low].sum() / misses.sum()) if float(misses.sum()) else None
            rows.append(dict(step=step, checkpoint_sha256=original_sha,
                posterior_mass=usage.tolist(), row_decoder_mean_hq=row_hq.tolist(),
                low_usage_rows=int(low.sum()), low_usage_threshold=.01,
                equal_row_quality=float(row_hq.mean()), low_usage_miss_attribution=low_attribution,
                posterior_prior_mass_tv=float((usage-1/16).abs().sum()/2),
                center_anchor_distance=torch.cdist(generator(centers), centers.new_tensor(ANCHORS)).min(1).values.tolist()))
        if file_hash(path) != original_sha:
            raise ValueError('saved checkpoint changed during read-only diagnosis')
    return dict(attempt=directory.name, terminal_states=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    result=dict(schema='vaegan_saved_terminal_rows_v1', qualification_input=False,
                optimizer_updates=0, source_sha256=file_hash(Path(__file__)),
                scope='Exploratory CPU row-stratified1024draws/row and4096-input posterior panel; no CUDA gate replacement.',
                attempts=[analyze(p) for p in sorted(args.campaign.iterdir()) if (p/'request.json').is_file()])
    atomic_json(args.output,result)
    print(json.dumps(dict(output=str(args.output),attempts=len(result['attempts']))))

if __name__=='__main__':
    main()
