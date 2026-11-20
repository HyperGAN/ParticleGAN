"""Inspect two frozen word states without training, model construction or draws.

Run from repository root: python reports/forge/r1r2-failure-diagnosis-20261022/word_saturation_probe.py
Five saved prior atoms and the five declared one-hot targets are diagnostic
enumerations, never replacement metric panels or historical training batches.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.state import state_digest

QUEUE = Path('/mnt/ml7tb/ParticleGAN-forge/recipe-owned-priors-20261010/queue')
CAMPAIGN = QUEUE / 'recipe-prior-refactor-selected-leaders-v1'
HOLD_ID = 'd765aa2da2d1434b9234f7114309c60e'
ACQUIRED_ID = '4e6ef24d55cf4b28b23a1be665a1f836'
PUBLICATION = ROOT / 'reports/forge/recipe-prior-refactor/publication/results.json'
ARCHIVE_COMMIT = '2cf7b0a4c796e29ace6cca12d2e553ad67211b5c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mlp(x, saved, indices):
    x = x.flatten(1)
    for index in indices:
        x = F.linear(x, saved[f'net.{index}.weight'], saved[f'net.{index}.bias'])
        if index != indices[-1]:
            x = F.leaky_relu(x, .2)
    return x


def summary(x):
    return dict(min=float(x.min().detach()), mean=float(x.mean().detach()),
                max=float(x.max().detach()))


def decoded(x):
    chars = 'abcdefghijklmnopqrstuvwxyz_ '
    return [''.join(chars[int(v)] for v in row) for row in x.argmax(1)]


def norm(x):
    return float(torch.linalg.vector_norm(x.detach()))


def analyze(path, observed, loss_api, target):
    packet = torch.load(path, map_location='cpu', weights_only=True)
    saved = packet['fixture']['api_state']
    original_streams = state_digest(packet['streams'])
    models = saved['models']
    assert torch.equal(target, observed['views'][0]['target'])
    prior32 = models['prior']['z'].detach()
    encoded32 = mlp(target, models['encoder'], (1, 3, 5))
    reconstruction32 = mlp(encoded32, models['generator'], (0, 2, 4)).reshape(5, 28, 6).softmax(1)
    stored_reconstruction = observed['views'][0]['samples']
    residual = float((reconstruction32 - stored_reconstruction).abs().max())
    # CPU functional arithmetic versus the original CUDA observer, not a regrade.
    assert residual < 2e-5, residual
    dtype_rows = []
    for dtype in (torch.float32, torch.float64):
        generator = {name: value.to(dtype).clone().requires_grad_(True)
                     for name, value in models['generator'].items()}
        encoder = {name: value.to(dtype).clone().requires_grad_(True)
                   for name, value in models['encoder'].items()}
        critic = {name: value.to(dtype) for name, value in models['critic'].items()}
        prior = prior32.to(dtype).clone().requires_grad_(True)
        words = target.to(dtype)
        encoded = mlp(words, encoder, (1, 3, 5))
        logits = mlp(prior, generator, (0, 2, 4)).reshape(5, 28, 6)
        probabilities = logits.softmax(1)
        fake_score = mlp(torch.cat((probabilities.flatten(1), prior), 1), critic, (0, 2, 4))[:, 0]
        real_score = mlp(torch.cat((words.flatten(1), encoded), 1), critic, (0, 2, 4))[:, 0]
        # Exact uniform independent five-by-five atom objective. No pairing draw.
        objective = loss_api.joint_g_loss(fake_score[:, None].expand(5, 5).flatten(),
                                         real_score[None, :].expand(5, 5).flatten())
        grad_probability, grad_logits = torch.autograd.grad(
            objective, (probabilities, logits), retain_graph=True)
        expected = probabilities * (grad_probability - (probabilities * grad_probability).sum(1, keepdim=True))
        softmax_backward_error = float((expected - grad_logits).abs().max().detach())
        assert softmax_backward_error < (1e-9 if dtype == torch.float64 else 1e-6)
        parameters = [*generator.values(), *encoder.values(), prior]
        derivatives = torch.autograd.grad(objective, parameters, retain_graph=True)
        prior_via_generator = torch.autograd.grad(probabilities, prior,
            grad_outputs=grad_probability.detach(), retain_graph=True)[0]
        prior_direct = derivatives[-1] - prior_via_generator
        gnorm = sum(norm(v)**2 for v in derivatives[:len(generator)])**.5
        enorm = sum(norm(v)**2 for v in derivatives[len(generator):-1])**.5
        pnorm = norm(derivatives[-1])
        recon_logits = mlp(encoded, generator, (0, 2, 4)).reshape(5, 28, 6)
        recon = recon_logits.softmax(1)
        correct = (recon * words).sum(1)
        clamped_nll = -correct.clamp_min(1e-8).log().mean()
        stable_cross_entropy = -(recon_logits.log_softmax(1) * words).sum(1).mean()
        clamp_gradient = torch.autograd.grad(clamped_nll, recon_logits, retain_graph=True)[0]
        ce_gradient = torch.autograd.grad(stable_cross_entropy, recon_logits)[0]
        top2 = logits.topk(2, dim=1).values
        probability_force_by_atom = grad_probability.flatten(1).norm(dim=1)
        assert bool((probability_force_by_atom > 0).all())
        ratio_by_atom = grad_logits.flatten(1).norm(dim=1) / probability_force_by_atom
        dtype_rows.append(dict(
            dtype=str(dtype), generated_atom_strings=decoded(probabilities),
            reconstructed_strings=decoded(recon),
            generated_logits=summary(logits), generated_top_two_logit_gap=summary(top2[:, 0] - top2[:, 1]),
            generated_exact_zero_probabilities=int((probabilities == 0).sum()),
            generated_exact_unit_probabilities=int((probabilities == 1).sum()),
            reconstructed_logits=summary(recon_logits),
            reconstructed_exact_zero_probabilities=int((recon == 0).sum()),
            minimum_correct_reconstruction_probability_by_word=correct.amin(1).detach().tolist(),
            joint_generator_loss=float(objective.detach()), fake_scores=fake_score.detach().tolist(),
            real_scores=real_score.detach().tolist(), probability_gradient_norm=norm(grad_probability),
            probability_gradient_norm_by_atom=probability_force_by_atom.detach().tolist(),
            logit_gradient_norm=norm(grad_logits), probability_to_logit_gradient_ratio=norm(grad_logits)/norm(grad_probability),
            probability_to_logit_gradient_ratio_by_atom=ratio_by_atom.detach().tolist(),
            atoms_with_gradient_ratio_below_1e_minus_6=int((ratio_by_atom < 1e-6).sum()),
            generator_parameter_gradient_norm=gnorm, encoder_parameter_gradient_norm=enorm,
            prior_gradient_norm=pnorm, analytic_softmax_backward_max_error=softmax_backward_error,
            prior_gradient_via_generator_norm=norm(prior_via_generator),
            prior_gradient_direct_joint_latent_norm=norm(prior_direct),
            prior_generator_to_direct_gradient_ratio_by_atom=(prior_via_generator.norm(dim=1)/prior_direct.norm(dim=1).clamp_min(torch.finfo(dtype).tiny)).detach().tolist(),
            diagnostic_reconstruction_clamped_nll=float(clamped_nll.detach()),
            diagnostic_reconstruction_stable_cross_entropy=float(stable_cross_entropy.detach()),
            diagnostic_reconstruction_clamped_gradient_norm=norm(clamp_gradient),
            diagnostic_reconstruction_stable_ce_gradient_norm=norm(ce_gradient),
            diagnostic_reconstruction_clamped_to_stable_gradient_ratio=norm(clamp_gradient)/norm(ce_gradient),
            reconstruction_tokens_below_scorer_clamp=int((correct < 1e-8).sum())))
    assert original_streams == state_digest(packet['streams'])
    return dict(path=str(path), checkpoint_sha256=sha(path),
                complete_state_sha256=state_digest(packet), consumed_streams_sha256=original_streams,
                step=saved['completed_steps'], observer_step=observed['step'],
                cpu_functional_vs_saved_cuda_reconstruction_max_error=residual,
                original_observer_metrics=observed['metrics'], diagnostics=dtype_rows)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    hold = CAMPAIGN / HOLD_ID
    request = json.loads((hold / 'request.json').read_text())['request']
    assert request['candidate']['recipe_overrides']['loss'] == 'relativistic'
    source = QUEUE / 'snapshots' / request['source']['digest']
    for name, expected in request['source']['files'].items():
        assert sha(source / name) == expected, name
    public_loss_path = source / 'particlegan/gan_loss.py'
    spec = importlib.util.spec_from_file_location('frozen_word_probe_loss', public_loss_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loss = module.GANLoss('relativistic')
    publication = json.loads(PUBLICATION.read_text())
    assert subprocess.check_output(['git', 'show', ARCHIVE_COMMIT + ':' + str(PUBLICATION.relative_to(ROOT))], cwd=ROOT) == PUBLICATION.read_bytes()
    certificate_proofs, raw_results = {}, {}
    for attempt_id in (ACQUIRED_ID, HOLD_ID):
        expected = publication['exact_certificate_hashes'][attempt_id]
        original = QUEUE.parent / 'certificates' / attempt_id
        for kind, digest in expected.items():
            assert sha(original / (kind + '.json')) == digest
        for kind in ('request', 'result'):
            assert sha(CAMPAIGN / attempt_id / (kind + '.json')) == expected[kind]
        result = json.loads((original / 'result.json').read_text())
        raw_path = CAMPAIGN / attempt_id / 'raw-result.json'
        raw = json.loads(raw_path.read_text())
        attempt_request = json.loads((original / 'request.json').read_text())['request']
        assert attempt_request['source']['digest'] == request['source']['digest']
        assert attempt_request['candidate_revision'] == request['candidate_revision']
        row = next(row for row in publication['certified_results'] if row['attempt_id'] == attempt_id)
        assert raw == result['raw']['result']
        raw_content_hash = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert raw_content_hash == row['independent_grading']['raw_result_sha256']
        raw_results[attempt_id] = raw
        certificate_proofs[attempt_id] = dict(published_certificate_sha256=expected,
            local_request_and_result_bytes_match=True,
            raw_result_matches_published_certificate=True,
            raw_result_file_sha256=sha(raw_path), raw_result_canonical_content_sha256=raw_content_hash)
    published = next(row for row in publication['certified_results'] if row['attempt_id'] == HOLD_ID)
    endpoint = hold / 'provenance/provenance-state.pt'
    assert sha(endpoint) == published['provenance_checkpoint']['sha256']
    assert state_digest(torch.load(endpoint, map_location='cpu', weights_only=True)) == published['provenance_checkpoint']['state_sha256']
    records_path = hold / 'observed-records.pt'
    assert sha(records_path) == raw_results[HOLD_ID]['evidence']['saved_observer_outputs']['sha256']
    records = torch.load(records_path, map_location='cpu', weights_only=True)
    acquired = CAMPAIGN / ACQUIRED_ID / 'acquisition/acquired-state.pt'
    acquired_proof = raw_results[ACQUIRED_ID]['evidence']['checkpoint']
    assert sha(acquired) == acquired_proof['sha256'] == raw_results[HOLD_ID]['evidence']['continuity']['parent_checkpoint_sha256']
    assert state_digest(torch.load(acquired, map_location='cpu', weights_only=True)) == acquired_proof['state_sha256']
    acquired_packet = torch.load(acquired, map_location='cpu', weights_only=True)
    endpoint_packet = torch.load(endpoint, map_location='cpu', weights_only=True)
    assert acquired_packet['fixture']['recipe'] == endpoint_packet['fixture']['recipe']
    assert acquired_packet['fixture']['api_state']['recipe'] == endpoint_packet['fixture']['api_state']['recipe']
    actual_recipe = endpoint_packet['fixture']['recipe']
    for name, value in dict(batch_size=256, z_dim=2, num_particles=5,
        prior_kind='particles', sigma_rel=0., standardize=False,
        total_steps=20000, prior_reg=0., input_noise_std=0., output_noise_std=0., ema_decay=0.,
        reg_arm='a_r1r2', optimizer_family='adam').items():
        assert actual_recipe[name] == value
    assert actual_recipe.get('kinetic_transport_weight', 0.) == 0.
    assert actual_recipe.get('kinetic_transport_local_weight', 0.) == 0.
    assert 'component_transport' not in endpoint_packet['fixture']
    declared_prior = request['tasks']['five_word_joint_hold']['execution']['prior']
    assert all(declared_prior[name] == value for name,value in
        dict(kind='particle_cloud', sigma=0., standardize=False).items())
    definition = request['tasks']['five_word_joint_hold']['execution']['host_definition']
    assert definition['words'] == ['apple', 'grape', 'lemon', 'melon', 'berry']
    assert definition['characters'] == 'abcdefghijklmnopqrstuvwxyz_ ' and definition['length'] == 6
    target = torch.zeros(5,28,6)
    for row, word in enumerate(definition['words']):
        for position, char in enumerate(word + '_'):
            target[row, definition['characters'].index(char), position] = 1
    for packet in (acquired_packet, endpoint_packet):
        assert tuple(packet['fixture']['api_state']['models']['prior']['z'].shape) == (5,2)
        for role, indices, widths in [('generator',(0,2,4),(2,64,128,168)),
            ('encoder',(1,3,5),(168,128,64,2)),('critic',(0,2,4),(170,256,128,1))]:
            for index, inputs, outputs in zip(indices,widths[:-1],widths[1:]):
                assert tuple(packet['fixture']['api_state']['models'][role][f'net.{index}.weight'].shape) == (outputs,inputs)
    assert records[0]['step'] == 2501 and records[-1]['step'] == 6501
    result = dict(schema_version=1, scope='saved_word_atoms_and_declared_target_enumeration_only',
        qualification_input=False, qualification_reuse=False, training_updates=0,
        random_draws=0, model_constructions=0, device='cpu', torch=str(torch.__version__),
        source_digest=request['source']['digest'], source_commit=request['source']['origin_commit'],
        request_sha256=sha(hold/'request.json'), publication_sha256=sha(PUBLICATION),
        immutable_publication_commit=ARCHIVE_COMMIT, publication_matches_immutable_archive=True,
        certificate_proofs=certificate_proofs, saved_recipe_architecture_and_declared_target_checks=True,
        acquisition_checkpoint_proof=acquired_proof,
        hold_endpoint_checkpoint_proof=published['provenance_checkpoint'],
        observed_records_sha256=sha(records_path), source_files_verified=len(request['source']['files']),
        used_source_sha256={name: request['source']['files'][name] for name in
            ('benchmarks/toy_audit/api_images.py', 'particlegan/gan_loss.py')},
        recipe_id=request['candidate']['id'], candidate_revision=request['candidate_revision'],
        acquisition_parent_attempt=ACQUIRED_ID, hold_attempt=HOLD_ID,
        protocol='Enumerate five saved prior rows and declared five one-hot words. Public joint loss averages all25 independent real/fake combinations. CPU float32 reconstruction agrees with original CUDA observer within2e-5; float64 is a separate numerical diagnostic.',
        numerical_fixture={'metric':'maximum absolute functional-vs-saved reconstruction probability error',
            'threshold':2e-5,'status':'PASS'},
        gradient_obstruction_threshold=1e-6,
        original_grades_preserved=dict(word_smoke='PASS acquired at2501',word_hold='FAIL'),
        loss_path='Frozen WordFixture trains joint adversarial generator/encoder plus zero prior regularization; reconstruction is scored, not an explicit optimized reconstruction loss.',
        limitations='Endpoint forces omit minibatch realizations, future critic/optimizer changes and causal history. Reconstruction CE/clamp gradients are arithmetic diagnostics, not the trained objective or a tested repair. No target sampling or task regrading.',
        states=[analyze(acquired,records[0],loss,target),analyze(endpoint,records[-1],loss,target)])
    assert rng_before[0] == random.getstate()
    after = np.random.get_state()
    assert rng_before[1][0] == after[0] and np.array_equal(rng_before[1][1],after[1]) and rng_before[1][2:] == after[2:]
    assert torch.equal(rng_before[2],torch.get_rng_state())
    result['global_rng_states_unchanged'] = True
    result['reproducer_sha256'] = sha(Path(__file__))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
