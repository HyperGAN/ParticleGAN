# Required global conditioning: a public routed-E22 diagnostic

The fixed scientific run was a valid negative result: G64 did not meet either10% improvement gate. This standalone fixture compares D16/G16 with D16/G64 for200 updates each. It uses only Torch and ParticleGAN's public recipe, initializer, optimizers, routed policy, native lifecycle and checkpoint APIs. The generator trains with the paired residual GAN loss; squared error is an evaluation metric only.

See the [negative result and limitations](routed_remote_conditioning_results_20261002.md). From a matching ParticleGAN checkout with Torch and pytest installed:

```sh
timeout 60s env PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u examples/routed_remote_conditioning.py --output /tmp/routed-remote-conditioning
python -m pytest -q tests/test_routed_remote_conditioning_public.py tests/test_routed_remote_conditioning_algebra.py
```

Use a new output directory for every execution. The scientific command exits 0 only when every frozen scientific, source, lifecycle and finite-health gate passes, 2 for a completed metric failure and 1 for an invalid attempt. A timeout is a nonzero failure. It writes dense per-step traces, both complete checkpoints and `result.json`. There is one fixed seed protocol and no automatic retries, best-checkpoint selection or parameter search.

## Fixture and units

Each paired observation contains the same 16×16 content field and time. A ±0.2 marker occupies the upper-left 2×2 pixels of a second observed channel. The lower-right 4×4 evaluation patch lies outside the direct generator CNN's radius-4 receptive field of that marker. A global pooled source/time encoder can carry the marker through the public coupled key/value particle bank to a LayerNorm(code, eps=1e-3) FiLM conditioner. The direct spatial source path stays available.

Labels come from one frozen same-family public-initialized teacher (generator seed 4). Its additive and multiplicative code columns are retained. The teacher's prescribed marker encoder uses public KEEP on its direct Linear parameter owners: the pooled marker is±.003125 and fixed scaling320 maps it to±1 before SiLU; no seed or teacher selection occurs. There are 128 fit content/time pairs, 4 guard pairs and 32 held pairs, all independently generated from named seeds 44 and 45. Labels are deterministic given the observed source and time; there is no hidden nuisance mode.

Before training, fit labels alone define channel means and population standard deviations:

$$
\mu_c=\mathrm{mean}_{i,h,w} y^{fit}_{ichw},\qquad
s_c=\max\left(\sqrt{\mathrm{mean}_{i,h,w}(y^{fit}_{ichw}-\mu_c)^2},10^{-4}\right).
$$

Every label and every student raw output use the same fixed transformation $\widetilde y_c=(y_c-\mu_c)/s_c$ before critic scores, penalties and GAN losses. Means/stds are generator buffers installed before policy/EMA construction and preserved in checkpoints. Held rows never determine calibration. Thus output noise σ=1.3 is expressed in normalized target units, matching the real task's convention. Raw teacher targets, calibration and normalized targets remain hash-bound evidence.

For two paired held teacher labels on the evaluation patch, the best predictor unable to distinguish the remote marker has squared-error lower bound

$$
V=\frac14\mathrm{mean}_{pair,c,h,w}(\widetilde y^+_{c,h,w}-\widetilde y^-_{c,h,w})^2.
$$

The fixed teacher must have $V\ge10^{-8}$. Failure refuses the fixture; it does not change the teacher, marker, mask, seed or threshold. Clean teacher capacity does **not** prove perfect capacity under native DV12 perturbations: information routed through a noisy code can have a conditioning tradeoff. This fixture leaves that native law intact and reports its outcome without attributing a negative result to a GAN or ParticleGAN bug.

## Game and single candidate factor

Let $e=\widetilde G(x,t)-\widetilde y$, $n=\sigma\epsilon$. The public loss is

$$
L_D=\mathrm{mean}\mathrm{softplus}(D(n+e)-D(n))+\mathrm{KA2},
$$

with G detached for D, followed by one true scalar antithetic G loss,

$$
L_G=\tfrac12\sum_{a\in\lbrace-1,+1\rbrace}\mathrm{mean}\mathrm{softplus}
\big(\mathrm{stopgrad}D(an)-D(an+e)\big).
$$

There is one D16 backward/step and one G backward/step per native lifecycle. Native DV12 perturbation, learned output-noise ownership, KA2's independent EMA, table damping/testing, routed restructuring, settling and averaged serving remain active. All optimizers use β=(0,.999). The generator group receives the same quarter factor once after public begin_step; encoder/table/noise rates remain native. Critic global score gain is 0, local gain .0625 and the free-sign quadratic head starts at zero with public KEEP and learns only through D. Score-inactive global critic features are retained for native routing controls.

G16 uses the exact D16 example IDs after the D update. G64 extends those IDs with exactly 48 caller draws. Both arms consume the same common64 IDs and Gaussian panel, with the unused48 declared as G16 shadow draws. Native private DV12 draw identity across batch sizes is not assumed. G batch size is the only candidate factor. This is the late-common profile initialized fresh, not a reproduction of a mature real winner.

## Frozen pass/fail and limits

At both update 100 and update 200, each arm's held normalized patch error must be at most 90% of its own initial error, and G64 must be at most 90% of G16. At update 200, each arm must also have error at most 50% of V, demonstrating use of the required conditional path beyond a marker-blind optimum. All errors and states must remain finite. All 400 updates must finish within the fixed combined 60-second deadline; active gradients, unique live ownership, dense128 table gradients, immutable frozen hosts, independent KA2 EMA, positive organic native evaluation/probe counts and matched caller histories/states are checked. Accepted moves/splits are reported without a minimum.

The four existing public CPU contracts use handcrafted unit labels, at most one genuine native update per unit owner and no production teacher fixture. They check the remote local-view construction/direct receptive field, prescribed public initialization, exact output calibration and averaged buffer ownership, caller coupling, full public step/finite optimizer state, complete checkpoint/resume and strict metric failures.

The actual Nova→Qwen raw64 counterfactual showed that collapsing the context-varying full bank to its mean code raised error by about 6.25%; that establishes useful bank dependence on one saved cohort. It does not isolate FiLM alone, establish a convergence defect or predict LPIPS transfer. The real frozen transformer prefix can already carry global context. This teacher deliberately makes a routed global distinction necessary, rather than assuming the real learned bridge has the same receptive-field bottleneck.

This V2 replaces an unrun V1 proposal because V1 lacked target-unit normalization and declared public KEEP on unsupported dotted parent names. V1 source and its observed plumbing failure are preserved externally. No production teacher evaluation or scientific training was performed during V2 source authoring. The root subsequently ran the sole frozen400-update campaign in16.41s: teacher and native-health qualifications passed; four fixed metric gates failed.

## Descriptive paired readout and publication provenance

The published evaluator reuses its existing clean prediction and label tensors to report paired midpoint and half-contrast error. With m=(y⁺+y⁻)/2 and h=(y⁺−y⁻)/2, pair MSE equals midpoint MSE plus half-contrast MSE. It reports contrast error/V, α=mean(predicted_h×target_h)/V and predicted contrast power/V. These fields are observations, with no extra forwards, model updates or changed campaign gates. One pure algebra CPU contract distinguishes perfect contrast with a large shared offset from a marker-blind repeated midpoint.

The original400-update V2 campaign and its later zero-update saved-endpoint decomposition are separately identified in the compact result. The latter found contrast error/V near1 for both arms, while midpoint error dominated their total error. That fixed-budget outcome reflects a teacher with prescribed pooled-marker scale320 and a publicly initialized learner; it is not a demonstrated equivalent conditioning mismatch in Nova→Qwen.

The executed driver and protocol are archived byte-for-byte. The published script loads the already-merged PR236 public helper in its own module namespace: its bytes equal the original V2 helper exactly, and the original example's globals stay separate. Package defaults are untouched. The four original public contracts remain byte-identical; only the pure algebra contract is new. No scientific training was rerun for publication.
