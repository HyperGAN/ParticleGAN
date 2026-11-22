No E22 configuration qualifies across the eight required API questions in this
round. All four configurations remain in the denominator: three stopped after a
study failure and one after an incomplete study hold. Five actual full-budget
training runs produced five nine-frame GIFs. Their original gates are two PASS
and three FAIL; the additional study gate is one PASS, three FAIL and one
INCOMPLETE. The other 27 configuration/case cells remain UNKNOWN.

The [compact receipt](e22-policy-readout.json) maps all 32 cells, preserves the
full effective recipes and binds the external source, runtime, raw receipts,
cloud arrays, final checkpoints and GIFs. Its frozen execution source is
`eb2d77fbd776eab0d2ce5e2550ab9cdfdb6c1162`. Independent readout verification
checked all five complete receipts, including the original PASS whose study
status is INCOMPLETE; it checked the recorded metrics against every retained
cloud, all artifact hashes/frame counts, original budgets, seed, effective
Recipe, exact source/runtime and finite learned state. The readout performed
zero model updates and zero model sampling. Raw metric streams remain in the
external archive.

The common grid is learning rate `{0.006375, 0.0085}` × prior-rate multiplier
`{1, 2}`. Each whole configuration retains the same original seed 24002, models,
data law, public sampler, gate and horizon on every required question. A
configuration stops at its first study non-pass. The scope is a provisional
eight-case API policy screen; Forge MoG, all-toy qualification, calibration and
reserved robustness are outside it. E22 paid **131.94360883370973 seconds** from
its 5,400-second allowance, including process/export overhead; durable raw
acquisition time totals 110.59298101090826 seconds. All full attempts completed
inside their fixed 180-second case caps. CUDA logical device 0 was an RTX A6000
with Python 3.12.13, Torch 2.13.0+cu126 and one Torch thread. These times are cost
records, with no speed ranking.

The first question asks whether the exact transpose-width-12 host recovers both
center-patch intensity templates, 0.35 and 0.85, with correct brightness and
balanced output probability. It uses 32 particles, latent dimension 8, 600
updates and 1,024 held-out served images per observation. Quality means nearest
template RMSE ≤ 0.06; HQ, accepted-template mass and both mode counts are gated.
Detecting two nearest templates alone cannot excuse blurred/off-template mass.

The second question asks whether the public host fits both equal-mass Gaussians
at means `(-1, 0)` and `(1, 0)`, each covariance `0.0625 I`, including their
observable within-mode spread. It uses 256 particles, latent dimension 4, 1,200
updates and 4,096 held-out served points per observation. The fixed coarse mass,
HQ and covariance bounds are supplemented by maximum KS ≤ 0.06 across 32
analytic projected Gaussian-mixture CDFs. This is a finite projection test,
without a claim that it establishes equality of the full two-dimensional law.

Both questions retain active public latent perturbation. `output_noise=False`
only disables additive output noise; it does not make these unperturbed
finite-atom serving laws. Held-out measurements are observers, not training
signals. There are 24 post-update metric observations and nine media frames per
run, so changing movie density cannot change the gate. The original gate needs
the final five metric observations to pass. The study additionally confirms
the first five consecutive passing observations and requires every later
observation to pass, with at least five observations after confirmation.

| Learning rate / prior multiplier | Case | Full original gate | Study gate | What the retained evidence shows |
| --- | --- | --- | --- | --- |
| 0.006375 / 1 | Intensity, 600 updates | PASS | INCOMPLETE | Confirmation at 500 leaves only four hold checks; all four pass. The declared 600-update budget was completed. No failure or timeout is substituted for the missing fifth hold. |
| 0.006375 / 2 | Intensity, 600 updates | PASS | PASS | Confirmation at 350, followed by ten passing hold checks. Final HQ 1, rejected mass 0, template TV 0.0146484. |
| 0.006375 / 2 | Two broad Gaussians, 1,200 updates | FAIL | FAIL | Confirmation at 950, then only two of five passing hold checks. Final projection KS 0.0721952 exceeds 0.06; final coarse mass and covariance gates pass. |
| 0.0085 / 1 | Intensity, 600 updates | FAIL | FAIL | No five-check acquisition window. Final HQ 0.736328 and rejected template mass 0.263672 fail despite both modes being detected. |
| 0.0085 / 2 | Intensity, 600 updates | FAIL | FAIL | Confirmation at 450, support failure at 525, then recovery. Final instantaneous metrics pass, but the final-five original gate and uninterrupted study hold fail. |

For the broad-Gaussian endpoint, nearest-mean mass is 0.489502 / 0.510498. The
two component covariance errors are 0.202032 / 0.214016, with core minimum
eigenvalue ratios 0.898128 / 0.934998. The failing analytic projection is
129.375°: at projected coordinate 0.638766, the target CDF is 0.753488 and the
empirical CDF immediately after the sample is 0.825684. Component mean biases
are `(0.0166212, -0.0577030)` and `(0.0237434, 0.0307585)`. A descriptive
transformation of this saved cloud that subtracts each nearest-mean group's
bias reduces maximum projected KS to 0.0403973, preserving those groups' masses
and covariances. This isolates a location contribution in the observed cloud;
it does not repair the trained generator or identify an optimizer mechanism.
The original FAIL is unchanged. Its best retained KS is 0.0291153 at update
1,150; that passing snapshot does not erase the earlier hold failures or the
failed endpoint.

For intensity at 0.0085/prior1, **270 of 1,024** terminal images lie outside
both RMSE neighborhoods: 222 nearest the dim template and 48 nearest the bright
template. Both accepted masses remain large enough for a two-mode count, so the
failure is off-template output quality/mass. At 0.0085/prior2's update-525
failure, **343 of 1,024** images are rejected, including 278 nearest the bright
template; its accepted bright mass falls to 0.174805 and fails the mode bound.
The terminal run recovers to 1,023 accepted images, but the frozen sustained
gate correctly retains FAIL. These are measured failure signatures. No
particular optimization or policy cause has been established by this readout.

The three native 100-mode cases, unequal-mass and anisotropic vector cases, and
the four-bar image case are UNKNOWN for every configuration: none reached that
part of its original eight-case schedule. In particular, there is no native
feature-cell training evidence here and no whole-configuration default
recommendation. No incomplete hold or failed endpoint was extended or repeated.

The actual movies are the unchanged original training GIFs:

- [Intensity 0.006375 / 1 — original PASS; study INCOMPLETE](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/policy-family-media/policy-family-defaults-round1-cli-recovery-v2--eb2d77fbd776/e22/da7513723838--image-develop-img_intensity2-source-transpose12.gif).
- [Intensity 0.006375 / 2 — original and study PASS](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/policy-family-media/policy-family-defaults-round1-cli-recovery-v2--eb2d77fbd776/e22/0acab20ca12f--image-develop-img_intensity2-source-transpose12.gif).
- [Two broad Gaussians 0.006375 / 2 — original and study FAIL](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/policy-family-media/policy-family-defaults-round1-cli-recovery-v2--eb2d77fbd776/e22/0acab20ca12f--api-vector-two-broad.gif).
- [Intensity 0.0085 / 1 — original and study FAIL](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/policy-family-media/policy-family-defaults-round1-cli-recovery-v2--eb2d77fbd776/e22/5d4d276493d6--image-develop-img_intensity2-source-transpose12.gif).
- [Intensity 0.0085 / 2 — original and study FAIL, recovered final metric](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/policy-family-media/policy-family-defaults-round1-cli-recovery-v2--eb2d77fbd776/e22/65e42a03035e--image-develop-img_intensity2-source-transpose12.gif).

A separate [reviewed broad-Gaussian display](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/policy-family-media/policy-family-defaults-round1-cli-recovery-v2--eb2d77fbd776/e22/0acab20ca12f--api-vector-two-broad-reviewed.gif) includes the analytic projection-CDF bound in its label. It presents the same retained updates; the original movie above is unchanged.

First, middle and terminal frames from all five movies were visually reviewed.
The intensity movies show ordered target templates above actual draws; the
vector movie shows the full cloud, mode mass and local width. Their permanent
“Default test” label reports the original numeric verdict, not the extra study
hold. The broad-vector movie's abbreviated metric text omits the CDF bound;
the linked compact receipt supplies the exact failure. All media step indices,
decoded frame counts, dimensions and SHA-256 identities are recorded. Raw
receipts, metrics and models were unchanged.
