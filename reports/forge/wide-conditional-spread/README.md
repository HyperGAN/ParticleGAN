# Wide conditional spread

<a id="wide_conditional_spread"></a>

This Tier 2 stretch task asks whether a GAN learns which group varies along each axis. The two equally likely group labels select zero-mean Gaussian signals with SDs (0.5, 2) and (2, 0.5). The remaining 63 real coordinates are zero. Every generated coordinate comes from the network. A separate label accompanies each generator, discriminator and gradient penalty call.

The frozen task uses 1,200 updates and 24 observations with five terminal passes required. Each observation draws 2,048 independent bags of 32 per group using the live generator and public learned MoG prior. For each signal coordinate, the 10th and 90th percentiles of generated bag means and sample SDs are evaluated on independent target bags. The interval score is width plus ten times the distance outside the interval. The reference score uses another independent sample from the target law. The task gates the worst group/coordinate errors and the summed mean squared output on the 63 inactive coordinates. Training uses protocol seed 0 and its own random streams. The fixed evaluation panels are shared across observations.

The initial adapter binds the public EMPF components with explicit continuous inputs and a penalty on each update. A future comparison can change the conditional discrepancy within EMPF and retain the identical task. The CPU observer regression in [#555](https://github.com/HyperGAN/ParticleGAN/issues/555) tests the mechanism. This task scores generated samples. [Task declaration](../../../configs/forge/tasks/wide_conditional_spread.json)
