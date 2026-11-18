# RHBI marginal transport in critic feature coordinates

The 288-update texture gate tests the prediction that adding the critic's learned representation to RHBI's distance restores diverse high-frequency output. The selected baseline is rhbi-v12. The image baseline EMPF and the BISPF reference have frozen positive [texture controls](../conv-texture32-tier1/README.md).

For the marginal generator loss, concatenate each raw sample with the input to the critic's single scalar linear readout. Center features on the real-batch mean, divide by its RMS Euclidean radius (floor 1e-6), and multiply by the existing detached RHBI horizon. Real feature statistics are detached. The same metric supplies real–fake cost and fake self-coupling. The critic objective, slack term, raw-sample horizon and thermal statistics, optimizer and learned SBNP prior follow v12. This opt-in recipe requires a scalar marginal critic with a single linear score head. Capture uses the actual critic forward and the generator derivative through its features.

## Frozen decision

One candidate, protocol seed 0. Compare with the parent on the complete 288-update texture gate. Only a full pass advances the candidate to the frozen rare-mass, sparse arc, patch-shift and saddle retention tasks. A failure on any of these stops external CIFAR evaluation. No retuning or seed search in this study. Results in this research view provide no ordinary qualification or default adoption credit.

The [probe](probe.py) restores the hash-bound baseline endpoint and measures the RHBI loss, critic gradients and nearest-neighbor distances for real textures and two paired damages against an independent real batch. [Probe results](probe.json).
