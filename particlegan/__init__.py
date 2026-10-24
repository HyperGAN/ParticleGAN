"""Composable PyTorch primitives for GANs with learned particle priors.

Networks and devices belong to the caller; GANTrainer optionally owns updates.
Importing ParticleGAN disables multithreaded autograd scheduling on this thread.
"""
from .execution import disable_autograd_multithreading, serial_autograd

disable_autograd_multithreading()

from .autoencoder import ParticleEncoding, particle_ae, particle_vae
from .capabilities import prior_capabilities, prior_mechanisms
from .birth_death import ParticleRows, ScalarHeadFeatures
from .conditioning import UCD, ucd_labels, ucd_loss, ucd_scores
from .conditional_score import conditional_energy_score
from .continuous import DataDriftController
from .diffusion import DDGAN
from .discriminators import BatchDistanceDiscriminator, LinearSkipDiscriminator
from .gan_loss import GANLoss
from .joint_reconstruction import joint_reconstruction_loss
from .masked_mixture import (
    NeuralComponentImputer,
    masked_mixture_nll,
    masked_quantile_transport_loss,
)
from .particle_prior import GaussianPrior, MoGParticlePrior, ParticlePrior, calibrate_mog_sigma
from .priors import (
    AdditiveNormalManipulator,
    ManipulatedPrior,
    NormalNoiseManipulator,
    PriorManipulator,
    unwrap_prior,
)
from .policy import E22Policy, ServedModel, StepNoise, UpdatePolicy
from .recipes import (
    DEFAULT_RECIPE,
    NetworkLRTransition,
    Recipe,
    get_recipe,
    get_recipe_fields,
    learning_rate_scale,
    learning_rate_scales,
    scale_learning_rates,
)
from .routing import RoutedBatch, RoutedCandidate, RoutedExecution, RoutedRows
from .training import GANTrainer, InputNoise
from .tensorflow_adam import TensorFlowV1Adam
from .vicreg_loss import ParticleRegularizer
from .coulomb_prior import CoulombPriorRegularizer
from .sigreg import SIGReg
from .initial_mmd import InitialMMDRegularizer
from .low_precision import EmulatedLinear, emulate_linear_training_
from .bispf import (
    BISPFGates,
    BISPFCriticAdam,
    BISPFCriticPenalty,
    BISPFEngine,
    BISPFLossAdapter,
    BISPFStepRecord,
    ContinuousThermalCooler,
    SBNPPriorRegularizer,
    SinkhornRelativisticLoss,
    StratifiedBlueNoisePrior,
)
from .empf import (
    EMPFCriticAdam,
    EMPFCriticPenalty,
    EMPFPriorRegularizer,
    EMPFSinkhornRelativisticLoss,
    EMPFStepRecord,
)
from .rhbi import (
    RHBICriticAdam,
    RHBICriticPenalty,
    RHBIGANLoss,
    RHBILoss,
    RHBIStepRecord,
    bind_rhbi_record,
)
from .priors import (
    GhostFreeDeMixingManipulator,
    ManipulatedPrior,
    PriorManipulator,
    SimplexRoutingManipulator,
    SobolevWavefrontManipulator,
    SplatCarrierManipulator,
    StratifiedSamplingManipulator,
)
from .wavefront import (
    DifferentiableSplatCarrierBasis,
    GhostFreeSimplexMixer,
    ResidualConvBlock,
    SimplexWavefrontGenerator,
    SobolevGradFilter,
    WavefrontManifoldPrior,
    sobolev_filter_2d,
