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
from . import init

__all__ = [
    "EmulatedLinear", "emulate_linear_training_",
    "disable_autograd_multithreading", "serial_autograd",
    "init",
    "TensorFlowV1Adam",
    "prior_capabilities", "prior_mechanisms",
    "ParticleEncoding", "particle_ae", "particle_vae",
    "ParticleRows", "ScalarHeadFeatures",
    "RoutedBatch", "RoutedCandidate", "RoutedExecution", "RoutedRows",
    "E22Policy", "UpdatePolicy", "ServedModel", "StepNoise",
    "ParticlePrior", "MoGParticlePrior", "calibrate_mog_sigma", "GaussianPrior", "GANLoss",
    "NormalNoiseManipulator", "AdditiveNormalManipulator", "ManipulatedPrior", "PriorManipulator", "unwrap_prior",
    "CoulombPriorRegularizer", "ParticleRegularizer", "InitialMMDRegularizer", "SIGReg",
    "StratifiedBlueNoisePrior", "SBNPPriorRegularizer", "ContinuousThermalCooler", "BISPFEngine", "BISPFLossAdapter",
    "BISPFStepRecord", "BISPFCriticAdam", "BISPFCriticPenalty", "SinkhornRelativisticLoss", "BISPFGates",
    "EMPFStepRecord", "EMPFCriticAdam", "EMPFCriticPenalty", "EMPFPriorRegularizer", "EMPFSinkhornRelativisticLoss",
    "RHBICriticAdam", "RHBICriticPenalty", "RHBIGANLoss", "RHBILoss", "RHBIStepRecord", "bind_rhbi_record",
    "DDGAN", "UCD", "ucd_labels", "ucd_loss", "ucd_scores",
    "conditional_energy_score", "joint_reconstruction_loss",
    "masked_mixture_nll", "NeuralComponentImputer", "masked_quantile_transport_loss",
    "DEFAULT_RECIPE", "Recipe", "NetworkLRTransition", "DataDriftController", "get_recipe", "get_recipe_fields", "learning_rate_scale", "learning_rate_scales", "scale_learning_rates", "GANTrainer", "InputNoise",
    "BatchDistanceDiscriminator", "LinearSkipDiscriminator",
]
