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
from .training import (
    GANTrainer,
    InputNoise,
    bind_formulation_records,
    configure_bf16_autocast,
    critic_g_step_mode,
    critic_source_context,
    float_output,
    uses_bf16_autocast,
    uses_channels_last,
    uses_simplicial_manifold_loss,
)
from .tensorflow_adam import TensorFlowV1Adam
from .vicreg_loss import ParticleRegularizer
from .coulomb_prior import CoulombPriorRegularizer
from .sigreg import SIGReg
from .initial_mmd import InitialMMDRegularizer
from .low_precision import EmulatedLinear, emulate_linear_training_, validate_emulated_extra_states
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
from ._t1_prior import T1PullbackPriorRegularizer
from .t1_coupling import (
    T1CouplingCriticAdam,
    T1CouplingCriticPenalty,
    T1CouplingGANLoss,
    T1CouplingLoss,
    T1CouplingStepRecord,
    bind_t1_coupling_record,
)
from .t2_critic import (
    T2SecantCriticAdam,
    T2SecantCriticPenalty,
    T2SecantGANLoss,
    T2SecantStepRecord,
    bind_t2_record,
)
from .t3_dynamics import (
    RGDTCriticAdam,
    RGDTCriticPenalty,
    RGDTGANLoss,
    RGDTPriorRegularizer,
    RGDTStepRecord,
    bind_rgdt_record,
)
from .t4_arch import (
    T4ArchPriorRegularizer,
    T4EquilibratedOptimizer,
)
from .t4_nash import (
    T4NashCriticAdam,
    T4NashCriticPenalty,
    T4NashGeneratorAdam,
    T4NashLoss,
    T4NashStepRecord,
    bind_t4_nash_record,
)
from .nash_v3 import (
    NashV3ConstraintGeneratorAdam,
    NashV3CriticAdam,
    NashV3CriticPenalty,
    NashV3GeneratorAdam,
    NashV3Loss,
    NashV3StepRecord,
    bind_nash_v3_record,
)
from .unified import (
    UnifiedCriticAdam,
    UnifiedCriticPenalty,
    UnifiedGANLoss,
    UnifiedPriorRegularizer,
    UnifiedStepRecord,
    bind_unified_record,
)
from .unified_v4 import (
    UnifiedV4CriticAdam,
    UnifiedV4CriticPenalty,
    UnifiedV4EquilibratedOptimizer,
    UnifiedV4GANLoss,
    UnifiedV4PriorRegularizer,
    UnifiedV4StepRecord,
    bind_unified_v4_record,
)
from .unified_v11 import (
    UnifiedV11CondFiberBank,
    UnifiedV11CriticAdam,
    UnifiedV11CriticPenalty,
    UnifiedV11EquilibratedOptimizer,
    UnifiedV11GANLoss,
    UnifiedV11PriorRegularizer,
    UnifiedV11StepRecord,
    bind_unified_v11_record,
    record_unified_v11_generator_probe,
)
from ._unified_v15 import (
    UnifiedV15CouplingLoss,
    UnifiedV15CriticAdam,
    UnifiedV15CriticPenalty,
    UnifiedV15GANLoss,
    UnifiedV15Optimizer,
    UnifiedV15PriorRegularizer,
    UnifiedV15StepRecord,
    bind_unified_v15_record,
    record_unified_v15_generator_probe,
)
from .cond_fiber import (
    CondFiberBank,
    CondFiberCriticAdam,
    CondFiberCriticPenalty,
    CondFiberGANLoss,
    CondFiberStepRecord,
    bind_cond_fiber_record,
)
from .minimal import (
    MinimalConstraintGeneratorOptimizer,
    MinimalCriticOptimizer,
    MinimalCriticPenalty,
    MinimalGeneratorOptimizer,
    MinimalLoss,
    MinimalPriorRegularizer,
    MinimalStepRecord,
    bind_minimal_record,
)
from .axiom import (
    AxiomCondFiberBank,
    AxiomCriticPenalty,
    AxiomGANLoss,
    AxiomOptimizer,
    AxiomPenaltyKernel,
    AxiomPriorRegularizer,
    AxiomStepRecord,
    AxiomTransportLoss,
    bind_axiom_record,
)
from . import init

__all__ = [
    "EmulatedLinear", "emulate_linear_training_", "validate_emulated_extra_states",
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
    "T1PullbackPriorRegularizer",
    "T1CouplingCriticAdam", "T1CouplingCriticPenalty", "T1CouplingGANLoss", "T1CouplingLoss", "T1CouplingStepRecord", "bind_t1_coupling_record",
    "T2SecantCriticAdam", "T2SecantCriticPenalty", "T2SecantGANLoss", "T2SecantStepRecord", "bind_t2_record",
    "RGDTCriticAdam", "RGDTCriticPenalty", "RGDTGANLoss", "RGDTPriorRegularizer", "RGDTStepRecord", "bind_rgdt_record",
    "T4ArchPriorRegularizer", "T4EquilibratedOptimizer",
    "T4NashCriticAdam", "T4NashCriticPenalty", "T4NashGeneratorAdam", "T4NashLoss", "T4NashStepRecord", "bind_t4_nash_record",
    "NashV3ConstraintGeneratorAdam", "NashV3CriticAdam", "NashV3CriticPenalty", "NashV3GeneratorAdam", "NashV3Loss", "NashV3StepRecord", "bind_nash_v3_record",
    "UnifiedCriticAdam", "UnifiedCriticPenalty", "UnifiedGANLoss", "UnifiedPriorRegularizer", "UnifiedStepRecord", "bind_unified_record",
    "UnifiedV4CriticAdam", "UnifiedV4CriticPenalty", "UnifiedV4EquilibratedOptimizer", "UnifiedV4GANLoss", "UnifiedV4PriorRegularizer", "UnifiedV4StepRecord", "bind_unified_v4_record",
    "UnifiedV11CondFiberBank", "UnifiedV11CriticAdam", "UnifiedV11CriticPenalty", "UnifiedV11EquilibratedOptimizer", "UnifiedV11GANLoss", "UnifiedV11PriorRegularizer", "UnifiedV11StepRecord", "bind_unified_v11_record", "record_unified_v11_generator_probe",
    "UnifiedV15CouplingLoss", "UnifiedV15CriticAdam", "UnifiedV15CriticPenalty", "UnifiedV15GANLoss", "UnifiedV15Optimizer", "UnifiedV15PriorRegularizer", "UnifiedV15StepRecord", "bind_unified_v15_record", "record_unified_v15_generator_probe",
    "CondFiberBank", "CondFiberCriticAdam", "CondFiberCriticPenalty", "CondFiberGANLoss", "CondFiberStepRecord", "bind_cond_fiber_record",
    "MinimalConstraintGeneratorOptimizer", "MinimalCriticOptimizer", "MinimalCriticPenalty", "MinimalGeneratorOptimizer", "MinimalLoss", "MinimalPriorRegularizer", "MinimalStepRecord", "bind_minimal_record",
    "AxiomCondFiberBank", "AxiomCriticPenalty", "AxiomGANLoss", "AxiomOptimizer", "AxiomPenaltyKernel", "AxiomPriorRegularizer", "AxiomStepRecord", "AxiomTransportLoss", "bind_axiom_record",
    "DDGAN", "UCD", "ucd_labels", "ucd_loss", "ucd_scores",
    "conditional_energy_score", "joint_reconstruction_loss",
    "masked_mixture_nll", "NeuralComponentImputer", "masked_quantile_transport_loss",
    "DEFAULT_RECIPE", "Recipe", "NetworkLRTransition", "DataDriftController", "get_recipe", "get_recipe_fields", "learning_rate_scale", "learning_rate_scales", "scale_learning_rates", "GANTrainer", "InputNoise",
    "bind_formulation_records", "configure_bf16_autocast", "critic_g_step_mode", "critic_source_context", "float_output", "uses_bf16_autocast", "uses_channels_last", "uses_simplicial_manifold_loss",
    "BatchDistanceDiscriminator", "LinearSkipDiscriminator",
]
