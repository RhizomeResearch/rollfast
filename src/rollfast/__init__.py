from .optim.adam import adamw as adamw
from .optim.adam import scale_by_adam as scale_by_adam
from .optim.aurora import (
    AuroraDimensionNumbers as AuroraDimensionNumbers,
)
from .optim.aurora import (
    aurora as aurora,
)
from .optim.aurora import (
    get_equinox_aurora_spec as get_equinox_aurora_spec,
)
from .optim.aurora import (
    riemannian_aurora as riemannian_aurora,
)
from .optim.aurora import (
    scale_by_aurora as scale_by_aurora,
)
from .optim.aurora import (
    scale_by_riemannian_aurora as scale_by_riemannian_aurora,
)
from .optim.dimension_numbers import MatrixDimensionNumbers as MatrixDimensionNumbers
from .optim.dimension_numbers import WeightDimNumOrFn as WeightDimNumOrFn
from .optim.hyperball import (
    adamw_hyperball as adamw_hyperball,
)
from .optim.hyperball import (
    apply_hyperball as apply_hyperball,
)
from .optim.hyperball import (
    aurora_hyperball as aurora_hyperball,
)
from .optim.hyperball import (
    hyperball_riemannian_aurora as hyperball_riemannian_aurora,
)
from .optim.hyperball import (
    kron_hyperball as kron_hyperball,
)
from .optim.hyperball import (
    muon_hyperball as muon_hyperball,
)
from .optim.hyperball import (
    prism_hyperball as prism_hyperball,
)
from .optim.hyperball import (
    riemannian_aurora_hyperball as riemannian_aurora_hyperball,
)
from .optim.hyperball import (
    rmnp_hyperball as rmnp_hyperball,
)
from .optim.hyperball import (
    scale_by_hyperball as scale_by_hyperball,
)
from .optim.muon import MomentumAccumulator as MomentumAccumulator
from .optim.muon import MuonDimensionNumbers as MuonDimensionNumbers
from .optim.muon import MuonState as MuonState
from .optim.muon import muon as muon
from .optim.muon import (
    orthogonalize_via_newton_schulz as orthogonalize_via_newton_schulz,
)
from .optim.muon import polar_express_coeffs as polar_express_coeffs
from .optim.muon import resolve_ns_coeffs as resolve_ns_coeffs
from .optim.muon import scale_by_muon as scale_by_muon
from .optim.muon import scale_by_muon_shape as scale_by_muon_shape
from .optim.normuon import contramuon as contramuon
from .optim.normuon import contranormuon as contranormuon
from .optim.normuon import normuon as normuon
from .optim.normuon import scale_by_normuon as scale_by_normuon
from .optim.normuon import scale_by_normuon_shape as scale_by_normuon_shape
from .optim.orthogonalization import MUON_NS_COEFFS as MUON_NS_COEFFS
from .optim.orthogonalization import MuonNsCoeffs as MuonNsCoeffs
from .optim.orthogonalization import NsCoeffs as NsCoeffs
from .optim.pion import pion as pion
from .optim.pion import scale_by_pion as scale_by_pion
from .optim.prism import (
    PrismDimensionNumbers as PrismDimensionNumbers,
)
from .optim.prism import (
    get_equinox_prism_spec as get_equinox_prism_spec,
)
from .optim.prism import (
    prism as prism,
)
from .optim.prism import (
    scale_by_prism as scale_by_prism,
)
from .optim.psgd import GradClipMode as GradClipMode
from .optim.psgd import PreconditionerMode as PreconditionerMode
from .optim.psgd import kron as kron
from .optim.psgd import precond_update_prob_schedule as precond_update_prob_schedule
from .optim.psgd import scale_by_kron as scale_by_kron
from .optim.rmnp import rmnp as rmnp
from .optim.rmnp import scale_by_rmnp as scale_by_rmnp
from .optim.rmnp import scale_by_rmnp_shape as scale_by_rmnp_shape
from .optim.soda import (
    soda as soda,
)
from .optim.soda import (
    soda_adam as soda_adam,
)
from .optim.soda import (
    soda_kron as soda_kron,
)
from .optim.soda import (
    soda_muon as soda_muon,
)
from .optim.soda import (
    soda_prism as soda_prism,
)
from .optim.soda import (
    soda_rmnp as soda_rmnp,
)
from .optim.trasmuon import scale_by_trasmuon as scale_by_trasmuon
from .optim.trasmuon import trasmuon as trasmuon
from .schedules.schedulefree import (
    WeightingMode as WeightingMode,
)
from .schedules.schedulefree import (
    schedule_free as schedule_free,
)
from .schedules.schedulefree import (
    schedule_free_adam as schedule_free_adam,
)
from .schedules.schedulefree import (
    schedule_free_aurora as schedule_free_aurora,
)
from .schedules.schedulefree import (
    schedule_free_eval_params as schedule_free_eval_params,
)
from .schedules.schedulefree import (
    schedule_free_kron as schedule_free_kron,
)
from .schedules.schedulefree import (
    schedule_free_prism as schedule_free_prism,
)
from .schedules.wsd import power_decay_schedule as power_decay_schedule
from .schedules.wsd import wsd_schedule as wsd_schedule
from .utils import (
    apply_updates as apply_updates,
)
from .utils import (
    apply_updates_prefix as apply_updates_prefix,
)

__all__ = [
    "AuroraDimensionNumbers",
    "GradClipMode",
    "MUON_NS_COEFFS",
    "MatrixDimensionNumbers",
    "MomentumAccumulator",
    "MuonDimensionNumbers",
    "MuonNsCoeffs",
    "MuonState",
    "NsCoeffs",
    "PreconditionerMode",
    "PrismDimensionNumbers",
    "WeightDimNumOrFn",
    "WeightingMode",
    "__version__",
    "adamw",
    "adamw_hyperball",
    "apply_hyperball",
    "apply_updates",
    "apply_updates_prefix",
    "aurora",
    "aurora_hyperball",
    "contramuon",
    "contranormuon",
    "get_equinox_aurora_spec",
    "get_equinox_prism_spec",
    "hyperball_riemannian_aurora",
    "kron",
    "kron_hyperball",
    "muon",
    "muon_hyperball",
    "normuon",
    "orthogonalize_via_newton_schulz",
    "pion",
    "polar_express_coeffs",
    "power_decay_schedule",
    "precond_update_prob_schedule",
    "prism",
    "prism_hyperball",
    "resolve_ns_coeffs",
    "riemannian_aurora",
    "riemannian_aurora_hyperball",
    "rmnp",
    "rmnp_hyperball",
    "scale_by_adam",
    "scale_by_aurora",
    "scale_by_hyperball",
    "scale_by_kron",
    "scale_by_muon",
    "scale_by_muon_shape",
    "scale_by_normuon",
    "scale_by_normuon_shape",
    "scale_by_pion",
    "scale_by_prism",
    "scale_by_riemannian_aurora",
    "scale_by_rmnp",
    "scale_by_rmnp_shape",
    "scale_by_trasmuon",
    "schedule_free",
    "schedule_free_adam",
    "schedule_free_aurora",
    "schedule_free_eval_params",
    "schedule_free_kron",
    "schedule_free_prism",
    "soda",
    "soda_adam",
    "soda_kron",
    "soda_muon",
    "soda_prism",
    "soda_rmnp",
    "trasmuon",
    "wsd_schedule",
]

__version__ = "0.4.2"
