from .optim.adam import adamw as adamw
from .optim.aurora import (
    aurora as aurora,
    get_equinox_aurora_spec as get_equinox_aurora_spec,
    riemannian_aurora as riemannian_aurora,
)
from .optim.hyperball import (
    adamw_hyperball as adamw_hyperball,
    aurora_hyperball as aurora_hyperball,
    kron_hyperball as kron_hyperball,
    muon_hyperball as muon_hyperball,
    prism_hyperball as prism_hyperball,
    riemannian_aurora_hyperball as riemannian_aurora_hyperball,
)
from .optim.pion import pion as pion
from .optim.pion import scale_by_pion as scale_by_pion
from .optim.prism import (
    get_equinox_prism_spec as get_equinox_prism_spec,
    prism as prism,
)
from .optim.psgd import kron as kron
from .schedules.schedulefree import (
    schedule_free_adam as schedule_free_adam,
    schedule_free_aurora as schedule_free_aurora,
    schedule_free_eval_params as schedule_free_eval_params,
    schedule_free_kron as schedule_free_kron,
    schedule_free_prism as schedule_free_prism,
)
from .schedules.soda import (
    soda as soda,
    soda_adam as soda_adam,
    soda_kron as soda_kron,
    soda_muon as soda_muon,
    soda_prism as soda_prism,
)
from .schedules.wsd import wsd_schedule as wsd_schedule
from .utils import (
    apply_updates as apply_updates,
    apply_updates_prefix as apply_updates_prefix,
)

__version__ = "0.4.2"
