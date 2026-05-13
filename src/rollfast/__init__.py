from .optim.adam import adamw as adamw
from .optim.aurora import (
    aurora as aurora,
    get_equinox_aurora_spec as get_equinox_aurora_spec,
    riemannian_aurora as riemannian_aurora,
)
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
from .schedules.wsd import wsd_schedule as wsd_schedule
from .utils import (
    apply_updates as apply_updates,
    apply_updates_prefix as apply_updates_prefix,
)

__version__ = "0.4.0"
