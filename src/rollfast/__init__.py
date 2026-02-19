from .optim.adam import adamw as adamw
from .optim.prism import (
    get_equinox_prism_spec as get_equinox_prism_spec,
    prism as prism,
)
from .optim.psgd import kron as kron
from .optim.rmsprop import rmsprop as rmsprop
from .schedules.schedulefree import (
    schedule_free_eval_params as schedule_free_eval_params,
    schedule_free_kron as schedule_free_kron,
    schedule_free_prism as schedule_free_prism,
)
from .schedules.wsd import wsd_schedule as wsd_schedule

__version__ = "0.1.2"
