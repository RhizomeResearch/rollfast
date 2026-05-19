import rollfast
import rollfast.schedules as schedules
from rollfast.optim.psgd import (
    GradClipMode,
    PreconditionerMode,
    precond_update_prob_schedule,
)
from rollfast.optim.adam import scale_by_adam
from rollfast.optim.psgd import scale_by_kron
from rollfast.schedules.schedulefree import WeightingMode


def test_root_exports_direct_adam_and_kron_transforms():
    assert rollfast.scale_by_adam is scale_by_adam
    assert rollfast.scale_by_kron is scale_by_kron


def test_root_and_schedule_exports_public_config_helpers():
    assert rollfast.PreconditionerMode is PreconditionerMode
    assert rollfast.GradClipMode is GradClipMode
    assert rollfast.precond_update_prob_schedule is precond_update_prob_schedule
    assert rollfast.WeightingMode is WeightingMode
    assert schedules.PreconditionerMode is PreconditionerMode
    assert schedules.GradClipMode is GradClipMode
    assert schedules.precond_update_prob_schedule is precond_update_prob_schedule
    assert schedules.WeightingMode is WeightingMode
