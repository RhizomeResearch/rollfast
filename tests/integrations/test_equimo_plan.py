from rollfast.integrations import equimo

from tests.finetune.helpers import tiny_plan


def test_equimo_integration_accepts_structural_plan_without_importing_equimo():
    bundle = equimo.adamw_from_equimo_plan(
        tiny_plan(),
        total_steps=10,
        schedule="constant",
        clip_global_norm=None,
    )

    assert bundle.report.trainable_params == 14
