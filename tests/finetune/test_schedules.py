import pytest

import rollfast.finetune as rfft


def test_warmup_cosine_schedule_boundaries():
    config = rfft.ScheduleConfig(
        kind="warmup_cosine",
        total_steps=10,
        warmup_steps=2,
        end_lr_ratio=0.1,
    )
    schedule = rfft.build_schedule(config, peak_lr=1.0)

    assert float(schedule(0)) == pytest.approx(0.5)
    assert float(schedule(1)) == pytest.approx(1.0)
    assert float(schedule(9)) == pytest.approx(0.1)


def test_constant_schedule_does_not_require_total_steps():
    schedule = rfft.build_schedule(rfft.ScheduleConfig(kind="constant"), peak_lr=0.2)

    assert float(schedule(0)) == pytest.approx(0.2)
    assert float(schedule(10_000)) == pytest.approx(0.2)


def test_wsd_schedule_config_explicit_steps_override_fractions():
    config = rfft.ScheduleConfig(
        kind="wsd",
        total_steps=10,
        warmup_steps=2,
        warmup_fraction=0.8,
        decay_steps=3,
        decay_fraction=0.0,
    )
    schedule = rfft.build_schedule(config, peak_lr=1.0)

    assert float(schedule(0)) == pytest.approx(0.5)
    assert float(schedule(1)) == pytest.approx(1.0)
    assert float(schedule(7)) == pytest.approx(2.0 / 3.0)
    assert float(schedule(9)) == pytest.approx(0.0)


def test_preview_uses_schedule_factory():
    preview = rfft.preview_schedule(
        rfft.ScheduleConfig(kind="linear", total_steps=5, warmup_steps=0),
        peak_lr=1.0,
    )

    assert preview[0].step == 0
    assert preview[-1].step == 4
    assert preview[-1].value == pytest.approx(0.01)
