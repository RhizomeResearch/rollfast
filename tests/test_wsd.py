from typing import Any, cast

import pytest

from rollfast.schedules.wsd import power_decay_schedule, wsd_schedule


def test_wsd_schedule():
    sched = wsd_schedule(
        peak_lr=0.01, total_steps=100, warmup_fraction=0.1, decay_fraction=0.1
    )
    lr_0 = cast(float, sched(0))
    lr_50 = cast(float, sched(50))
    lr_99 = cast(float, sched(99))
    assert lr_0 < 0.01
    assert lr_50 == 0.01
    assert lr_99 < 0.01


def test_wsd_schedule_supports_cosine_decay_to_ratio():
    sched = wsd_schedule(
        peak_lr=1.0,
        total_steps=100,
        warmup_fraction=0.0,
        decay_fraction=0.2,
        decay_shape="cosine",
        final_lr_ratio=0.1,
    )

    assert cast(float, sched(50)) == 1.0
    assert cast(float, sched(99)) == pytest.approx(0.1)


def test_wsd_schedule_power_decay_shape_changes_mid_cooldown():
    linear = wsd_schedule(
        peak_lr=1.0,
        total_steps=100,
        warmup_fraction=0.0,
        decay_fraction=0.2,
        decay_shape="linear",
    )
    power = wsd_schedule(
        peak_lr=1.0,
        total_steps=100,
        warmup_fraction=0.0,
        decay_fraction=0.2,
        decay_shape="power",
        decay_power=2.0,
    )

    assert cast(float, power(90)) > cast(float, linear(90))


def test_wsd_schedule_rejects_invalid_shape():
    invalid_shape = cast(Any, "bad")
    with pytest.raises(ValueError, match="decay_shape"):
        wsd_schedule(
            peak_lr=1.0,
            total_steps=100,
            warmup_fraction=0.0,
            decay_fraction=0.2,
            decay_shape=invalid_shape,  # type: ignore[arg-type]
        )


def test_power_decay_schedule():
    sched = power_decay_schedule(peak_lr=1.0, total_steps=101, power=2.0)

    assert cast(float, sched(0)) == 1.0
    assert cast(float, sched(50)) == pytest.approx(0.25)
    assert cast(float, sched(100)) == 0.0


def test_power_decay_schedule_with_warmup_and_final_ratio():
    sched = power_decay_schedule(
        peak_lr=1.0,
        total_steps=101,
        power=1.0,
        warmup_fraction=0.1,
        final_lr_ratio=0.2,
    )

    assert cast(float, sched(0)) < 1.0
    assert cast(float, sched(10)) == 1.0
    assert cast(float, sched(100)) == pytest.approx(0.2)
