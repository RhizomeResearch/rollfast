from typing import cast

import pytest

from rollfast.schedules.wsd import wsd_schedule


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


def test_wsd_schedule_honors_explicit_step_counts():
    sched = wsd_schedule(
        peak_lr=1.0,
        total_steps=10,
        warmup_steps=2,
        warmup_fraction=0.8,
        decay_steps=3,
        decay_fraction=0.0,
    )

    assert float(sched(0)) == pytest.approx(0.5)
    assert float(sched(1)) == pytest.approx(1.0)
    assert float(sched(2)) == pytest.approx(1.0)
    assert float(sched(6)) == pytest.approx(1.0)
    assert float(sched(7)) == pytest.approx(2.0 / 3.0)
    assert float(sched(9)) == pytest.approx(0.0)


def test_wsd_schedule_rejects_overlapping_regions():
    with pytest.raises(ValueError, match="must not exceed total_steps"):
        wsd_schedule(
            peak_lr=1.0,
            total_steps=10,
            warmup_steps=8,
            decay_steps=3,
        )
