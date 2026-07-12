from typing import Any, cast

import pytest

import rollfast.schedules.wsd as wsd_module
from rollfast.schedules.wsd import power_decay_schedule, wsd_schedule


def test_wsd_module_all_lists_public_schedules():
    assert "wsd_schedule" in wsd_module.__all__
    assert "power_decay_schedule" in wsd_module.__all__
    assert "_make_wsd_schedule_pair" not in wsd_module.__all__


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


def test_wsd_schedule_honors_explicit_step_counts():
    sched = wsd_schedule(
        peak_lr=1.0,
        total_steps=10,
        warmup_steps=2,
        warmup_fraction=0.8,
        decay_steps=3,
        decay_fraction=0.0,
    )

    assert float(sched(0)) < 1.0
    assert float(sched(1)) <= 1.0
    assert float(sched(2)) == pytest.approx(1.0)
    assert float(sched(6)) == pytest.approx(1.0)
    assert float(sched(7)) == pytest.approx(1.0)
    assert float(sched(8)) < 1.0
    assert float(sched(9)) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("decay_steps", "expected_tail"),
    [
        (0, (1.0, 1.0, 1.0)),
        (1, (1.0, 1.0, 0.2)),
        (2, (1.0, 1.0, 0.2)),
        (3, (1.0, 0.6, 0.2)),
    ],
)
def test_wsd_schedule_cooldown_boundary_truth_table(decay_steps, expected_tail):
    sched = wsd_schedule(
        peak_lr=1.0,
        total_steps=6,
        warmup_steps=0,
        decay_steps=decay_steps,
        final_lr_ratio=0.2,
    )

    assert tuple(float(sched(count)) for count in range(3, 6)) == pytest.approx(
        expected_tail
    )


@pytest.mark.parametrize("decay_shape", ["linear", "cosine", "power", "exponential"])
def test_wsd_schedule_one_step_fraction_reaches_final_ratio(decay_shape):
    sched = wsd_schedule(
        peak_lr=2.0,
        total_steps=10,
        warmup_steps=0,
        decay_fraction=0.1,
        decay_shape=decay_shape,
        final_lr_ratio=0.25,
    )

    assert float(sched(8)) == pytest.approx(2.0)
    assert float(sched(9)) == pytest.approx(0.5)
    assert float(sched(10)) == pytest.approx(0.5)


def test_wsd_schedule_accepts_end_lr_ratio_alias():
    sched = wsd_schedule(
        peak_lr=1.0,
        total_steps=10,
        warmup_fraction=0.0,
        decay_fraction=0.2,
        end_lr_ratio=0.2,
    )

    assert float(sched(9)) == pytest.approx(0.2)


def test_wsd_schedule_documents_inclusive_boundaries():
    sched = wsd_schedule(
        peak_lr=1.0,
        total_steps=20,
        warmup_fraction=0.1,
        decay_fraction=0.2,
        final_lr_ratio=0.1,
    )

    assert cast(float, sched(1)) < 1.0
    assert cast(float, sched(2)) == 1.0
    assert cast(float, sched(16)) == 1.0
    assert cast(float, sched(17)) < 1.0
    assert cast(float, sched(19)) == pytest.approx(0.1)


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


def test_wsd_schedule_rejects_overlapping_explicit_regions():
    with pytest.raises(ValueError, match="must not exceed total_steps"):
        wsd_schedule(
            peak_lr=1.0,
            total_steps=10,
            warmup_steps=8,
            decay_steps=3,
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
