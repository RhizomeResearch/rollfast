import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rollfast.finetune as rfft
from rollfast.schedules.wsd import wsd_schedule


@pytest.mark.parametrize("kind", ["warmup_cosine", "linear", "polynomial"])
@pytest.mark.parametrize(("total", "warmup"), [(1, 0), (2, 0), (7, 2), (4, 9)])
def test_decay_schedules_match_reference_at_and_beyond_boundaries(kind, total, warmup):
    peak, end, power = 0.3, 0.06, 1.7
    schedule = rfft.build_schedule(
        rfft.ScheduleConfig(
            kind=kind,
            total_steps=total,
            warmup_steps=warmup,
            end_lr_ratio=end / peak,
            power=power,
        ),
        peak_lr=peak,
    )
    steps = list(range(-1, total + 3))
    expected = []
    warmup = min(warmup, total)
    for count in steps:
        if total <= 1:
            value = peak
        elif count < warmup:
            value = peak * (count + 1) / warmup if warmup else peak
        else:
            progress = min(max((count - warmup) / max(total - warmup - 1, 1), 0), 1)
            remaining = {
                "warmup_cosine": (1 + math.cos(math.pi * progress)) / 2,
                "linear": 1 - progress,
                "polynomial": (1 - progress) ** power,
            }[kind]
            value = end + (peak - end) * remaining
        expected.append(value)
    for evaluate in (jax.vmap(schedule), jax.jit(jax.vmap(schedule))):
        np.testing.assert_allclose(evaluate(jnp.asarray(steps)), expected, rtol=2e-6)


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


@pytest.mark.parametrize(
    "shapes",
    (
        {},
        {"warmup_shape": "cosine", "decay_shape": "cosine"},
        {
            "warmup_shape": "power",
            "warmup_power": 2.0,
            "decay_shape": "power",
            "decay_power": 3.0,
        },
        {"warmup_shape": "exponential", "exponential_rate": 2.5},
        {"decay_shape": "exponential", "exponential_rate": -2.5},
    ),
    ids=("defaults", "cosine", "power", "exponential_warmup", "exponential_decay"),
)
def test_wsd_schedule_config_matches_wsd_schedule_at_every_slot(shapes):
    peak, total = 0.4, 40
    config = rfft.ScheduleConfig(
        kind="wsd",
        total_steps=total,
        warmup_steps=6,
        decay_steps=9,
        end_lr_ratio=0.05,
        **shapes,
    )
    schedule = rfft.build_schedule(config, peak_lr=peak)
    reference = wsd_schedule(
        peak,
        total,
        warmup_steps=6,
        decay_steps=9,
        final_lr_ratio=0.05,
        **shapes,
    )
    steps = jnp.arange(-1, total + 2)

    for evaluate in (jax.vmap, lambda fn: jax.jit(jax.vmap(fn))):
        np.testing.assert_array_equal(
            evaluate(schedule)(steps),
            evaluate(reference)(steps),
        )


def test_wsd_schedule_config_expresses_decay_below_peak_on_first_slot():
    peak, floor, slots, warmup, decay = 0.8, 0.08, 50, 5, 12
    schedule = rfft.build_schedule(
        rfft.ScheduleConfig(
            kind="wsd",
            total_steps=slots,
            warmup_steps=warmup - 1,
            decay_steps=decay + 1,
            decay_shape="cosine",
            end_lr_ratio=floor / peak,
        ),
        peak_lr=peak,
    )

    assert float(schedule(warmup - 1)) == pytest.approx(peak)
    for slot in range(slots - decay, slots):
        progress = (slot - (slots - decay) + 1) / decay
        expected = floor + (peak - floor) / 2 * (1 + math.cos(math.pi * progress))
        assert float(schedule(slot)) == pytest.approx(expected, rel=1e-6)


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

    assert float(schedule(0)) == pytest.approx(1.0 / 3.0)
    assert float(schedule(1)) == pytest.approx(2.0 / 3.0)
    assert float(schedule(7)) == pytest.approx(1.0)
    assert float(schedule(8)) == pytest.approx(0.505)
    assert float(schedule(9)) == pytest.approx(0.01)


def test_wsd_schedule_config_one_step_decay_reaches_end_lr_ratio():
    config = rfft.ScheduleConfig(
        kind="wsd",
        total_steps=10,
        warmup_steps=0,
        decay_steps=1,
        end_lr_ratio=0.2,
    )
    schedule = rfft.build_schedule(config, peak_lr=2.0)

    assert float(schedule(8)) == pytest.approx(2.0)
    assert float(schedule(9)) == pytest.approx(0.4)


def test_preview_uses_schedule_factory():
    preview = rfft.preview_schedule(
        rfft.ScheduleConfig(kind="linear", total_steps=5, warmup_steps=0),
        peak_lr=1.0,
    )

    assert preview[0].step == 0
    assert preview[-1].step == 4
    assert preview[-1].value == pytest.approx(0.01)


def test_warmup_cosine_from_epochs_resolves_step_counts():
    config = rfft.ScheduleConfig.warmup_cosine_from_epochs(
        num_epochs=5,
        num_batches=100,
        warmup_epochs=2,
        end_lr_ratio=0.05,
    )

    assert config.kind == "warmup_cosine"
    assert config.total_steps == 500
    assert config.warmup_steps == 200
    assert config.end_lr_ratio == pytest.approx(0.05)


def test_warmup_cosine_from_epochs_rejects_invalid_counts():
    with pytest.raises(ValueError, match="num_epochs"):
        rfft.ScheduleConfig.warmup_cosine_from_epochs(
            num_epochs=0,
            num_batches=100,
            warmup_epochs=2,
        )
    with pytest.raises(ValueError, match="warmup_epochs"):
        rfft.ScheduleConfig.warmup_cosine_from_epochs(
            num_epochs=5,
            num_batches=100,
            warmup_epochs=-1,
        )
