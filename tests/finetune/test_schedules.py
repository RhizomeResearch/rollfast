import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rollfast.finetune as rfft
from rollfast.schedules import interval_schedule
from rollfast.schedules.wsd import wsd_schedule


def _pytorch_cosine_annealing(index, *, base, t_max, eta_min=0.0):
    """CosineAnnealingLR's closed form after ``index`` scheduler steps."""

    return eta_min + (base - eta_min) * (1 + math.cos(math.pi * index / t_max)) / 2


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


@pytest.mark.parametrize(
    ("epochs", "per_epoch", "end_lr_ratio"),
    ((2, 3, 0.0), (4, 3, 0.0), (20, 7, 0.0), (50, 1, 0.0), (9, 2, 0.1)),
)
def test_cosine_annealing_matches_pytorch_stepped_once_per_epoch(
    epochs, per_epoch, end_lr_ratio
):
    base = 1e-3
    schedule = rfft.build_schedule(
        rfft.ScheduleConfig(
            kind="cosine_annealing",
            total_steps=epochs,
            steps_per_interval=per_epoch,
            warmup_fraction=0.0,
            end_lr_ratio=end_lr_ratio,
        ),
        peak_lr=base,
    )
    updates = jnp.arange(epochs * per_epoch)
    expected = [
        _pytorch_cosine_annealing(
            update // per_epoch,
            base=base,
            t_max=epochs,
            eta_min=end_lr_ratio * base,
        )
        for update in range(epochs * per_epoch)
    ]

    for evaluate in (jax.vmap(schedule), jax.jit(jax.vmap(schedule))):
        np.testing.assert_allclose(evaluate(updates), expected, rtol=2e-6)


def test_cosine_annealing_matches_values_measured_with_pytorch():
    # torch 2.11.0 CosineAnnealingLR(T_max=4) on AdamW(lr=1e-3), stepped after
    # each epoch of three updates.
    measured = [1.0e-3, 8.535534e-4, 5.0e-4, 1.464466e-4]
    schedule = rfft.build_schedule(
        rfft.ScheduleConfig(
            kind="cosine_annealing",
            total_steps=4,
            steps_per_interval=3,
            warmup_fraction=0.0,
            end_lr_ratio=0.0,
        ),
        peak_lr=1e-3,
    )

    np.testing.assert_allclose(
        jax.vmap(schedule)(jnp.arange(12)),
        np.repeat(measured, 3),
        rtol=1e-6,
    )


@pytest.mark.parametrize("after_end", ("clamp", "cycle"))
def test_cosine_annealing_after_end(after_end):
    peak, end, t_max = 0.5, 0.05, 4
    schedule = rfft.build_schedule(
        rfft.ScheduleConfig(
            kind="cosine_annealing",
            total_steps=10,
            t_max=t_max,
            warmup_fraction=0.0,
            end_lr_ratio=end / peak,
            after_end=after_end,
        ),
        peak_lr=peak,
    )
    indices = list(range(3 * t_max + 1))
    expected = [
        end
        if after_end == "clamp" and index >= t_max
        else _pytorch_cosine_annealing(index, base=peak, t_max=t_max, eta_min=end)
        for index in indices
    ]

    np.testing.assert_allclose(
        jax.jit(jax.vmap(schedule))(jnp.asarray(indices)), expected, rtol=2e-6
    )


def test_cosine_annealing_warms_up_like_warmup_cosine():
    peak, end = 1.0, 0.1
    schedule = rfft.build_schedule(
        rfft.ScheduleConfig(
            kind="cosine_annealing",
            total_steps=10,
            warmup_steps=2,
            end_lr_ratio=end / peak,
        ),
        peak_lr=peak,
    )
    expected = [0.5, 1.0] + [
        _pytorch_cosine_annealing(index, base=peak, t_max=8, eta_min=end)
        for index in range(8)
    ]

    np.testing.assert_allclose(jax.vmap(schedule)(jnp.arange(10)), expected, rtol=2e-6)


def test_cosine_annealing_uses_the_shared_warmup_and_floor_defaults():
    schedule = rfft.build_schedule(
        rfft.ScheduleConfig(kind="cosine_annealing", total_steps=40), peak_lr=1.0
    )

    # warmup_fraction=0.05 gives two warmup slots and t_max=38.
    assert float(schedule(0)) == pytest.approx(0.5)
    assert float(schedule(2)) == pytest.approx(1.0)
    assert float(schedule(40)) == pytest.approx(0.01)


@pytest.mark.parametrize(
    "kind",
    ("constant", "warmup_cosine", "cosine_annealing", "wsd", "linear", "polynomial"),
)
def test_steps_per_interval_holds_every_kind_at_its_interval_index(kind):
    per_interval = 4
    config = rfft.ScheduleConfig(
        kind=kind,
        total_steps=6,
        warmup_steps=1,
        decay_steps=2,
        end_lr_ratio=0.1,
        power=2.0,
        steps_per_interval=per_interval,
    )
    held = rfft.build_schedule(config, peak_lr=0.3)
    per_count = rfft.build_schedule(replace(config, steps_per_interval=1), peak_lr=0.3)
    counts = jnp.arange(config.total_updates + 2 * per_interval)

    for evaluate in (jax.vmap, lambda fn: jax.jit(jax.vmap(fn))):
        np.testing.assert_array_equal(
            evaluate(held)(counts), evaluate(per_count)(counts // per_interval)
        )


def test_one_step_interval_keeps_the_schedule_unchanged():
    def schedule(count):
        return count * 2.0

    assert interval_schedule(schedule, 1) is schedule


@pytest.mark.parametrize("steps_per_interval", (0, -2, 2.5, True))
def test_interval_schedule_rejects_non_positive_integers(steps_per_interval):
    with pytest.raises(ValueError, match="steps_per_interval"):
        interval_schedule(lambda count: count, steps_per_interval)


def test_preview_spans_updates_and_shows_held_values():
    config = rfft.ScheduleConfig(
        kind="cosine_annealing",
        total_steps=4,
        steps_per_interval=3,
        warmup_fraction=0.0,
        end_lr_ratio=0.0,
    )

    default = rfft.preview_schedule(config, peak_lr=1.0)
    every_update = rfft.preview_schedule(config, peak_lr=1.0, steps=tuple(range(12)))

    assert [point.step for point in default] == [0, 6, 11]
    for point in every_update:
        assert point.value == pytest.approx(
            _pytorch_cosine_annealing(point.step // 3, base=1.0, t_max=4), rel=2e-6
        )


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
