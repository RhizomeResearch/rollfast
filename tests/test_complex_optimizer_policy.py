import jax.numpy as jnp
import optax
import pytest

from rollfast.optim.adam8 import scale_by_adam8
from rollfast.optim.apollo import apollo_adamw
from rollfast.optim.galore import galore_adamw
from rollfast.optim.hyperball import apply_hyperball
from rollfast.optim.soda import soda
from rollfast.schedules.schedulefree import schedule_free


@pytest.mark.parametrize(
    ("family", "make_tx"),
    [
        ("AdamW8", lambda: scale_by_adam8()),
        ("APOLLO", lambda: apollo_adamw(learning_rate=0.01, rank=1)),
        ("GaLore", lambda: galore_adamw(learning_rate=0.01, rank=1)),
        ("SODA", lambda: soda(optax.sgd(0.01))),
        (
            "Schedule-Free",
            lambda: schedule_free(optax.sgd(0.01), learning_rate=0.01),
        ),
        ("Hyperball", lambda: apply_hyperball(learning_rate=0.01)),
    ],
)
def test_real_only_optimizers_reject_complex_params_during_init(family, make_tx):
    params = {"layer": {"w": jnp.ones((2, 2), dtype=jnp.complex64)}}

    with pytest.raises(ValueError, match=rf"{family}.*complex.*layer.*w"):
        make_tx().init(params)


@pytest.mark.parametrize(
    ("family", "make_tx"),
    [
        ("AdamW8", lambda: scale_by_adam8()),
        ("APOLLO", lambda: apollo_adamw(learning_rate=0.01, rank=1)),
        ("GaLore", lambda: galore_adamw(learning_rate=0.01, rank=1)),
        ("SODA", lambda: soda(optax.sgd(0.01))),
        (
            "Schedule-Free",
            lambda: schedule_free(optax.sgd(0.01), learning_rate=0.01),
        ),
        ("Hyperball", lambda: apply_hyperball(learning_rate=0.01)),
    ],
)
def test_real_only_optimizers_reject_complex_updates_before_casts(family, make_tx):
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    updates = {"w": jnp.ones((2, 2), dtype=jnp.complex64)}
    tx = make_tx()

    with pytest.raises(ValueError, match=rf"{family}.*complex.*w"):
        tx.update(updates, tx.init(params), params)
