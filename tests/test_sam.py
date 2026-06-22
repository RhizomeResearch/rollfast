import jax.numpy as jnp

import rollfast
from rollfast.optim.sam import add_perturbation


def test_sam_perturbation_has_requested_global_l2_norm():
    grads = {"w": jnp.array([3.0, 4.0], dtype=jnp.float32), "b": None}
    perturbation, perturbation_norm = rollfast.sam_perturbation(
        grads,
        rho=0.2,
    )

    assert jnp.allclose(perturbation_norm, 0.2, atol=1e-6)
    assert jnp.allclose(rollfast.global_l2_norm(perturbation), 0.2, atol=1e-6)
    assert perturbation["b"] is None


def test_sam_perturbation_respects_masked_leaves():
    grads = {
        "w": jnp.ones((2,), dtype=jnp.float32),
        "b": jnp.ones((2,), dtype=jnp.float32),
    }
    perturbation, _ = rollfast.sam_perturbation(
        grads,
        rho=0.1,
        mask={"w": True, "b": False},
    )

    assert not jnp.allclose(perturbation["w"], 0.0)
    assert jnp.allclose(perturbation["b"], 0.0)


def test_asam_perturbation_scales_with_parameter_magnitude():
    params = {
        "small": jnp.ones((2,), dtype=jnp.float32),
        "large": jnp.ones((2,), dtype=jnp.float32) * 10.0,
    }
    grads = {
        "small": jnp.ones((2,), dtype=jnp.float32),
        "large": jnp.ones((2,), dtype=jnp.float32),
    }
    perturbation, _ = rollfast.sam_perturbation(
        grads,
        params=params,
        rho=0.5,
        adaptive=True,
        eta=0.01,
    )
    perturbed = add_perturbation(params, perturbation)

    assert jnp.linalg.norm(perturbation["large"]) > jnp.linalg.norm(
        perturbation["small"]
    )
    assert jnp.all(perturbed["large"] > params["large"])
