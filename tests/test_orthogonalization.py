import pytest
import jax.numpy as jnp
from typing import cast

import rollfast
from rollfast.optim import muon as muon_module
from rollfast.optim.orthogonalization import (
    MUON_NS_COEFFS,
    NsCoeffs,
    polar_express_coeffs,
    resolve_ns_coeffs,
)


def test_muon_ns_coeffs_matches_existing_default():
    assert MUON_NS_COEFFS == (3.4445, -4.7750, 2.0315)


def test_resolve_ns_coeffs_handles_presets_and_ordered_schedules():
    assert jnp.allclose(
        resolve_ns_coeffs("standard", 5),
        jnp.asarray((3.4445, -4.7750, 2.0315)),
    )

    polar = resolve_ns_coeffs("polar_express", 3)
    assert polar.shape == (3, 3)
    assert jnp.allclose(polar, jnp.asarray(polar_express_coeffs(num_iters=3)))

    schedule = ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    assert jnp.allclose(
        resolve_ns_coeffs(schedule, 2),
        jnp.asarray(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
    )


def test_resolve_ns_coeffs_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Unknown ns_coeff"):
        resolve_ns_coeffs("unknown", 2)

    with pytest.raises(ValueError, match="shape"):
        resolve_ns_coeffs(cast(NsCoeffs, (1.0, 2.0)), 2)

    with pytest.raises(ValueError, match="Not enough coeffs"):
        resolve_ns_coeffs(((1.0, 0.0, 0.0),), 2)


def test_muon_compatibility_exports_point_at_shared_implementation():
    assert muon_module.resolve_ns_coeffs is resolve_ns_coeffs
    assert muon_module.polar_express_coeffs is polar_express_coeffs
    assert rollfast.resolve_ns_coeffs is resolve_ns_coeffs
    assert rollfast.MUON_NS_COEFFS == MUON_NS_COEFFS
    assert NsCoeffs is rollfast.NsCoeffs
