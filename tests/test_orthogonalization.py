from rollfast.optim.orthogonalization import MUON_NS_COEFFS


def test_muon_ns_coeffs_matches_existing_default():
    assert MUON_NS_COEFFS == (3.4445, -4.7750, 2.0315)
