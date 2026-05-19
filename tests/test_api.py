import rollfast
from rollfast.optim.adam import scale_by_adam
from rollfast.optim.psgd import scale_by_kron


def test_root_exports_direct_adam_and_kron_transforms():
    assert rollfast.scale_by_adam is scale_by_adam
    assert rollfast.scale_by_kron is scale_by_kron
