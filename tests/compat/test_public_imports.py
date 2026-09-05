import json
from pathlib import Path

import pytest

import rollfast
import rollfast.finetune as rfft
import rollfast.integrations as integrations
import rollfast.optim as optim
import rollfast.schedules as schedules


V1_PUBLIC_API = json.loads(Path(__file__).with_name("public_api_v1.json").read_text())


@pytest.mark.parametrize(
    "module",
    (rollfast, optim, schedules, rfft, integrations),
    ids=lambda module: module.__name__,
)
def test_v1_public_api(module):
    expected = V1_PUBLIC_API[module.__name__]
    assert sorted(module.__all__) == expected, (
        f"{module.__name__} public API changed; review compatibility and update "
        "the v1 baseline only for an intentional release change"
    )
    for name in expected:
        assert hasattr(module, name), f"{module.__name__}.{name}"
