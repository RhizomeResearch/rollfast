import json
from pathlib import Path
from typing import Literal, get_args, get_origin

import pytest

import rollfast
import rollfast.finetune as rfft
from rollfast import integrations, optim, schedules

V1_PUBLIC_API = json.loads(Path(__file__).with_name("public_api_v1.json").read_text())


def test_public_typing_aliases_preserve_runtime_introspection():
    assert get_origin(rollfast.MomentumAccumulator) is Literal
    assert get_args(rollfast.MomentumAccumulator) == ("ema", "heavy_ball")
    assert rollfast.MuonNsCoeffs is rollfast.NsCoeffs
    assert str in get_args(rollfast.NsCoeffs)


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
