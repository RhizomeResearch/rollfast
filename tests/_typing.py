from typing import cast

import jax

ArrayDict = dict[str, jax.Array]


def as_array_dict(tree: object) -> ArrayDict:
    return cast(ArrayDict, tree)
