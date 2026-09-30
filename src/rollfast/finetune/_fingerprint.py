"""Deterministic, non-executable descriptions for fingerprint format v1."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import math
import sys
from typing import Any

import jax.tree_util as jtu
import numpy as np

FINGERPRINT_VERSION = 1


def _type_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def encode_static(value: Any, *, depth: int = 0) -> Any:
    """Describe supported static values without repr, code, or array contents."""
    if depth > 64:
        raise TypeError("fingerprint static metadata is cyclic or exceeds depth 64.")
    cls = type(value)

    def encode(child: Any) -> Any:
        return encode_static(child, depth=depth + 1)

    if value is None:
        return ["none"]
    if cls in (bool, int, str):
        return [cls.__name__, value]
    if cls is float:
        if not math.isfinite(value):
            raise ValueError("fingerprint static floats must be finite.")
        return ["float", value.hex()]
    if isinstance(value, np.dtype):
        if value.fields is not None or value.subdtype is not None or value.hasobject:
            raise TypeError(
                "fingerprint static dtypes must be plain non-object dtypes."
            )
        return ["dtype", value.name, value.str]
    if isinstance(value, type):
        return ["type", _type_name(value)]
    if cls is functools.partial:
        return [
            "partial",
            encode(value.func),
            encode(value.args),
            encode(value.keywords),
        ]
    if cls in (tuple, list):
        return [cls.__name__, [encode(child) for child in value]]
    if cls in (set, frozenset):
        return [cls.__name__, sorted((encode(child) for child in value), key=_sort_key)]
    if cls is dict:
        return [
            "dict",
            sorted(
                ([encode(key), encode(child)] for key, child in value.items()),
                key=_sort_key,
            ),
        ]
    if dataclasses.is_dataclass(value):
        return [
            "dataclass",
            _type_name(cls),
            [
                [field.name, encode(getattr(value, field.name))]
                for field in dataclasses.fields(value)
            ],
        ]
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", "")
        target = sys.modules.get(module)
        for part in name.split("."):
            target = getattr(target, part, None)
        unwrapped = inspect.unwrap(value)
        if (
            not name
            or "<" in name
            or target is not value
            or not (inspect.isfunction(unwrapped) or inspect.isbuiltin(unwrapped))
            or getattr(unwrapped, "__closure__", None)
            or inspect.ismethod(value)
        ):
            raise TypeError(
                "unsupported fingerprint static callable: "
                f"{module}.{name or cls.__qualname__}; use a named function or partial."
            )
        return ["callable", module, name]
    # Registered metadata includes Equinox's empty Missing sentinel. Respect its
    # PyTree contract rather than importing Equinox or inspecting opaque objects.
    leaves, definition = jtu.tree_flatten(value)
    if definition.node_data() is not None:
        return [
            "pytree",
            encode_treedef(definition, depth=depth + 1),
            [encode(leaf) for leaf in leaves],
        ]
    raise TypeError(f"unsupported fingerprint static metadata: {_type_name(cls)}.")


def encode_treedef(definition: Any, *, depth: int = 0) -> Any:
    """Describe node types, auxiliary data, and children in PyTree order."""
    if depth > 64:
        raise TypeError("fingerprint tree is cyclic or exceeds depth 64.")
    data = definition.node_data()
    if data is None:
        return ["leaf"]
    cls, auxiliary = data
    return [
        "node",
        _type_name(cls),
        encode_static(auxiliary, depth=depth + 1),
        [encode_treedef(child, depth=depth + 1) for child in definition.children()],
    ]


def encode_path(path: tuple[Any, ...]) -> list[Any]:
    """Preserve key types and values instead of joining printable path tokens."""
    result = []
    for key in path:
        if isinstance(key, jtu.DictKey | jtu.FlattenedIndexKey):
            value = key.key
        elif isinstance(key, jtu.SequenceKey):
            value = key.idx
        elif isinstance(key, jtu.GetAttrKey):
            value = key.name
        else:
            raise TypeError(
                f"unsupported fingerprint path key: {_type_name(type(key))}."
            )
        result.append([_type_name(type(key)), encode_static(value)])
    return result
