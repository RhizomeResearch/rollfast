"""Describe checkpoint trees without serializing Python object constructors."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import jax
import numpy as np


def checkpoint_parts(checkpoint: Any) -> dict[str, Any]:
    """Return the mutable components stored by an optimizer checkpoint."""
    return {
        name: getattr(checkpoint, name)
        for name in (
            "state",
            "master_params",
            "accumulation",
            "model_state",
            "loss_scale",
            "rng",
            "counters",
        )
    }


def describe_tree(value: Any) -> tuple[dict[str, Any], list[Any]]:
    """Describe supported containers and array leaves in deterministic order."""
    leaves: list[Any] = []

    def visit(node: Any, depth: int) -> dict[str, Any]:
        if depth > 64:
            raise ValueError("checkpoint tree exceeds maximum depth (64).")
        if isinstance(node, (jax.Array, np.ndarray)):
            dtype = node.dtype
            if (
                not jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)
                and np.dtype(dtype).hasobject
            ):
                raise TypeError(
                    "object arrays cannot be saved in optimizer checkpoints."
                )
            index = len(leaves)
            leaves.append(node)
            return {
                "kind": "array",
                "index": index,
                "shape": list(node.shape),
                "dtype": str(dtype),
                "scope": _array_scope(node),
            }
        if node is None:
            return {"kind": "none"}
        if type(node) in (bool, int, float, str):
            if isinstance(node, float) and not math.isfinite(node):
                raise ValueError("nonfinite Python checkpoint scalar is unsupported.")
            return {"kind": "scalar", "type": type(node).__name__, "value": node}
        if isinstance(node, Mapping) and type(node) is dict:
            keys = sorted(node, key=lambda key: (type(key).__name__, str(key)))
            if any(type(key) not in (str, int) for key in keys):
                raise TypeError(
                    "checkpoint dictionaries require string or integer keys."
                )
            return {
                "kind": "dict",
                "keys": keys,
                "children": [visit(node[key], depth + 1) for key in keys],
            }
        if isinstance(node, tuple) and hasattr(node, "_fields"):
            return {
                "kind": "namedtuple",
                "type": _type_name(node),
                "fields": list(node._fields),
                "children": [visit(child, depth + 1) for child in node],
            }
        if type(node) in (tuple, list):
            return {
                "kind": type(node).__name__,
                "children": [visit(child, depth + 1) for child in node],
            }
        if callable(getattr(node, "tree_flatten", None)):
            children, _ = node.tree_flatten()
            return {
                "kind": "pytree",
                "type": _type_name(node),
                "children": [visit(child, depth + 1) for child in children],
            }
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            flattened, definition = jax.tree_util.tree_flatten(node)
            if definition.num_nodes > 1:
                return {
                    "kind": "registered_pytree",
                    "type": _type_name(node),
                    "children": [visit(child, depth + 1) for child in flattened],
                }
            fields = [field.name for field in dataclasses.fields(node)]
            return {
                "kind": "dataclass",
                "type": _type_name(node),
                "fields": fields,
                "children": [visit(getattr(node, name), depth + 1) for name in fields],
            }
        raise TypeError(f"unsupported checkpoint node: {_type_name(node)}.")

    return visit(value, 0), leaves


def match_tree_schema(actual: Any, expected: Any, *, path: str = "checkpoint") -> None:
    """Validate an encoded tree against a trusted local tree descriptor."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise TypeError(f"{path} has malformed state schema.")
    if actual.keys() != expected.keys():
        raise ValueError(f"{path} state schema fields mismatch.")
    fields = set(expected) - {"value"}
    if {key: actual.get(key) for key in fields if key != "children"} != {
        key: expected[key] for key in fields if key != "children"
    }:
        raise ValueError(f"{path} state schema mismatch.")
    if "children" in expected:
        actual_children = actual.get("children")
        expected_children = expected["children"]
        if not isinstance(actual_children, list) or len(actual_children) != len(
            expected_children
        ):
            raise ValueError(f"{path} state schema child count mismatch.")
        for index, (actual_child, expected_child) in enumerate(
            zip(actual_children, expected_children, strict=True)
        ):
            match_tree_schema(actual_child, expected_child, path=f"{path}/{index}")


def restore_tree(template: Any, schema: dict[str, Any], arrays: list[Any]) -> Any:
    """Rebuild using only types and static fields from a trusted template."""
    kind = schema["kind"]
    if kind == "array":
        return arrays[schema["index"]]
    if kind == "none":
        return None
    if kind == "scalar":
        return schema["value"]
    children = schema["children"]
    if kind == "dict":
        return {
            key: restore_tree(template[key], child, arrays)
            for key, child in zip(schema["keys"], children, strict=True)
        }
    if kind in ("list", "tuple", "namedtuple"):
        values = [
            restore_tree(item, child, arrays)
            for item, child in zip(template, children, strict=True)
        ]
        if kind == "list":
            return values
        if kind == "namedtuple":
            return type(template)(*values)
        return tuple(values)
    if kind == "dataclass":
        return dataclasses.replace(
            template,
            **{
                name: restore_tree(getattr(template, name), child, arrays)
                for name, child in zip(schema["fields"], children, strict=True)
            },
        )
    if kind == "pytree":
        template_children, auxiliary = template.tree_flatten()
        values = [
            restore_tree(item, child, arrays)
            for item, child in zip(template_children, children, strict=True)
        ]
        return type(template).tree_unflatten(auxiliary, values)
    if kind == "registered_pytree":
        template_leaves, definition = jax.tree_util.tree_flatten(template)
        values = [
            restore_tree(item, child, arrays)
            for item, child in zip(template_leaves, children, strict=True)
        ]
        return definition.unflatten(values)
    raise ValueError(f"unknown checkpoint node kind: {kind!r}.")


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _array_scope(value: Any) -> str:
    if not isinstance(value, jax.Array):
        return "local"
    return (
        "global"
        if len(value.sharding.device_set) > 1
        or any(
            device.process_index != jax.process_index()
            for device in value.sharding.device_set
        )
        else "local"
    )
