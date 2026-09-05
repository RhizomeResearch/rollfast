"""State traversal and path encoding shared by diagnostics and migration."""

from typing import Any

import jax

from rollfast.optim.adam8 import QuantizedBlocks


def _path_leaves(tree: Any) -> list[tuple[Any, Any]]:
    return jax.tree_util.tree_flatten_with_path(tree, is_leaf=_is_state_leaf)[0]


def _is_state_leaf(leaf: Any) -> bool:
    return leaf is None or isinstance(leaf, QuantizedBlocks) or _is_masked_node(leaf)


def _is_masked_node(leaf: Any) -> bool:
    return leaf.__class__.__name__ == "MaskedNode"


def _path_tokens(path: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(_path_token(part) for part in path)


def _path_token(part: Any) -> str:
    if hasattr(part, "name"):
        return f"attr:{part.name}"
    if hasattr(part, "key"):
        return f"key:{part.key}"
    if hasattr(part, "idx"):
        return f"idx:{part.idx}"
    return repr(part)


def _format_tokens(tokens: tuple[str, ...]) -> str:
    return "/".join(tokens)
