"""Group-rule resolution for plan-aware optimizers."""

from __future__ import annotations

from collections.abc import Iterable

from .config import CompiledGroup, GroupRule, OptimizerConfig, OptimizerName, PlanGroup


def compile_groups(
    groups: dict[str, PlanGroup],
    optimizer: OptimizerConfig,
    rules: Iterable[GroupRule] = (),
) -> tuple[CompiledGroup, ...]:
    """Resolve plan groups into optimizer groups."""

    compiled = []
    for label, group in sorted(groups.items()):
        matched = tuple(rule for rule in rules if matches_rule(group, rule))
        matched_rule_names = tuple(
            dict.fromkeys(rule.name for rule in matched if rule.name)
        )
        rule_lr = _rule_lr_multiplier(group, optimizer, matched)
        weight_decay_value = _resolve_weight_decay_value(group, optimizer, matched)
        weight_decay = weight_decay_value != 0.0
        optimizer_name = _resolve_optimizer(optimizer.name, matched)
        effective_lr = optimizer.base_lr * group.lr_multiplier * rule_lr
        compiled.append(
            CompiledGroup(
                source_label=label,
                optimizer=optimizer_name,
                base_lr=optimizer.base_lr,
                plan_lr_multiplier=group.lr_multiplier,
                rule_lr_multiplier=rule_lr,
                effective_lr=effective_lr,
                weight_decay=weight_decay,
                weight_decay_value=weight_decay_value,
                role=group.role,
                depth=group.depth,
                tags=group.tags,
                param_count=group.param_count,
                byte_count=group.byte_count,
                leaf_count=group.leaf_count,
                matched_rule_names=matched_rule_names,
            )
        )
    return tuple(compiled)


def matches_rule(group: PlanGroup, rule: GroupRule) -> bool:
    """Return whether a plan group satisfies a Rollfast override rule."""

    if rule.label is not None and group.label != rule.label:
        return False
    if rule.label_prefix is not None and not group.label.startswith(rule.label_prefix):
        return False
    if rule.role is not None and group.role != rule.role:
        return False
    if rule.tag is not None and rule.tag not in group.tags:
        return False
    if rule.min_depth is not None:
        if group.depth is None or group.depth < rule.min_depth:
            return False
    if rule.max_depth is not None:
        if group.depth is None or group.depth > rule.max_depth:
            return False
    return True


def preview_groups(groups: tuple[CompiledGroup, ...]) -> tuple[dict[str, object], ...]:
    """Return JSON-friendly group rows for diagnostics."""

    return tuple(
        {
            "label": group.source_label,
            "optimizer": group.optimizer,
            "base_lr": group.base_lr,
            "plan_lr_multiplier": group.plan_lr_multiplier,
            "rule_lr_multiplier": group.rule_lr_multiplier,
            "effective_lr": group.effective_lr,
            "weight_decay": group.weight_decay_value,
            "params": group.param_count,
            "bytes": group.byte_count,
            "role": group.role,
            "depth": group.depth,
            "tags": tuple(sorted(group.tags)),
            "matched_rules": group.matched_rule_names,
        }
        for group in groups
    )


def unmatched_rule_warnings(
    groups: dict[str, PlanGroup],
    rules: Iterable[GroupRule] = (),
) -> tuple[str, ...]:
    """Return warnings for named rules that do not match any plan group."""

    plan_groups = tuple(groups.values())
    warnings = []
    for rule in rules:
        if not rule.name:
            continue
        if not any(matches_rule(group, rule) for group in plan_groups):
            warnings.append(f"group rule {rule.name!r} matched no groups.")
    return tuple(dict.fromkeys(warnings))


def _rule_lr_multiplier(
    group: PlanGroup,
    optimizer: OptimizerConfig,
    rules: tuple[GroupRule, ...],
) -> float:
    multiplier = 1.0
    for rule in rules:
        multiplier *= rule.lr_multiplier
    if optimizer.lora_b_lr_ratio is not None and _is_lora_b_group(group):
        multiplier *= optimizer.lora_b_lr_ratio
    return multiplier


def _resolve_weight_decay_value(
    group: PlanGroup,
    optimizer: OptimizerConfig,
    rules: tuple[GroupRule, ...],
) -> float:
    selected = [
        rule
        for rule in rules
        if rule.weight_decay is not None or rule.weight_decay_value is not None
    ]
    if not selected:
        return optimizer.weight_decay if group.weight_decay else 0.0
    max_priority = max(rule.priority for rule in selected)
    highest = [rule for rule in selected if rule.priority == max_priority]
    values = {_rule_weight_decay_value(rule, optimizer) for rule in highest}
    if len(values) > 1:
        names = tuple(rule.name or repr(rule) for rule in highest)
        raise ValueError(
            f"conflicting weight_decay rules at priority {max_priority}: {names}"
        )
    return _rule_weight_decay_value(highest[-1], optimizer)


def _rule_weight_decay_value(rule: GroupRule, optimizer: OptimizerConfig) -> float:
    if rule.weight_decay_value is not None:
        return float(rule.weight_decay_value)
    return optimizer.weight_decay if rule.weight_decay else 0.0


def _resolve_optimizer(
    default: OptimizerName,
    rules: tuple[GroupRule, ...],
) -> OptimizerName:
    selected = [rule for rule in rules if rule.optimizer is not None]
    if not selected:
        return default
    max_priority = max(rule.priority for rule in selected)
    highest = [rule for rule in selected if rule.priority == max_priority]
    values = {rule.optimizer for rule in highest}
    if len(values) > 1:
        names = tuple(rule.name or repr(rule) for rule in highest)
        raise ValueError(
            f"conflicting optimizer rules at priority {max_priority}: {names}"
        )
    return highest[-1].optimizer or default


def _is_lora_b_group(group: PlanGroup) -> bool:
    return (
        group.label.startswith("lora_B")
        or "lora.factor_B" in group.tags
        or "lora_b" in group.tags
    )


__all__ = (
    "compile_groups",
    "matches_rule",
    "preview_groups",
    "unmatched_rule_warnings",
)
