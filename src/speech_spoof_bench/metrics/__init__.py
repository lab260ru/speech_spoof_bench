"""Pluggable metric registry.

Adding a new metric: drop a file under this package that calls
``@register_metric(...)`` at import time. The dataset's ``eval.yaml``
references metrics by their ``id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

ScoresMap = dict[str, float]
LabelsMap = dict[str, int]
MetricConfig = dict[str, Any]


@dataclass(frozen=True)
class MetricResult:
    value: float
    extras: dict[str, Any] = field(default_factory=dict)


# A metric fn is (scores, labels) -> MetricResult, and MAY accept an optional
# third positional `config` (e.g. srr_complement needs a calibrated threshold).
# Invoke via call_metric() so 2-arg metrics keep working unchanged.
MetricFn = Callable[..., MetricResult]


@dataclass(frozen=True)
class MetricSpec:
    id: str
    display_name: str
    lower_is_better: bool
    requires_audio: bool
    fn: MetricFn
    # True if fn takes a 3rd `config` arg (e.g. srr_complement needs a
    # calibrated threshold). Declared explicitly at registration rather than
    # introspected, so a wrapped/decorated fn can't silently mis-dispatch.
    wants_config: bool = False


_REGISTRY: dict[str, MetricSpec] = {}


def register_metric(
    *,
    id: str,
    display_name: str,
    lower_is_better: bool,
    requires_audio: bool = False,
    wants_config: bool = False,
) -> Callable[[MetricFn], MetricFn]:
    def decorator(fn: MetricFn) -> MetricFn:
        if id in _REGISTRY:
            raise ValueError(f"metric id {id!r} already registered")
        _REGISTRY[id] = MetricSpec(
            id=id,
            display_name=display_name,
            lower_is_better=lower_is_better,
            requires_audio=requires_audio,
            fn=fn,
            wants_config=wants_config,
        )
        return fn

    return decorator


def get_metric(id: str) -> MetricSpec:
    try:
        return _REGISTRY[id]
    except KeyError:
        raise KeyError(f"metric id {id!r} is not registered") from None


def list_metrics() -> list[MetricSpec]:
    return list(_REGISTRY.values())


def is_registered(id: str) -> bool:
    return id in _REGISTRY


def call_metric(
    spec: MetricSpec,
    scores: ScoresMap,
    labels: LabelsMap,
    config: MetricConfig | None = None,
) -> MetricResult:
    """Invoke ``spec.fn``, passing ``config`` only if it declared ``wants_config``.

    Dispatch keys on the explicit ``MetricSpec.wants_config`` flag (set at
    registration), NOT on parameter introspection — a config-needing metric
    wrapped by a decorator without ``functools.wraps`` would otherwise look
    2-arg and silently drop the config (and a defensively variadic 2-arg metric
    would wrongly receive one). ``config`` is per-call (a threshold differs per
    submission), so it lives here rather than on the registration.
    """
    if spec.wants_config:
        return spec.fn(scores, labels, config or {})
    return spec.fn(scores, labels)


# Auto-import built-in metric modules so their @register_metric decorators
# fire on package import. New metric files dropped into this directory need
# to be added here (one line) to participate without an explicit import at
# every call site.
from . import eer  # noqa: E402, F401
from . import srr_complement  # noqa: E402, F401
