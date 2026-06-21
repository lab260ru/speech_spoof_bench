"""Tests for call_metric: the config-aware dispatch that keeps 2-arg metrics working."""

import pytest

from speech_spoof_bench.metrics import MetricResult, MetricSpec, call_metric, get_metric
import speech_spoof_bench.metrics.eer  # noqa: F401
import speech_spoof_bench.metrics.srr_complement  # noqa: F401


def _eer_inputs():
    scores = {"b0": 10.0, "b1": 11.0, "s0": -10.0, "s1": -11.0}
    labels = {"b0": 0, "b1": 0, "s0": 1, "s1": 1}
    return scores, labels


def test_call_metric_eer_two_arg():
    # A 2-arg metric (eer) is invoked without config and matches a direct call.
    scores, labels = _eer_inputs()
    spec = get_metric("eer_percent")
    via_helper = call_metric(spec, scores, labels)
    direct = spec.fn(scores, labels)
    assert via_helper.value == direct.value


def test_call_metric_eer_ignores_config():
    # Passing a config to a 2-arg metric must not break it.
    scores, labels = _eer_inputs()
    spec = get_metric("eer_percent")
    result = call_metric(spec, scores, labels, {"threshold": 0.0})
    assert result.value == pytest.approx(0.0, abs=1e-9)


def test_call_metric_passes_config_to_three_arg():
    # A 3-arg metric (srr_complement) receives the config and computes the value.
    scores = {"s0": -1.0, "s1": -1.0, "s2": 1.0, "s3": 1.0}
    labels = {"s0": 1, "s1": 1, "s2": 1, "s3": 1}
    spec = get_metric("srr_complement")
    result = call_metric(spec, scores, labels, {"threshold": 0.0})
    assert result.value == pytest.approx(50.0, abs=1e-9)


def test_call_metric_three_arg_without_config_raises():
    scores = {"s0": -1.0, "s1": 1.0}
    labels = {"s0": 1, "s1": 1}
    spec = get_metric("srr_complement")
    with pytest.raises(ValueError, match="threshold"):
        call_metric(spec, scores, labels)


def test_call_metric_uses_explicit_flag_not_introspection():
    # A config-needing fn wrapped so its signature is hidden (e.g. a decorator
    # without functools.wraps -> (*args, **kwargs)) must STILL receive config.
    # Dispatch keys on the explicit wants_config flag, not on parameter count.
    def hidden(*args, **kwargs):
        scores, labels, config = args
        return MetricResult(value=float(config["threshold"]))

    spec = MetricSpec(
        id="t_cfg", display_name="x", lower_is_better=True,
        requires_audio=False, fn=hidden, wants_config=True,
    )
    out = call_metric(spec, {"a": 1.0}, {"a": 1}, {"threshold": 7.0})
    assert out.value == 7.0


def test_call_metric_no_flag_never_passes_config():
    # A 2-arg metric defensively written with *args must NOT receive config
    # just because its parameter count looks variadic.
    seen = {}

    def two_arg(*args):
        seen["n"] = len(args)
        return MetricResult(value=0.0)

    spec = MetricSpec(
        id="t_nocfg", display_name="x", lower_is_better=True,
        requires_audio=False, fn=two_arg, wants_config=False,
    )
    call_metric(spec, {"a": 1.0}, {"a": 1}, {"threshold": 7.0})
    assert seen["n"] == 2
