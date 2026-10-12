"""Unit tests for marker rendering and the factory helpers (Task 2.2).

These cover the display contract of :class:`week1_harness.markers.MarkedValue`
and the ``real()`` / ``placeholder()`` / ``estimate()`` factories:

* ``render()`` leaves a ``REAL`` value untagged (R4.6).
* ``render()`` appends ``[PLACEHOLDER: <reason>]`` / ``[ESTIMATE: <reason>]``
  so the marker and its reason are visible inline (R4.4, R4.5).
* each factory builds a ``MarkedValue`` with the expected marker, reason, and
  unit.

Requirements: 4.4, 4.5, 4.6.
"""

from week1_harness.markers import (
    Marker,
    MarkedValue,
    estimate,
    placeholder,
    real,
)


# --- render(): REAL values are untagged (R4.6) ---------------------------------

def test_render_real_with_unit_is_untagged():
    # Canonical example from the task: a REAL time cell renders bare.
    assert real(3.42, unit="s").render() == "3.42s"


def test_render_real_without_unit_is_untagged():
    assert real(42).render() == "42"


def test_render_real_never_shows_a_marker_tag():
    rendered = real("done", unit=" items").render()
    assert rendered == "done items"
    assert "[" not in rendered and "]" not in rendered


# --- render(): PLACEHOLDER shows [PLACEHOLDER: <reason>] (R4.4) -----------------

def test_render_placeholder_shows_marker_and_reason():
    assert (
        placeholder("n/a", "training stubbed").render()
        == "n/a [PLACEHOLDER: training stubbed]"
    )


def test_render_placeholder_with_unit_keeps_value_and_tag():
    assert (
        placeholder(0, "ungated weights not in local cache", unit=" dets").render()
        == "0 dets [PLACEHOLDER: ungated weights not in local cache]"
    )


# --- render(): ESTIMATE shows [ESTIMATE: <reason>] (R4.5) -----------------------

def test_render_estimate_shows_marker_and_reason():
    assert (
        estimate("~25 min", "NVIDIA GPU absent").render()
        == "~25 min [ESTIMATE: NVIDIA GPU absent]"
    )


def test_render_estimate_with_unit_keeps_value_and_tag():
    assert (
        estimate(14.3, "webcam and GPU absent", unit=" fps").render()
        == "14.3 fps [ESTIMATE: webcam and GPU absent]"
    )


# --- factories build MarkedValue instances with the expected fields ------------

def test_real_factory_builds_expected_instance():
    mv = real(3.42, unit="s")
    assert isinstance(mv, MarkedValue)
    assert mv.value == 3.42
    assert mv.marker is Marker.REAL
    assert mv.reason is None  # REAL carries no reason (R4.6)
    assert mv.unit == "s"


def test_real_factory_defaults_unit_to_none():
    mv = real(42)
    assert mv.marker is Marker.REAL
    assert mv.reason is None
    assert mv.unit is None


def test_placeholder_factory_builds_expected_instance():
    mv = placeholder("n/a", "training stubbed")
    assert isinstance(mv, MarkedValue)
    assert mv.value == "n/a"
    assert mv.marker is Marker.PLACEHOLDER
    assert mv.reason == "training stubbed"  # reason recorded alongside marker (R4.4)
    assert mv.unit is None


def test_placeholder_factory_preserves_unit():
    mv = placeholder(0, "gated weights absent", unit=" masks")
    assert mv.marker is Marker.PLACEHOLDER
    assert mv.reason == "gated weights absent"
    assert mv.unit == " masks"


def test_estimate_factory_builds_expected_instance():
    mv = estimate(14.3, "webcam and GPU absent", unit=" fps")
    assert isinstance(mv, MarkedValue)
    assert mv.value == 14.3
    assert mv.marker is Marker.ESTIMATE
    assert mv.reason == "webcam and GPU absent"  # absent hardware recorded (R4.5)
    assert mv.unit == " fps"


def test_estimate_factory_defaults_unit_to_none():
    mv = estimate("~25 min", "NVIDIA GPU absent")
    assert mv.marker is Marker.ESTIMATE
    assert mv.reason == "NVIDIA GPU absent"
    assert mv.unit is None
