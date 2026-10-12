"""Property-based tests for the marker-reason invariant (Task 2.3).

Feature: week1-data-loop-harness, Property 1: Marker-reason invariant --
for any emitted value, the value is ``REAL`` with no reason, or it is
``PLACEHOLDER`` / ``ESTIMATE`` with a non-empty reason; it is never any other
combination. ``render()`` leaves a ``REAL`` value untagged and appends
``[MARKER: reason]`` for the non-real markers.

These tests exercise :class:`week1_harness.markers.MarkedValue` and the
``real()`` / ``placeholder()`` / ``estimate()`` factories across arbitrary
values, units, and reasons using Hypothesis.

**Validates: Requirements 4.2, 4.4, 4.5, 4.6, 10.2, 10.3, 11.2**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from week1_harness.markers import (
    Marker,
    MarkedValue,
    estimate,
    placeholder,
    real,
)

# --- Strategies ----------------------------------------------------------------

# Arbitrary underlying values the harness might wrap (numbers and strings).
# Brackets are stripped from generated text so the "no marker tag" check on a
# rendered REAL value is meaningful -- a bare value must not contain "[" itself.
_plain_text = st.text().map(lambda s: s.replace("[", "").replace("]", ""))

values = st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    _plain_text,
)

# Optional display units, including None and the empty string (both render as "").
units = st.one_of(
    st.none(),
    st.sampled_from(["s", " fps", "%", " min", ""]),
    _plain_text,
)

# Non-empty reasons: text that still has content after stripping whitespace, so
# it is always truthy and satisfies the PLACEHOLDER/ESTIMATE invariant.
reasons = st.text(min_size=1).filter(str.strip)


# --- Property 1a: REAL carries no reason and renders untagged (R4.6) -----------

@given(value=values, unit=units)
@settings(max_examples=100, deadline=None)
def test_real_has_no_reason_and_renders_untagged(value, unit):
    """real(value, unit) always has reason None and renders with no marker tag."""
    mv = real(value, unit)

    assert mv.marker is Marker.REAL
    assert mv.reason is None

    rendered = mv.render()
    # A REAL value renders as just value+unit, with no "[MARKER: reason]" tag.
    assert rendered == f"{value}{unit or ''}"
    assert "[" not in rendered


# --- Property 1b: PLACEHOLDER / ESTIMATE construct and render tagged (R4.4/4.5) -

@given(value=values, unit=units, reason=reasons)
@settings(max_examples=100, deadline=None)
def test_placeholder_and_estimate_construct_and_render_tagged(value, unit, reason):
    """A non-empty reason always constructs and renders as "[MARKER: reason]"."""
    for factory, marker, label in (
        (placeholder, Marker.PLACEHOLDER, "PLACEHOLDER"),
        (estimate, Marker.ESTIMATE, "ESTIMATE"),
    ):
        mv = factory(value, reason, unit)

        assert mv.marker is marker
        assert mv.reason == reason

        rendered = mv.render()
        assert rendered == f"{value}{unit or ''} [{label}: {reason}]"
        assert f"[{label}: {reason}]" in rendered


# --- Property 1c: invariant violations raise (R4.2) ----------------------------

@given(value=values, unit=units, reason=st.text(min_size=1))
@settings(max_examples=100, deadline=None)
def test_real_with_a_reason_raises(value, unit, reason):
    """Constructing a REAL MarkedValue with a non-None reason is rejected."""
    # The implementation enforces the invariant with `assert`, so the violation
    # surfaces as AssertionError; tolerate ValueError too.
    with pytest.raises((AssertionError, ValueError)):
        MarkedValue(value, Marker.REAL, reason, unit)


@given(
    value=values,
    unit=units,
    marker=st.sampled_from([Marker.PLACEHOLDER, Marker.ESTIMATE]),
    reason=st.sampled_from([None, ""]),
)
@settings(max_examples=100, deadline=None)
def test_nonreal_without_reason_raises(value, unit, marker, reason):
    """PLACEHOLDER/ESTIMATE with reason None or an empty string is rejected."""
    with pytest.raises((AssertionError, ValueError)):
        MarkedValue(value, marker, reason, unit)
