"""A code change must not silently alter books, features, labels, fills, cash or metrics."""

from __future__ import annotations

import json

import pytest

from tests.regression.outputs import EXPECTED, compute_outputs


def test_pipeline_outputs_match_pinned_fixture():
    if not EXPECTED.exists():
        pytest.fail("expected_outputs.json missing; run scripts/update_regression_fixtures.py")
    expected = json.loads(EXPECTED.read_text())
    got = json.loads(json.dumps(compute_outputs()))
    for key in expected:
        assert got[key] == expected[key], f"regression output changed: {key}"
