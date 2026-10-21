"""Regenerate tests/regression/expected_outputs.json after an INTENTIONAL behaviour change.

Usage: python scripts/update_regression_fixtures.py   (from the repository root)
Document why the outputs changed in the commit message.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.regression.outputs import EXPECTED, compute_outputs  # noqa: E402

if __name__ == "__main__":
    out = compute_outputs()
    EXPECTED.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {EXPECTED}")
    print(json.dumps(out, indent=2))
