"""Copy an experiment's report, figures and small result files into reports/results/<id>/ for version control.

Experiment directories under reports/experiments/ are git-ignored (they contain large per-fill artifacts).
This keeps only what a reader needs: the report, figures, results.json, trials.csv, configs and manifests.

Usage: python scripts/publish_results.py <experiment_id> [<experiment_id> ...]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SRC = Path("reports/experiments")
DST = Path("reports/results")
FILES = [
    "report.md",
    "results.json",
    "trials.csv",
    "config_snapshot.yaml",
    "config_selected.yaml",
    "manifests.json",
]


def publish(exp_id: str) -> Path:
    src, dst = SRC / exp_id, DST / exp_id
    if not (src / "report.md").exists():
        raise FileNotFoundError(f"{src}/report.md not found; run `amm report --experiment {exp_id}` first")
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "figures").mkdir(parents=True)
    for name in FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    for png in sorted((src / "figures").glob("*.png")):
        shutil.copy2(png, dst / "figures" / png.name)
    return dst


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for exp in sys.argv[1:]:
        print(f"published {publish(exp)}")
