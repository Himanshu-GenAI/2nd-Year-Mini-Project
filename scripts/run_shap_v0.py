"""CLI script to compute Model v0 global and High/Critical local SHAP explanations."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain.shap_v0 import run_model_v0_shap  # noqa: E402


def main() -> None:
    res = run_model_v0_shap()
    print(f"Route: {res['route']}")
    print(f"Model version: {res['model_version']}")
    print(f"Base value (log-odds): {res['base_value']:.6f}")
    print(
        f"Max additivity diff: 1k={res['max_diff_1k']:.2e}, "
        f"20k={res['max_diff_20k']:.2e}, local={res['max_diff_local']:.2e}"
    )
    print(f"Local SHAP rows: {res['local_rows']:,} ({res['local_cell_days']:,} High/Critical cell-days x 5)")
    print(f"Runtime: {res['runtime_seconds']:.2f}s")
    print("\nGlobal SHAP ranking:")
    print(res["global_summary"].to_string(index=False))


if __name__ == "__main__":
    main()

