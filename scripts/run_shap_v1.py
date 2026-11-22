"""Execute Model v1 SHAP pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain.shap_v1 import run_model_v1_shap  # noqa: E402


def main() -> None:
    res = run_model_v1_shap()
    print("Model v1 SHAP completed successfully!")
    print(f"Runtime: {res['runtime_s']:.2f}s")
    print(f"Local High/Critical cell-days: {res['local_cell_days']}")
    print(f"Local grouped rows: {res['local_rows_count']}")
    print(f"Max additivity diff (1k): {res['max_diff_1k']:.2e}")
    print(f"Max additivity diff (20k): {res['max_diff_20k']:.2e}")
    print(f"Max additivity diff (local): {res['max_diff_local_grouped']:.2e}")


if __name__ == "__main__":
    main()
