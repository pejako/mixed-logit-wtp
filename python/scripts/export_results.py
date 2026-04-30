"""Export Python-side estimation results as JSON for cross-language parity.

Reads the canonical synthetic dataset, fits MNL and MXL on the Python side,
and writes the estimates to a JSON file the R parity comparison helper can
read.

Run from python/ after generating the CSV:

    python -m mixedlogit.export_csv
    python scripts/export_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mixedlogit.dgp import default_config
from mixedlogit.mnl import fit_mnl
from mixedlogit.mxl import fit_mxl


def export_results(out_dir: Path | str | None = None) -> None:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[2] / "r" / "data"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "synthetic_choices.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python -m mixedlogit.export_csv` first."
        )
    df = pd.read_csv(csv_path)

    cfg = default_config()
    attr_names = ["price", "quality", "brand_known"]

    print("Fitting MNL...")
    mnl = fit_mnl(df, attr_names=attr_names)
    mnl_payload = {
        "model": "MNL",
        "loglik": mnl.loglik,
        "loglik_null": mnl.loglik_null,
        "mcfadden_r2": mnl.mcfadden_r2,
        "n_observations": mnl.n_observations,
        "coefficients": dict(zip(mnl.coef_names, mnl.coefficients, strict=True)),
        "std_errors": dict(zip(mnl.coef_names, mnl.std_errors, strict=True)),
    }
    out_mnl = out_dir / "python_mnl_results.json"
    out_mnl.write_text(json.dumps(mnl_payload, indent=2))
    print(f"Wrote {out_mnl}")

    print("Fitting MXL (this takes ~30s)...")
    mxl = fit_mxl(df, cfg.attributes, n_draws=200, halton_seed=0)
    mxl_payload = {
        "model": "MXL",
        "loglik": mxl.loglik,
        "loglik_null": mxl.loglik_null,
        "mcfadden_r2": mxl.mcfadden_r2,
        "n_observations": mxl.n_observations,
        "n_individuals": mxl.n_individuals,
        "n_draws": mxl.n_draws,
        "coefficients": dict(zip(mxl.coef_names, mxl.coefficients, strict=True)),
        "std_errors": dict(zip(mxl.coef_names, mxl.std_errors, strict=True)),
    }
    out_mxl = out_dir / "python_mxl_results.json"
    out_mxl.write_text(json.dumps(mxl_payload, indent=2))
    print(f"Wrote {out_mxl}")


if __name__ == "__main__":
    export_results()
