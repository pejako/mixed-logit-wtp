"""Export synthetic choice data to CSV for cross-language reproducibility.

Run this from python/ to generate the canonical dataset that the R parity
layer reads:

    python -m mixedlogit.export_csv

It writes:
  - r/data/synthetic_choices.csv  : long-format choice data
  - r/data/ground_truth.json      : the true population parameters

Both sides use seed=42 so any cross-language comparison is on identical data.
"""

from __future__ import annotations

import json
from pathlib import Path

from mixedlogit.dgp import default_config, simulate_choices


def export(out_dir: Path | str | None = None, seed: int = 42) -> None:
    """Write the canonical synthetic dataset and ground-truth metadata.

    Parameters
    ----------
    out_dir : path, optional
        Directory to write to. Defaults to ``../r/data`` relative to this
        file (the v0.2 R parity layer's data directory).
    seed : int
        DGP seed. Default 42 matches the R parity test fixtures.
    """
    if out_dir is None:
        # Resolve to ../../../r/data relative to this file
        # (python/mixedlogit/export_csv.py -> ../../../r/data)
        out_dir = Path(__file__).resolve().parents[2] / "r" / "data"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = default_config()
    df = simulate_choices(cfg, seed=seed)

    csv_path = out_dir / "synthetic_choices.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df):,} rows)")

    truth = df.attrs["true_params"]
    truth["export_seed"] = seed
    truth_path = out_dir / "ground_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2))
    print(f"Wrote {truth_path}")


if __name__ == "__main__":
    export()
