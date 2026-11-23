"""Run site-held-out model evaluation and fit a reusable final GCN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the baselines and GCN from a prepared cache."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from abide_gcn.data import load_cache
    from abide_gcn.experiment import run_experiment, train_final_gcn

    cache_path = args.data_dir / "processed" / "abide_cc200_graphs.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Prepared cache not found at {cache_path}. Run download_data.py first."
        )
    dataset = load_cache(cache_path, expected_top_k=args.top_k)
    run_experiment(
        dataset=dataset,
        output_dir=args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )

    training_log = pd.read_csv(args.output_dir / "results" / "gcn_training_by_site.csv")
    selected_epochs = max(1, int(np.median(training_log["best_epoch"])))
    checkpoint_path = args.output_dir / "models" / "signed_gcn.pt"
    train_final_gcn(
        dataset=dataset,
        checkpoint_path=checkpoint_path,
        device_name=args.device,
        batch_size=args.batch_size,
        epochs=selected_epochs,
        hidden_dim=args.hidden_dim,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(f"Saved final full-data GCN to {checkpoint_path}")


if __name__ == "__main__":
    main()
