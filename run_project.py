"""Command-line entry point for the ABIDE connectome GCN project."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from abide_gcn.data import prepare_abide_cache
from abide_gcn.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ABIDE CC200 connectomes and compare site-held-out models."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root directory for Nilearn download cache and prepared arrays.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for metrics and figures.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device used for GCN training.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild connectome arrays from downloaded ROI time series.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = args.data_dir / "processed"
    cache_path = processed_dir / "abide_cc200_graphs.npz"
    dataset = prepare_abide_cache(
        data_dir=args.data_dir / "raw",
        cache_path=cache_path,
        top_k=args.top_k,
        rebuild=args.rebuild_cache,
    )
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


if __name__ == "__main__":
    main()
