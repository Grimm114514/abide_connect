"""Download ABIDE data and build the reusable connectome feature cache."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ABIDE PCP CC200 time series and prepare graph features."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild derived graph features while reusing downloaded source files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from abide_gcn.data import prepare_abide_cache

    cache_path = args.data_dir / "processed" / "abide_cc200_graphs.npz"
    prepare_abide_cache(
        data_dir=args.data_dir / "raw",
        cache_path=cache_path,
        top_k=args.top_k,
        rebuild=args.rebuild_cache,
    )


if __name__ == "__main__":
    main()
