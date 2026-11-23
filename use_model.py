"""Apply a saved GCN checkpoint to a connectome feature cache."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict ASD probabilities from a prepared connectome feature cache."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models" / "signed_gcn.pt",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "abide_cc200_graphs.npz",
        help="NPZ cache with node_features, positive_adjacency, negative_adjacency, covariates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "results" / "predictions.csv",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from abide_gcn.model import ConnectomeGCN

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch environment.")
    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {args.model}. Run train_model.py first.")
    if not args.input.exists():
        raise FileNotFoundError(f"Feature cache not found at {args.input}. Run download_data.py first.")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    with np.load(args.input, allow_pickle=False) as cache:
        required = (
            "node_features",
            "positive_adjacency",
            "negative_adjacency",
            "covariates",
        )
        missing = [name for name in required if name not in cache]
        if missing:
            raise ValueError(f"Input cache is missing arrays: {', '.join(missing)}")
        node_features = cache["node_features"].astype(np.float32)
        positive = cache["positive_adjacency"].astype(np.float32)
        negative = cache["negative_adjacency"].astype(np.float32)
        covariates = cache["covariates"].astype(np.float32)
        if "subject_ids" in cache:
            subject_ids = cache["subject_ids"].astype(str)
        else:
            subject_ids = np.asarray(
                [f"subject_{index + 1:04d}" for index in range(len(node_features))]
            )
        if "top_k" in cache and int(cache["top_k"]) != int(checkpoint["top_k"]):
            raise ValueError(
                "Input cache and model use different top_k values. "
                "Prepare the input with the same --top-k used for training."
            )

    sample_count = len(node_features)
    if sample_count == 0 or args.batch_size < 1:
        raise ValueError("Input cache must contain subjects and batch size must be positive.")
    if (
        node_features.shape != (sample_count, 200, 5)
        or positive.shape != (sample_count, 200, 200)
        or negative.shape != (sample_count, 200, 200)
        or covariates.shape != (sample_count, 3)
        or len(subject_ids) != sample_count
    ):
        raise ValueError("Input feature arrays have incompatible shapes.")
    if not all(
        np.all(np.isfinite(values))
        for values in (node_features, positive, negative, covariates)
    ):
        raise ValueError("Input feature arrays contain NaN or infinite values.")

    node_features = (
        (node_features - checkpoint["node_mean"].numpy())
        / checkpoint["node_std"].numpy()
    ).astype(np.float32)
    covariates = (
        (covariates - checkpoint["covariate_mean"].numpy())
        / checkpoint["covariate_std"].numpy()
    ).astype(np.float32)

    model = ConnectomeGCN(hidden_dim=int(checkpoint["hidden_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, sample_count, args.batch_size):
            end = start + args.batch_size
            logits = model(
                torch.from_numpy(node_features[start:end]).to(device),
                torch.from_numpy(positive[start:end]).to(device),
                torch.from_numpy(negative[start:end]).to(device),
                torch.from_numpy(covariates[start:end]).to(device),
            )
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    asd_probability = np.concatenate(probabilities)
    output = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "probability_asd": asd_probability,
            "predicted_label": (asd_probability >= 0.5).astype(np.int64),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Predicted {sample_count} subjects. Saved results to {args.output}")


if __name__ == "__main__":
    main()
