"""Run site-held-out baselines and GCN, then save metrics and figures."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import ConnectomeDataset, N_ROIS
from .model import ConnectomeGCN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    sensitivity = (
        true_positive / positive_count if positive_count else float("nan")
    )
    specificity = (
        true_negative / negative_count if negative_count else float("nan")
    )
    auroc = (
        float(roc_auc_score(labels, probabilities))
        if np.unique(labels).size == 2
        else float("nan")
    )
    return {
        "auroc": auroc,
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def _fit_logistic_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=500,
            solver="lbfgs",
        ),
    )
    model.fit(train_features, train_labels)
    return model.predict_proba(test_features)[:, 1]


def _scale_from_training(
    training_values: np.ndarray, all_values: np.ndarray
) -> np.ndarray:
    mean = training_values.mean(axis=0)
    standard_deviation = training_values.std(axis=0)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    return ((all_values - mean) / standard_deviation).astype(np.float32)


def _make_loader(
    node_features: np.ndarray,
    positive_adjacency: np.ndarray,
    negative_adjacency: np.ndarray,
    covariates: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(node_features),
        torch.from_numpy(positive_adjacency),
        torch.from_numpy(negative_adjacency),
        torch.from_numpy(covariates),
        torch.from_numpy(labels.astype(np.int64)),
    )
    return DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _move_batch(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor.to(device, non_blocking=device.type == "cuda")
        for tensor in batch
    )


def _evaluate_gcn(
    model: ConnectomeGCN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    loss_function = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            node_x, positive, negative, covariates, labels = _move_batch(
                batch, device
            )
            logits = model(node_x, positive, negative, covariates)
            losses.append(float(loss_function(logits, labels).item()))
            probabilities.append(
                torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            )
    return float(np.mean(losses)), np.concatenate(probabilities)


def _fit_gcn_for_site_fold(
    dataset: ConnectomeDataset,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    device: torch.device,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    hidden_dim: int,
) -> tuple[np.ndarray, int, float]:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=seed
    )
    relative_train, relative_validation = next(
        splitter.split(
            np.zeros(len(outer_train)),
            dataset.labels[outer_train],
            groups=dataset.sites[outer_train],
        )
    )
    training_indices = outer_train[relative_train]
    validation_indices = outer_train[relative_validation]

    node_values = dataset.node_features
    training_nodes = node_values[training_indices].reshape(
        -1, node_values.shape[-1]
    )
    node_mean = training_nodes.mean(axis=0)
    node_std = training_nodes.std(axis=0)
    node_std[node_std < 1e-6] = 1.0
    scaled_nodes = ((node_values - node_mean) / node_std).astype(np.float32)

    scaled_covariates = _scale_from_training(
        dataset.covariates[training_indices], dataset.covariates
    )

    train_loader = _make_loader(
        scaled_nodes[training_indices],
        dataset.positive_adjacency[training_indices],
        dataset.negative_adjacency[training_indices],
        scaled_covariates[training_indices],
        dataset.labels[training_indices],
        batch_size=batch_size,
        shuffle=True,
        device=device,
    )
    validation_loader = _make_loader(
        scaled_nodes[validation_indices],
        dataset.positive_adjacency[validation_indices],
        dataset.negative_adjacency[validation_indices],
        scaled_covariates[validation_indices],
        dataset.labels[validation_indices],
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )

    class_counts = np.bincount(dataset.labels[training_indices], minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(
            "The site-grouped inner training split has only one diagnosis class."
        )
    class_weights = len(training_indices) / np.maximum(2.0 * class_counts, 1.0)
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    model = ConnectomeGCN(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            node_x, positive, negative, covariates, labels = _move_batch(
                batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(node_x, positive, negative, covariates)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

        validation_loss, _ = _evaluate_gcn(
            model, validation_loader, device
        )
        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    selected_epoch_count = max(best_epoch, 1)
    seed_everything(seed)
    all_training_nodes = dataset.node_features[outer_train].reshape(
        -1, dataset.node_features.shape[-1]
    )
    full_node_mean = all_training_nodes.mean(axis=0)
    full_node_std = all_training_nodes.std(axis=0)
    full_node_std[full_node_std < 1e-6] = 1.0
    full_scaled_nodes = (
        (dataset.node_features - full_node_mean) / full_node_std
    ).astype(np.float32)
    full_scaled_covariates = _scale_from_training(
        dataset.covariates[outer_train], dataset.covariates
    )
    full_train_loader = _make_loader(
        full_scaled_nodes[outer_train],
        dataset.positive_adjacency[outer_train],
        dataset.negative_adjacency[outer_train],
        full_scaled_covariates[outer_train],
        dataset.labels[outer_train],
        batch_size=batch_size,
        shuffle=True,
        device=device,
    )
    full_class_counts = np.bincount(
        dataset.labels[outer_train], minlength=2
    )
    full_class_weights = len(outer_train) / np.maximum(
        2.0 * full_class_counts, 1.0
    )
    full_loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(
            full_class_weights, dtype=torch.float32, device=device
        )
    )
    model = ConnectomeGCN(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )
    for _ in range(selected_epoch_count):
        model.train()
        for batch in full_train_loader:
            node_x, positive, negative, covariates, labels = _move_batch(
                batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(node_x, positive, negative, covariates)
            loss = full_loss_function(logits, labels)
            loss.backward()
            optimizer.step()

    test_loader = _make_loader(
        full_scaled_nodes[outer_test],
        dataset.positive_adjacency[outer_test],
        dataset.negative_adjacency[outer_test],
        full_scaled_covariates[outer_test],
        dataset.labels[outer_test],
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )
    _, test_probabilities = _evaluate_gcn(model, test_loader, device)
    return test_probabilities, best_epoch, best_validation_loss


def _connectivity_matrix(
    fisher_z_edges: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((N_ROIS, N_ROIS), dtype=np.float32)
    upper_triangle = np.triu_indices(N_ROIS, k=1)
    matrix[upper_triangle] = fisher_z_edges
    matrix[(upper_triangle[1], upper_triangle[0])] = fisher_z_edges
    return matrix


def _save_figures(
    dataset: ConnectomeDataset,
    predictions: pd.DataFrame,
    site_metrics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    asd_edges = dataset.fc_edges[dataset.labels == 1].mean(axis=0)
    control_edges = dataset.fc_edges[dataset.labels == 0].mean(axis=0)
    matrices = [
        _connectivity_matrix(asd_edges),
        _connectivity_matrix(control_edges),
        _connectivity_matrix(asd_edges - control_edges),
    ]
    titles = [
        "ASD mean Fisher-z connectivity",
        "Control mean Fisher-z connectivity",
        "ASD minus control",
    ]

    mean_limit = max(
        float(np.quantile(np.abs(matrices[0]), 0.99)),
        float(np.quantile(np.abs(matrices[1]), 0.99)),
        0.05,
    )
    difference_limit = max(
        float(np.quantile(np.abs(matrices[2]), 0.99)),
        0.01,
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for index, (axis, matrix, title) in enumerate(
        zip(axes, matrices, titles)
    ):
        color_limit = difference_limit if index == 2 else mean_limit
        image = axis.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
        )
        axis.set_title(title)
        axis.set_xlabel("ROI")
        axis.set_ylabel("ROI")
        figure.colorbar(image, ax=axis, shrink=0.8, label="Fisher z")
    figure.tight_layout()
    figure.savefig(figure_dir / "mean_fc_by_group.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    for column, label in (
        ("prob_covariates_logistic", "Covariates logistic"),
        ("prob_fc_covariates_logistic", "FC + covariates logistic"),
        ("prob_signed_gcn", "Signed GCN"),
    ):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            predictions["label_asd"],
            predictions[column],
        )
        auc = roc_auc_score(predictions["label_asd"], predictions[column])
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{label} (AUC {auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Pooled out-of-site ROC")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(figure_dir / "roc_oof.png", dpi=180)
    plt.close(figure)

    pivot = site_metrics.pivot(
        index="site", columns="model", values="auroc"
    )
    figure, axis = plt.subplots(figsize=(13, 6))
    pivot.plot(kind="bar", ax=axis, width=0.8)
    axis.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    axis.set_ylim(0, 1)
    axis.set_ylabel("AUROC")
    axis.set_xlabel("Held-out site")
    axis.set_title("Site-held-out AUROC")
    axis.legend(title="Model")
    figure.tight_layout()
    figure.savefig(figure_dir / "site_auc.png", dpi=180)
    plt.close(figure)


def run_experiment(
    dataset: ConnectomeDataset,
    output_dir: Path,
    device_name: str = "auto",
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 12,
    hidden_dim: int = 32,
    seed: int = 42,
) -> pd.DataFrame:
    """Run leave-one-site-out evaluation and write all result tables/figures."""

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access a CUDA device. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )

    if device.type == "cuda":
        print(f"Training the GCN on {torch.cuda.get_device_name(device)}")
    else:
        print("Training the GCN on CPU")

    if len(np.unique(dataset.sites)) < 3:
        raise ValueError("At least three acquisition sites are needed.")
    if np.unique(dataset.labels).size != 2:
        raise ValueError("The prepared dataset must contain both diagnosis groups.")

    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    covariate_probabilities = np.full(len(dataset.labels), np.nan)
    fc_probabilities = np.full(len(dataset.labels), np.nan)
    gcn_probabilities = np.full(len(dataset.labels), np.nan)
    site_metric_rows: list[dict[str, object]] = []
    gcn_training_rows: list[dict[str, object]] = []

    outer_splitter = LeaveOneGroupOut()
    folds = outer_splitter.split(
        np.zeros(len(dataset.labels)),
        dataset.labels,
        groups=dataset.sites,
    )

    for fold_number, (outer_train, outer_test) in enumerate(folds, start=1):
        test_site_values = np.unique(dataset.sites[outer_test])
        if len(test_site_values) != 1:
            raise RuntimeError("Each outer fold must hold out exactly one site.")
        held_out_site = str(test_site_values[0])
        print(
            f"Fold {fold_number:02d}: held-out site {held_out_site}, "
            f"n={len(outer_test)}"
        )

        covariate_probabilities[outer_test] = _fit_logistic_model(
            dataset.covariates[outer_train],
            dataset.labels[outer_train],
            dataset.covariates[outer_test],
        )
        fc_features = np.column_stack(
            [dataset.fc_edges, dataset.covariates]
        )
        fc_probabilities[outer_test] = _fit_logistic_model(
            fc_features[outer_train],
            dataset.labels[outer_train],
            fc_features[outer_test],
        )

        gcn_seed = seed + fold_number
        fold_gcn_probabilities, best_epoch, validation_loss = (
            _fit_gcn_for_site_fold(
                dataset=dataset,
                outer_train=outer_train,
                outer_test=outer_test,
                device=device,
                seed=gcn_seed,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                hidden_dim=hidden_dim,
            )
        )
        gcn_probabilities[outer_test] = fold_gcn_probabilities
        gcn_training_rows.append(
            {
                "site": held_out_site,
                "best_epoch": best_epoch,
                "best_validation_loss": validation_loss,
            }
        )

        fold_probabilities = {
            "covariates_logistic": covariate_probabilities[outer_test],
            "fc_covariates_logistic": fc_probabilities[outer_test],
            "signed_gcn": gcn_probabilities[outer_test],
        }
        for model_name, probabilities in fold_probabilities.items():
            fold_metrics = _classification_metrics(
                dataset.labels[outer_test], probabilities
            )
            site_metric_rows.append(
                {
                    "site": held_out_site,
                    "model": model_name,
                    "n": len(outer_test),
                    **fold_metrics,
                }
            )

    prediction_table = pd.DataFrame(
        {
            "subject_id": dataset.subject_ids,
            "file_id": dataset.file_ids,
            "site": dataset.sites,
            "label_asd": dataset.labels,
            "prob_covariates_logistic": covariate_probabilities,
            "prob_fc_covariates_logistic": fc_probabilities,
            "prob_signed_gcn": gcn_probabilities,
        }
    )
    site_metrics = pd.DataFrame(site_metric_rows)
    summary_rows = []
    probability_columns = {
        "covariates_logistic": "prob_covariates_logistic",
        "fc_covariates_logistic": "prob_fc_covariates_logistic",
        "signed_gcn": "prob_signed_gcn",
    }
    for model_name, probability_column in probability_columns.items():
        summary_rows.append(
            {
                "model": model_name,
                "n_subjects": len(dataset.labels),
                "n_sites": len(np.unique(dataset.sites)),
                **_classification_metrics(
                    dataset.labels,
                    prediction_table[probability_column].to_numpy(),
                ),
            }
        )
    summary_metrics = pd.DataFrame(summary_rows)

    site_metrics.to_csv(results_dir / "metrics_by_site.csv", index=False)
    summary_metrics.to_csv(results_dir / "summary_metrics.csv", index=False)
    prediction_table.to_csv(
        results_dir / "oof_predictions.csv", index=False
    )
    pd.DataFrame(gcn_training_rows).to_csv(
        results_dir / "gcn_training_by_site.csv", index=False
    )
    _save_figures(dataset, prediction_table, site_metrics, figure_dir)

    print("\nPooled out-of-site metrics")
    print(summary_metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved tables to {results_dir}")
    print(f"Saved figures to {figure_dir}")
    return summary_metrics
