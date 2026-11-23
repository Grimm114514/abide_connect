"""Download ABIDE PCP ROI time series and prepare connectome graph arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_abide_pcp


N_ROIS = 200


@dataclass
class ConnectomeDataset:
    """Arrays required by baseline models and the graph classifier."""

    subject_ids: np.ndarray
    file_ids: np.ndarray
    sites: np.ndarray
    labels: np.ndarray
    covariates: np.ndarray
    node_features: np.ndarray
    positive_adjacency: np.ndarray
    negative_adjacency: np.ndarray
    fc_edges: np.ndarray


def validate_dataset(dataset: ConnectomeDataset) -> None:
    """Validate cached array shapes, identifiers, labels, and numeric values."""

    sample_count = len(dataset.labels)
    if sample_count == 0:
        raise ValueError("The prepared dataset is empty.")

    one_dimensional_arrays = {
        "subject_ids": dataset.subject_ids,
        "file_ids": dataset.file_ids,
        "sites": dataset.sites,
        "labels": dataset.labels,
    }
    for name, values in one_dimensional_arrays.items():
        if values.ndim != 1 or len(values) != sample_count:
            raise ValueError(
                f"{name} must be one-dimensional with {sample_count} rows, "
                f"got shape {values.shape}."
            )

    if len(np.unique(dataset.subject_ids)) != sample_count:
        raise ValueError("The prepared dataset contains duplicate subject IDs.")
    if (
        np.any(dataset.subject_ids == "")
        or np.any(dataset.file_ids == "")
        or np.any(dataset.file_ids == "no_filename")
        or np.any(dataset.sites == "")
    ):
        raise ValueError("Subject IDs, file IDs, and site IDs must be valid.")

    label_values = np.unique(dataset.labels)
    if not np.all(np.isin(label_values, [0, 1])):
        raise ValueError(f"Labels must be binary 0/1, got {label_values.tolist()}.")
    if label_values.size != 2:
        raise ValueError("The prepared dataset must contain both diagnosis groups.")

    expected_edge_count = N_ROIS * (N_ROIS - 1) // 2
    expected_shapes = {
        "covariates": (sample_count, 3),
        "node_features": (sample_count, N_ROIS, 5),
        "positive_adjacency": (sample_count, N_ROIS, N_ROIS),
        "negative_adjacency": (sample_count, N_ROIS, N_ROIS),
        "fc_edges": (sample_count, expected_edge_count),
    }
    arrays = {
        "covariates": dataset.covariates,
        "node_features": dataset.node_features,
        "positive_adjacency": dataset.positive_adjacency,
        "negative_adjacency": dataset.negative_adjacency,
        "fc_edges": dataset.fc_edges,
    }
    for name, expected_shape in expected_shapes.items():
        values = arrays[name]
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinite values.")

    if np.any(dataset.positive_adjacency < 0) or np.any(
        dataset.negative_adjacency < 0
    ):
        raise ValueError("Positive and negative adjacency values must be nonnegative.")


def _number(value: object) -> float:
    """Convert a phenotype cell to a finite float, returning NaN if invalid."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _node_time_series_features(time_series: np.ndarray) -> np.ndarray:
    """Create five scale-sensitive time-series summaries for each ROI."""

    mean = time_series.mean(axis=0)
    centered = time_series - mean
    variance = np.mean(centered**2, axis=0)
    standard_deviation = np.sqrt(np.maximum(variance, 1e-8))
    mean_absolute = np.mean(np.abs(time_series), axis=0)
    skewness = np.mean(centered**3, axis=0) / (standard_deviation**3)
    excess_kurtosis = (
        np.mean(centered**4, axis=0) / (standard_deviation**4) - 3.0
    )

    features = np.column_stack(
        [mean, standard_deviation, mean_absolute, skewness, excess_kurtosis]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def _normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Apply symmetric degree normalization to a nonnegative adjacency."""

    degree = adjacency.sum(axis=1)
    inverse_sqrt_degree = np.zeros_like(degree)
    connected = degree > 0
    inverse_sqrt_degree[connected] = 1.0 / np.sqrt(degree[connected])
    normalized = (
        inverse_sqrt_degree[:, None]
        * adjacency
        * inverse_sqrt_degree[None, :]
    )
    return normalized.astype(np.float32)


def _connectome(
    time_series: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Fisher-z edges and normalized positive/negative graph channels."""

    values = np.asarray(time_series, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("ROI time series must be a two-dimensional array.")
    if values.shape[1] != N_ROIS and values.shape[0] == N_ROIS:
        values = values.T
    if values.shape[1] != N_ROIS or values.shape[0] < 20:
        raise ValueError(
            f"Expected at least 20 time points and {N_ROIS} ROIs, got {values.shape}."
        )

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(
        correlation, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    correlation = np.clip(correlation, -0.999999, 0.999999)
    np.fill_diagonal(correlation, 0.0)

    upper_triangle = np.triu_indices(N_ROIS, k=1)
    fisher_z_edges = np.arctanh(correlation[upper_triangle]).astype(np.float32)

    neighbors_per_node = min(max(int(top_k), 1), N_ROIS - 1)
    strongest_neighbors = np.argpartition(
        np.abs(correlation), -neighbors_per_node, axis=1
    )[:, -neighbors_per_node:]
    mask = np.zeros_like(correlation, dtype=bool)
    rows = np.arange(N_ROIS)[:, None]
    mask[rows, strongest_neighbors] = True
    mask = np.logical_or(mask, mask.T)
    np.fill_diagonal(mask, False)

    sparse_correlation = np.where(mask, correlation, 0.0)
    positive = np.maximum(sparse_correlation, 0.0)
    negative = np.maximum(-sparse_correlation, 0.0)
    positive = _normalize_adjacency(positive)
    negative = _normalize_adjacency(negative)
    return fisher_z_edges, positive, negative


def load_cache(cache_path: Path, expected_top_k: int) -> ConnectomeDataset:
    """Load the compressed graph cache without enabling pickle loading."""

    with np.load(cache_path, allow_pickle=False) as cache:
        cached_top_k = int(cache["top_k"])
        if cached_top_k != expected_top_k:
            raise ValueError(
                f"The cache uses top_k={cached_top_k}, but top_k={expected_top_k} "
                "was requested. Rebuild it with --rebuild-cache."
            )
        dataset = ConnectomeDataset(
            subject_ids=cache["subject_ids"],
            file_ids=cache["file_ids"],
            sites=cache["sites"],
            labels=cache["labels"],
            covariates=cache["covariates"],
            node_features=cache["node_features"],
            positive_adjacency=cache["positive_adjacency"],
            negative_adjacency=cache["negative_adjacency"],
            fc_edges=cache["fc_edges"],
        )
    validate_dataset(dataset)
    return dataset


def _save_participant_table(
    dataset: ConnectomeDataset, participant_path: Path
) -> None:
    participant_table = pd.DataFrame(
        {
            "subject_id": dataset.subject_ids,
            "file_id": dataset.file_ids,
            "site": dataset.sites,
            "label_asd": dataset.labels,
            "age_at_scan": dataset.covariates[:, 0],
            "sex_binary": dataset.covariates[:, 1],
            "mean_fd": dataset.covariates[:, 2],
        }
    )
    participant_path.parent.mkdir(parents=True, exist_ok=True)
    participant_table.to_csv(participant_path, index=False)
    site_counts = (
        participant_table.groupby(["site", "label_asd"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
        .rename(columns={0: "n_control", 1: "n_asd"})
        .reset_index()
    )
    site_counts.to_csv(
        participant_path.with_name("site_group_counts.csv"), index=False
    )


def prepare_abide_cache(
    data_dir: Path,
    cache_path: Path,
    top_k: int = 20,
    rebuild: bool = False,
) -> ConnectomeDataset:
    """Fetch quality-checked ABIDE PCP CC200 data and create a reusable cache."""

    if top_k < 1 or top_k >= N_ROIS:
        raise ValueError(f"top_k must be between 1 and {N_ROIS - 1}.")
    if cache_path.exists() and not rebuild:
        print(f"Loading prepared data cache from {cache_path}")
        dataset = load_cache(cache_path, expected_top_k=top_k)
        _save_participant_table(
            dataset, cache_path.with_name("participants.csv")
        )
        return dataset

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print("Fetching ABIDE PCP CC200 ROI time series. The first run needs internet.")
    fetched = fetch_abide_pcp(
        data_dir=str(data_dir),
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=["rois_cc200"],
        quality_checked=True,
        verbose=1,
    )

    phenotypes = pd.DataFrame(fetched.phenotypic).reset_index(drop=True)
    time_series_list = fetched.rois_cc200
    if len(phenotypes) != len(time_series_list):
        raise RuntimeError(
            "Nilearn returned different numbers of phenotype rows and ROI time series."
        )

    subject_ids: list[str] = []
    file_ids: list[str] = []
    sites: list[str] = []
    labels: list[int] = []
    covariates: list[list[float]] = []
    node_features: list[np.ndarray] = []
    positive_adjacencies: list[np.ndarray] = []
    negative_adjacencies: list[np.ndarray] = []
    fc_edges: list[np.ndarray] = []
    skipped = 0

    for row, raw_time_series in zip(
        phenotypes.to_dict(orient="records"), time_series_list
    ):
        diagnosis = _number(row.get("DX_GROUP"))
        age = _number(row.get("AGE_AT_SCAN"))
        sex_code = _number(row.get("SEX"))
        mean_fd = _number(row.get("func_mean_fd"))
        file_id = str(row.get("FILE_ID", "")).strip()
        site = str(row.get("SITE_ID", "")).strip()
        subject_id = str(row.get("SUB_ID", file_id)).strip()

        if (
            diagnosis not in (1.0, 2.0)
            or not np.isfinite(age)
            or sex_code not in (1.0, 2.0)
            or not np.isfinite(mean_fd)
            or mean_fd < 0
            or not file_id
            or file_id == "no_filename"
            or not site
        ):
            skipped += 1
            continue

        values = np.asarray(raw_time_series, dtype=np.float32)
        if values.ndim != 2:
            skipped += 1
            continue
        if values.shape[1] != N_ROIS and values.shape[0] == N_ROIS:
            values = values.T
        if values.shape[1] != N_ROIS or values.shape[0] < 20:
            skipped += 1
            continue

        try:
            node_x = _node_time_series_features(values)
            edges, positive, negative = _connectome(values, top_k=top_k)
        except (FloatingPointError, ValueError):
            skipped += 1
            continue

        subject_ids.append(subject_id)
        file_ids.append(file_id)
        sites.append(site)
        labels.append(1 if diagnosis == 1.0 else 0)
        covariates.append([age, 1.0 if sex_code == 2.0 else 0.0, mean_fd])
        node_features.append(node_x)
        positive_adjacencies.append(positive)
        negative_adjacencies.append(negative)
        fc_edges.append(edges)

    if not labels:
        raise RuntimeError("No usable ABIDE subjects remained after data checks.")

    dataset = ConnectomeDataset(
        subject_ids=np.asarray(subject_ids, dtype=str),
        file_ids=np.asarray(file_ids, dtype=str),
        sites=np.asarray(sites, dtype=str),
        labels=np.asarray(labels, dtype=np.int64),
        covariates=np.asarray(covariates, dtype=np.float32),
        node_features=np.stack(node_features).astype(np.float32),
        positive_adjacency=np.stack(positive_adjacencies).astype(np.float32),
        negative_adjacency=np.stack(negative_adjacencies).astype(np.float32),
        fc_edges=np.stack(fc_edges).astype(np.float32),
    )
    validate_dataset(dataset)

    np.savez_compressed(
        cache_path,
        subject_ids=dataset.subject_ids,
        file_ids=dataset.file_ids,
        sites=dataset.sites,
        labels=dataset.labels,
        covariates=dataset.covariates,
        node_features=dataset.node_features,
        positive_adjacency=dataset.positive_adjacency,
        negative_adjacency=dataset.negative_adjacency,
        fc_edges=dataset.fc_edges,
        top_k=np.asarray(top_k, dtype=np.int64),
    )

    _save_participant_table(
        dataset, cache_path.with_name("participants.csv")
    )
    group_counts = (
        pd.Series(dataset.labels).value_counts().sort_index().to_dict()
    )
    print(
        f"Prepared {len(dataset.labels)} subjects across "
        f"{len(np.unique(dataset.sites))} sites. Group counts: {group_counts}"
    )
    print(f"Skipped {skipped} rows with missing metadata or invalid ROI arrays.")
    print(f"Saved graph cache to {cache_path}")
    return dataset
