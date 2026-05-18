import gc
import math
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DATA_FOLDER = Path("Data")

OUTPUT_ROOT = Path("outputs")
MODEL_OUTPUT_ROOT = OUTPUT_ROOT / "models"

ML_OUTPUT_FOLDER = MODEL_OUTPUT_ROOT / "visium_7class_neighborhood_classification"
ML_FIGURE_FOLDER = ML_OUTPUT_FOLDER / "figures"

LABEL_COLUMN_SOURCE = "Region"
LABEL_COLUMN = "brain_region_label"

MIN_GENES = 200
MIN_COUNTS = 500

N_HVG = 2000
TOP_N_GENES_PER_REGION = 20
STABLE_FRACTION_THRESHOLD = 0.50

NEIGHBOR_K_VALUES = [3, 5, 10]

ML_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
ML_FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)


def load_single_sample(h5ad_path: Path) -> tuple[str, sc.AnnData, dict]:
    sample_id = h5ad_path.stem

    print("\n" + "=" * 80)
    print("Loading sample:", sample_id)
    print("=" * 80)

    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()

    if LABEL_COLUMN_SOURCE not in adata.obs.columns:
        raise ValueError(
            f"{sample_id} is missing required label column '{LABEL_COLUMN_SOURCE}'. "
            f"Available columns: {adata.obs.columns.tolist()}"
        )

    if "spatial" not in adata.obsm:
        raise ValueError(f"{sample_id} is missing spatial coordinates in adata.obsm['spatial'].")

    adata.obs[LABEL_COLUMN] = adata.obs[LABEL_COLUMN_SOURCE].copy()

    valid_label_mask = (
        adata.obs[LABEL_COLUMN].notna()
        & (adata.obs[LABEL_COLUMN].astype(str).str.lower() != "nan")
        & (adata.obs[LABEL_COLUMN].astype(str).str.strip() != "")
    )

    sc.pp.calculate_qc_metrics(adata, inplace=True)

    adata = adata[
        (adata.obs["n_genes_by_counts"] >= MIN_GENES)
        & (adata.obs["total_counts"] >= MIN_COUNTS)
        & valid_label_mask
    ].copy()

    if adata.n_obs == 0:
        raise ValueError(f"{sample_id} has zero spots after filtering.")

    adata.obs[LABEL_COLUMN] = adata.obs[LABEL_COLUMN].astype(str)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.obs["sample_id"] = sample_id

    label_counts = (
        adata.obs[LABEL_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    summary = {
        "sample_id": sample_id,
        "spots_used": int(adata.n_obs),
        "genes_available": int(adata.n_vars),
        "labels_present": ", ".join(sorted(adata.obs[LABEL_COLUMN].unique())),
        "label_counts": str(label_counts),
    }

    print("Spots used:", adata.n_obs)
    print("Genes available:", adata.n_vars)
    print("Labels:", label_counts)

    return sample_id, adata, summary


def load_all_samples() -> tuple[dict[str, sc.AnnData], pd.DataFrame]:
    h5ad_files = sorted(DATA_FOLDER.glob("*.h5ad"))

    if len(h5ad_files) == 0:
        raise FileNotFoundError(f"No .h5ad files found in {DATA_FOLDER}")

    sample_adatas = {}
    summaries = []

    for h5ad_path in h5ad_files:
        sample_id, adata, summary = load_single_sample(h5ad_path)
        sample_adatas[sample_id] = adata
        summaries.append(summary)

    dataset_summary = pd.DataFrame(summaries)

    dataset_summary.to_csv(
        ML_OUTPUT_FOLDER / "neighborhood_ml_dataset_summary_all12.csv",
        index=False,
    )

    print("\nLoaded samples:", len(sample_adatas))
    print("Total spots:", int(dataset_summary["spots_used"].sum()))

    return sample_adatas, dataset_summary


def get_models() -> dict:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "linear_svm": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    LinearSVC(
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def select_hvg_from_training_data(
    train_adatas: list[sc.AnnData],
    fold_index: int,
    test_sample_id: str,
    neighbor_k: int,
) -> list[str]:
    train_combined = ad.concat(
        train_adatas,
        join="inner",
        label="concat_sample_id",
        keys=[adata.obs["sample_id"].iloc[0] for adata in train_adatas],
        index_unique="-",
    )

    sc.pp.highly_variable_genes(
        train_combined,
        flavor="seurat",
        n_top_genes=N_HVG,
    )

    hvg_genes = (
        train_combined.var[train_combined.var["highly_variable"]]
        .index
        .astype(str)
        .tolist()
    )

    if len(hvg_genes) == 0:
        raise ValueError(f"No HVG genes selected for fold {fold_index}.")

    feature_rows = pd.DataFrame(
        {
            "feature_mode": "hvg",
            "neighbor_k": neighbor_k,
            "fold": fold_index,
            "test_sample": test_sample_id,
            "gene": hvg_genes,
        }
    )

    feature_rows.to_csv(
        ML_OUTPUT_FOLDER / f"hvg_features_k{neighbor_k}_fold_{fold_index}_{test_sample_id}.csv",
        index=False,
    )

    del train_combined
    gc.collect()

    return hvg_genes


def discover_stable_genes_from_training_data(
    sample_adatas: dict[str, sc.AnnData],
    train_sample_ids: list[str],
    fold_index: int,
    test_sample_id: str,
    neighbor_k: int,
) -> list[str]:
    marker_rows = []
    layer_coverage_rows = []

    for sample_id in train_sample_ids:
        adata = sample_adatas[sample_id]

        sc.tl.rank_genes_groups(
            adata,
            groupby=LABEL_COLUMN,
            method="wilcoxon",
            use_raw=False,
        )

        labels = sorted(adata.obs[LABEL_COLUMN].astype(str).unique())

        for label in labels:
            layer_coverage_rows.append(
                {
                    "neighbor_k": neighbor_k,
                    "fold": fold_index,
                    "test_sample": test_sample_id,
                    "sample_id": sample_id,
                    "group": label,
                }
            )

            marker_df = sc.get.rank_genes_groups_df(
                adata,
                group=label,
            )

            marker_df = marker_df[
                ~marker_df["names"].astype(str).str.startswith("MT-")
            ].copy()

            marker_df = marker_df.head(TOP_N_GENES_PER_REGION)

            for _, row in marker_df.iterrows():
                marker_rows.append(
                    {
                        "neighbor_k": neighbor_k,
                        "fold": fold_index,
                        "test_sample": test_sample_id,
                        "sample_id": sample_id,
                        "group": label,
                        "gene": str(row["names"]),
                        "score": float(row["scores"]),
                        "logfoldchange": float(row["logfoldchanges"]),
                        "pvals_adj": float(row["pvals_adj"]),
                    }
                )

    marker_table = pd.DataFrame(marker_rows)
    layer_coverage = pd.DataFrame(layer_coverage_rows)

    if marker_table.empty:
        raise ValueError(
            f"No marker genes found in fold {fold_index} while testing {test_sample_id}."
        )

    layer_sample_coverage = (
        layer_coverage.groupby("group")["sample_id"]
        .nunique()
        .reset_index(name="sample_count_with_layer")
    )

    recurrence = (
        marker_table.groupby(["group", "gene"])
        .agg(
            sample_count=("sample_id", "nunique"),
            sample_ids=("sample_id", lambda values: ", ".join(sorted(set(values)))),
            mean_score=("score", "mean"),
            mean_logfoldchange=("logfoldchange", "mean"),
            best_pvals_adj=("pvals_adj", "min"),
        )
        .reset_index()
    )

    recurrence = recurrence.merge(
        layer_sample_coverage,
        on="group",
        how="left",
    )

    recurrence["recurrence_fraction"] = (
        recurrence["sample_count"] / recurrence["sample_count_with_layer"]
    )

    recurrence["minimum_samples_required"] = recurrence[
        "sample_count_with_layer"
    ].apply(
        lambda value: max(1, math.ceil(value * STABLE_FRACTION_THRESHOLD))
    )

    stable_recurrence = recurrence[
        recurrence["sample_count"] >= recurrence["minimum_samples_required"]
    ].copy()

    stable_recurrence = stable_recurrence.sort_values(
        [
            "group",
            "recurrence_fraction",
            "sample_count",
            "mean_logfoldchange",
        ],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    marker_table.to_csv(
        ML_OUTPUT_FOLDER / f"neighborhood_training_top_markers_k{neighbor_k}_fold_{fold_index}_{test_sample_id}.csv",
        index=False,
    )

    recurrence.to_csv(
        ML_OUTPUT_FOLDER / f"neighborhood_training_gene_recurrence_k{neighbor_k}_fold_{fold_index}_{test_sample_id}.csv",
        index=False,
    )

    stable_recurrence.to_csv(
        ML_OUTPUT_FOLDER / f"neighborhood_stable_gene_features_k{neighbor_k}_fold_{fold_index}_{test_sample_id}.csv",
        index=False,
    )

    stable_genes = sorted(stable_recurrence["gene"].astype(str).unique())

    if len(stable_genes) == 0:
        raise ValueError(
            f"No stable recurrent genes found in fold {fold_index} while testing {test_sample_id}. "
            "This means the threshold is too strict for this training split."
        )

    print(
        f"Fold {fold_index}, k={neighbor_k}, stable genes discovered from training samples only:",
        len(stable_genes),
    )

    return stable_genes


def extract_gene_matrix_sparse(
    adata: sc.AnnData,
    genes: list[str],
) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    available_pairs = [
        (index, gene)
        for index, gene in enumerate(genes)
        if gene in adata.var_names
    ]

    available_genes = [gene for _, gene in available_pairs]
    missing_genes = [gene for gene in genes if gene not in adata.var_names]

    if len(available_genes) == 0:
        raise ValueError("None of the selected genes were found in this h5ad file.")

    if len(available_genes) == len(genes):
        X = adata[:, genes].X

        if sparse.issparse(X):
            return X.tocsr(), available_genes, missing_genes

        return sparse.csr_matrix(np.asarray(X, dtype=np.float32)), available_genes, missing_genes

    X_available = adata[:, available_genes].X

    if sparse.issparse(X_available):
        X_available = X_available.tocoo()
    else:
        X_available = sparse.coo_matrix(np.asarray(X_available, dtype=np.float32))

    target_columns = np.array([index for index, _ in available_pairs], dtype=int)
    remapped_columns = target_columns[X_available.col]

    X_full = sparse.coo_matrix(
        (
            X_available.data,
            (X_available.row, remapped_columns),
        ),
        shape=(adata.n_obs, len(genes)),
    ).tocsr()

    return X_full, available_genes, missing_genes


def get_neighbor_indices(adata: sc.AnnData, neighbor_k: int) -> np.ndarray:
    coords = np.asarray(adata.obsm["spatial"])

    if coords.shape[0] <= 1:
        raise ValueError("At least two spots are required to calculate neighbors.")

    n_neighbors = min(neighbor_k + 1, coords.shape[0])

    nearest_neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    )

    nearest_neighbors.fit(coords)

    indices = nearest_neighbors.kneighbors(
        coords,
        return_distance=False,
    )

    return indices[:, 1:]


def build_neighborhood_features(
    adata: sc.AnnData,
    genes: list[str],
    neighbor_k: int,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, pd.DataFrame]:
    X_own, available_genes, missing_genes = extract_gene_matrix_sparse(
        adata=adata,
        genes=genes,
    )

    neighbor_indices = get_neighbor_indices(
        adata=adata,
        neighbor_k=neighbor_k,
    )

    actual_k = neighbor_indices.shape[1]

    if actual_k == 0:
        raise ValueError("No neighbors were found.")

    row_indices = np.repeat(np.arange(adata.n_obs), actual_k)
    column_indices = neighbor_indices.reshape(-1)
    values = np.full(row_indices.shape[0], 1.0 / actual_k, dtype=np.float32)

    neighbor_weight_matrix = sparse.coo_matrix(
        (
            values,
            (row_indices, column_indices),
        ),
        shape=(adata.n_obs, adata.n_obs),
    ).tocsr()

    X_neighbor_mean = neighbor_weight_matrix @ X_own

    X_combined = sparse.hstack(
        [X_own, X_neighbor_mean],
        format="csr",
    )

    y = adata.obs[LABEL_COLUMN].astype(str).to_numpy()

    feature_audit = pd.DataFrame(
        [
            {
                "sample_id": adata.obs["sample_id"].iloc[0],
                "neighbor_k": neighbor_k,
                "selected_genes": len(genes),
                "available_genes": len(available_genes),
                "missing_genes": len(missing_genes),
                "missing_gene_names": ", ".join(missing_genes),
                "feature_count_expected": len(genes) * 2,
                "feature_count_actual": X_combined.shape[1],
                "actual_neighbor_count": actual_k,
            }
        ]
    )

    return X_combined, y, neighbor_indices, feature_audit


def combine_training_data_with_neighbors(
    sample_adatas: dict[str, sc.AnnData],
    train_sample_ids: list[str],
    test_sample_id: str,
    genes: list[str],
    neighbor_k: int,
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
]:
    X_train_parts = []
    y_train_parts = []
    train_audit_rows = []

    for sample_id in train_sample_ids:
        X_sample, y_sample, _, feature_audit = build_neighborhood_features(
            adata=sample_adatas[sample_id],
            genes=genes,
            neighbor_k=neighbor_k,
        )

        X_train_parts.append(X_sample)
        y_train_parts.append(y_sample)
        train_audit_rows.append(feature_audit)

    X_train = sparse.vstack(X_train_parts).tocsr()
    y_train = np.concatenate(y_train_parts)

    X_test, y_test, test_neighbor_indices, test_audit = build_neighborhood_features(
        adata=sample_adatas[test_sample_id],
        genes=genes,
        neighbor_k=neighbor_k,
    )

    train_audit = pd.concat(train_audit_rows, ignore_index=True)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        test_neighbor_indices,
        train_audit,
        test_audit,
    )


def smooth_predictions(
    predictions: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    smoothed_predictions = []

    for spot_index in range(len(predictions)):
        neighbor_predictions = predictions[neighbor_indices[spot_index]]
        candidate_labels = np.concatenate(
            [
                np.array([predictions[spot_index]]),
                neighbor_predictions,
            ]
        )

        label_counts = pd.Series(candidate_labels).value_counts()
        max_count = label_counts.max()
        tied_labels = sorted(label_counts[label_counts == max_count].index.astype(str).tolist())

        if predictions[spot_index] in tied_labels:
            smoothed_predictions.append(predictions[spot_index])
        else:
            smoothed_predictions.append(tied_labels[0])

    return np.array(smoothed_predictions)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    accuracy = accuracy_score(y_true, y_pred)

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    feature_mode: str,
    neighbor_k: int,
    model_name: str,
    prediction_type: str,
) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest")
    plt.title(f"{feature_mode}, k={neighbor_k}, {model_name}, {prediction_type}")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    threshold = matrix.max() / 2.0 if matrix.max() > 0 else 0

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            color = "white" if value > threshold else "black"
            plt.text(
                column_index,
                row_index,
                value,
                ha="center",
                va="center",
                color=color,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    output_path = (
        ML_FIGURE_FOLDER
        / f"confusion_matrix_{feature_mode}_k{neighbor_k}_{model_name}_{prediction_type}.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def evaluate_feature_mode_and_neighbor_k(
    sample_adatas: dict[str, sc.AnnData],
    feature_mode: str,
    neighbor_k: int,
) -> pd.DataFrame:
    sample_ids = sorted(sample_adatas.keys())
    all_labels = sorted(
        {
            label
            for adata in sample_adatas.values()
            for label in adata.obs[LABEL_COLUMN].astype(str).unique()
        }
    )

    models = get_models()

    model_tracking = {
        model_name: {
            "y_true": [],
            "raw_predictions": [],
            "smoothed_predictions": [],
            "fold_rows": [],
        }
        for model_name in models
    }

    feature_audit_tables = []
    feature_frequency_rows = []

    for fold_index, test_sample_id in enumerate(sample_ids, start=1):
        print("\n" + "=" * 80)
        print(
            f"Feature mode: {feature_mode} | k={neighbor_k} | Fold {fold_index}/{len(sample_ids)} | Test sample: {test_sample_id}"
        )
        print("=" * 80)

        train_sample_ids = [
            sample_id
            for sample_id in sample_ids
            if sample_id != test_sample_id
        ]

        if feature_mode == "hvg":
            selected_genes = select_hvg_from_training_data(
                train_adatas=[
                    sample_adatas[sample_id]
                    for sample_id in train_sample_ids
                ],
                fold_index=fold_index,
                test_sample_id=test_sample_id,
                neighbor_k=neighbor_k,
            )
        elif feature_mode == "stable_genes":
            selected_genes = discover_stable_genes_from_training_data(
                sample_adatas=sample_adatas,
                train_sample_ids=train_sample_ids,
                fold_index=fold_index,
                test_sample_id=test_sample_id,
                neighbor_k=neighbor_k,
            )
        else:
            raise ValueError(f"Unknown feature mode: {feature_mode}")

        for gene in selected_genes:
            feature_frequency_rows.append(
                {
                    "feature_mode": feature_mode,
                    "neighbor_k": neighbor_k,
                    "fold": fold_index,
                    "test_sample": test_sample_id,
                    "gene": gene,
                }
            )

        (
            X_train,
            y_train,
            X_test,
            y_test,
            test_neighbor_indices,
            train_audit,
            test_audit,
        ) = combine_training_data_with_neighbors(
            sample_adatas=sample_adatas,
            train_sample_ids=train_sample_ids,
            test_sample_id=test_sample_id,
            genes=selected_genes,
            neighbor_k=neighbor_k,
        )

        feature_audit = pd.concat(
            [train_audit, test_audit],
            ignore_index=True,
        )

        feature_audit["feature_mode"] = feature_mode
        feature_audit["fold"] = fold_index
        feature_audit["test_sample"] = test_sample_id

        feature_audit_tables.append(feature_audit)

        for model_name, model_template in models.items():
            print(f"Training model: {model_name}")

            model = clone(model_template)

            model.fit(X_train, y_train)

            raw_predictions = model.predict(X_test)
            smoothed_predictions = smooth_predictions(
                predictions=raw_predictions,
                neighbor_indices=test_neighbor_indices,
            )

            raw_metrics = calculate_metrics(y_test, raw_predictions)
            smoothed_metrics = calculate_metrics(y_test, smoothed_predictions)

            model_tracking[model_name]["y_true"].extend(y_test)
            model_tracking[model_name]["raw_predictions"].extend(raw_predictions)
            model_tracking[model_name]["smoothed_predictions"].extend(smoothed_predictions)

            raw_fold_row = {
                "feature_mode": feature_mode,
                "neighbor_k": neighbor_k,
                "model": model_name,
                "prediction_type": "raw",
                "fold": fold_index,
                "test_sample": test_sample_id,
                "train_sample_count": len(train_sample_ids),
                "test_spot_count": len(y_test),
                "selected_gene_count": len(selected_genes),
                "feature_count_after_neighborhood": X_train.shape[1],
            }

            raw_fold_row.update(raw_metrics)

            smoothed_fold_row = {
                "feature_mode": feature_mode,
                "neighbor_k": neighbor_k,
                "model": model_name,
                "prediction_type": "smoothed",
                "fold": fold_index,
                "test_sample": test_sample_id,
                "train_sample_count": len(train_sample_ids),
                "test_spot_count": len(y_test),
                "selected_gene_count": len(selected_genes),
                "feature_count_after_neighborhood": X_train.shape[1],
            }

            smoothed_fold_row.update(smoothed_metrics)

            model_tracking[model_name]["fold_rows"].append(raw_fold_row)
            model_tracking[model_name]["fold_rows"].append(smoothed_fold_row)

            del model
            gc.collect()

        del X_train, y_train, X_test, y_test
        gc.collect()

    if feature_audit_tables:
        feature_audit_df = pd.concat(feature_audit_tables, ignore_index=True)
        feature_audit_df.to_csv(
            ML_OUTPUT_FOLDER / f"neighborhood_feature_availability_audit_{feature_mode}_k{neighbor_k}.csv",
            index=False,
        )

    feature_frequency_source = pd.DataFrame(feature_frequency_rows)

    feature_frequency_source.to_csv(
        ML_OUTPUT_FOLDER / f"neighborhood_selected_features_all_folds_{feature_mode}_k{neighbor_k}.csv",
        index=False,
    )

    feature_frequency = (
        feature_frequency_source.groupby("gene")
        .agg(
            selected_count=("gene", "size"),
            selected_fold_count=("fold", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["selected_count", "selected_fold_count", "gene"],
            ascending=[False, False, True],
        )
    )

    feature_frequency.to_csv(
        ML_OUTPUT_FOLDER / f"neighborhood_feature_selection_frequency_{feature_mode}_k{neighbor_k}.csv",
        index=False,
    )

    summary_rows = []

    for model_name, tracking in model_tracking.items():
        fold_df = pd.DataFrame(tracking["fold_rows"])

        fold_df.to_csv(
            ML_OUTPUT_FOLDER / f"neighborhood_ml_fold_results_{feature_mode}_k{neighbor_k}_{model_name}.csv",
            index=False,
        )

        y_true_all = np.array(tracking["y_true"])
        raw_predictions_all = np.array(tracking["raw_predictions"])
        smoothed_predictions_all = np.array(tracking["smoothed_predictions"])

        prediction_sets = {
            "raw": raw_predictions_all,
            "smoothed": smoothed_predictions_all,
        }

        for prediction_type, predictions in prediction_sets.items():
            metrics = calculate_metrics(y_true_all, predictions)

            summary_row = {
                "feature_mode": feature_mode,
                "neighbor_k": neighbor_k,
                "model": model_name,
                "prediction_type": prediction_type,
            }

            summary_row.update(metrics)
            summary_rows.append(summary_row)

            report_dict = classification_report(
                y_true_all,
                predictions,
                labels=all_labels,
                output_dict=True,
                zero_division=0,
            )

            report_df = pd.DataFrame(report_dict).transpose()

            report_df.to_csv(
                ML_OUTPUT_FOLDER / f"neighborhood_classification_report_{feature_mode}_k{neighbor_k}_{model_name}_{prediction_type}.csv",
                index=True,
            )

            cm = confusion_matrix(
                y_true_all,
                predictions,
                labels=all_labels,
            )

            cm_df = pd.DataFrame(
                cm,
                index=all_labels,
                columns=all_labels,
            )

            cm_df.to_csv(
                ML_OUTPUT_FOLDER / f"confusion_matrix_{feature_mode}_k{neighbor_k}_{model_name}_{prediction_type}.csv",
                index=True,
            )

            plot_confusion_matrix(
                matrix=cm,
                labels=all_labels,
                feature_mode=feature_mode,
                neighbor_k=neighbor_k,
                model_name=model_name,
                prediction_type=prediction_type,
            )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        ML_OUTPUT_FOLDER / f"neighborhood_model_summary_{feature_mode}_k{neighbor_k}.csv",
        index=False,
    )

    return summary_df


def save_final_summary(all_summary_tables: list[pd.DataFrame]) -> pd.DataFrame:
    combined_summary = pd.concat(all_summary_tables, ignore_index=True)

    combined_summary.to_csv(
        ML_OUTPUT_FOLDER / "neighborhood_7class_combined_model_comparison_summary.csv",
        index=False,
    )

    return combined_summary


def check_outputs() -> None:
    expected_files = [
        "neighborhood_ml_dataset_summary_all12.csv",
        "neighborhood_7class_combined_model_comparison_summary.csv",
    ]

    feature_modes = [
        "hvg",
        "stable_genes",
    ]

    model_names = [
        "logistic_regression",
        "random_forest",
        "linear_svm",
    ]

    prediction_types = [
        "raw",
        "smoothed",
    ]

    for feature_mode in feature_modes:
        for neighbor_k in NEIGHBOR_K_VALUES:
            expected_files.extend(
                [
                    f"neighborhood_model_summary_{feature_mode}_k{neighbor_k}.csv",
                    f"neighborhood_selected_features_all_folds_{feature_mode}_k{neighbor_k}.csv",
                    f"neighborhood_feature_selection_frequency_{feature_mode}_k{neighbor_k}.csv",
                    f"neighborhood_feature_availability_audit_{feature_mode}_k{neighbor_k}.csv",
                ]
            )

            for model_name in model_names:
                expected_files.append(
                    f"neighborhood_ml_fold_results_{feature_mode}_k{neighbor_k}_{model_name}.csv"
                )

                for prediction_type in prediction_types:
                    expected_files.extend(
                        [
                            f"neighborhood_classification_report_{feature_mode}_k{neighbor_k}_{model_name}_{prediction_type}.csv",
                            f"confusion_matrix_{feature_mode}_k{neighbor_k}_{model_name}_{prediction_type}.csv",
                        ]
                    )

    expected_figures = []

    for feature_mode in feature_modes:
        for neighbor_k in NEIGHBOR_K_VALUES:
            for model_name in model_names:
                for prediction_type in prediction_types:
                    expected_figures.append(
                        f"confusion_matrix_{feature_mode}_k{neighbor_k}_{model_name}_{prediction_type}.png"
                    )

    missing_files = []

    for file_name in expected_files:
        file_path = ML_OUTPUT_FOLDER / file_name

        if not file_path.exists():
            missing_files.append(file_path)

    for file_name in expected_figures:
        file_path = ML_FIGURE_FOLDER / file_name

        if not file_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print("Missing expected neighborhood ML output files:")

        for file_path in missing_files:
            print("-", file_path)

        raise FileNotFoundError("Some expected neighborhood ML output files are missing.")

    print("No missing files. All expected neighborhood-aware 7-class ML outputs were generated successfully.")


def main() -> None:
    sample_adatas, _ = load_all_samples()

    all_summary_tables = []

    for feature_mode in ["hvg", "stable_genes"]:
        for neighbor_k in NEIGHBOR_K_VALUES:
            summary_df = evaluate_feature_mode_and_neighbor_k(
                sample_adatas=sample_adatas,
                feature_mode=feature_mode,
                neighbor_k=neighbor_k,
            )

            all_summary_tables.append(summary_df)

    save_final_summary(all_summary_tables)

    check_outputs()

    print("\n7-class neighborhood-aware Visium ML classification completed.")
    print("Model outputs saved in:", ML_OUTPUT_FOLDER)


if __name__ == "__main__":
    main()