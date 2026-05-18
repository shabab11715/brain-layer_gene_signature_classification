import gc
import json
import math
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


INTERNAL_DATA_FOLDER = Path("Data")

OUTPUT_FOLDER = Path("outputs") / "models" / "final_7class_model"

LABEL_COLUMN_SOURCE = "Region"
LABEL_COLUMN = "brain_region_label"

MIN_GENES = 200
MIN_COUNTS = 500

TOP_N_GENES_PER_REGION = 20
STABLE_FRACTION_THRESHOLD = 0.50
NEIGHBOR_K = 10

MODEL_NAME = "linear_svm"
FEATURE_MODE = "stable_genes_top20"
PREDICTION_TARGET = "Layer1_Layer2_Layer3_Layer4_Layer5_Layer6_WM"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def clean_label(label):
    label = str(label).strip()
    label = label.replace(" ", "")

    if label == "WhiteMatter":
        return "WM"

    return label


def load_internal_sample(h5ad_path):
    sample_id = h5ad_path.stem

    print("\n" + "=" * 80)
    print("Loading internal sample:", sample_id)
    print("=" * 80)

    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()

    if LABEL_COLUMN_SOURCE not in adata.obs.columns:
        raise ValueError(
            f"{sample_id} is missing required label column '{LABEL_COLUMN_SOURCE}'."
        )

    if "spatial" not in adata.obsm:
        raise ValueError(
            f"{sample_id} is missing spatial coordinates in adata.obsm['spatial']."
        )

    adata.obs[LABEL_COLUMN] = adata.obs[LABEL_COLUMN_SOURCE].apply(clean_label)

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

    adata.obs["sample_id"] = sample_id

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("Spots used:", adata.n_obs)
    print("Genes:", adata.n_vars)
    print(adata.obs[LABEL_COLUMN].value_counts().sort_index())

    return sample_id, adata


def load_all_internal_samples():
    h5ad_files = sorted(INTERNAL_DATA_FOLDER.glob("*.h5ad"))

    if len(h5ad_files) == 0:
        raise FileNotFoundError(f"No internal .h5ad files found in {INTERNAL_DATA_FOLDER}")

    sample_adatas = {}

    for h5ad_path in h5ad_files:
        sample_id, adata = load_internal_sample(h5ad_path)
        sample_adatas[sample_id] = adata

    return sample_adatas


def discover_stable_genes_from_internal_data(sample_adatas):
    marker_rows = []
    layer_coverage_rows = []

    sample_ids = sorted(sample_adatas.keys())

    for sample_id in sample_ids:
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
        raise ValueError("No marker genes were found from internal samples.")

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
        OUTPUT_FOLDER / "final_internal_top20_markers_all12.csv",
        index=False,
    )

    recurrence.to_csv(
        OUTPUT_FOLDER / "final_internal_gene_recurrence_all12.csv",
        index=False,
    )

    stable_recurrence.to_csv(
        OUTPUT_FOLDER / "final_internal_stable_gene_features_all12.csv",
        index=False,
    )

    stable_genes = sorted(stable_recurrence["gene"].astype(str).unique())

    if len(stable_genes) == 0:
        raise ValueError("No stable recurrent genes were found from internal samples.")

    pd.DataFrame({"gene": stable_genes}).to_csv(
        OUTPUT_FOLDER / "final_selected_stable_genes.csv",
        index=False,
    )

    print("\nSelected stable genes:", len(stable_genes))

    return stable_genes


def extract_gene_matrix_sparse(adata, genes):
    available_pairs = [
        (index, gene)
        for index, gene in enumerate(genes)
        if gene in adata.var_names
    ]

    available_genes = [gene for _, gene in available_pairs]
    missing_genes = [gene for gene in genes if gene not in adata.var_names]

    if len(available_genes) == 0:
        raise ValueError("None of the selected genes were found in this sample.")

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


def get_neighbor_indices(adata, neighbor_k):
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


def build_neighborhood_feature_matrix(adata, genes, neighbor_k):
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

    return X_combined, y, neighbor_indices, available_genes, missing_genes


def combine_internal_training_data(sample_adatas, genes):
    X_parts = []
    y_parts = []
    audit_rows = []
    label_rows = []

    for sample_id, adata in sample_adatas.items():
        X_sample, y_sample, _, available_genes, missing_genes = build_neighborhood_feature_matrix(
            adata=adata,
            genes=genes,
            neighbor_k=NEIGHBOR_K,
        )

        X_parts.append(X_sample)
        y_parts.append(y_sample)

        audit_rows.append(
            {
                "sample_id": sample_id,
                "split": "internal_train",
                "selected_genes": len(genes),
                "available_genes": len(available_genes),
                "missing_genes": len(missing_genes),
                "feature_count_after_neighborhood": X_sample.shape[1],
                "missing_gene_names": ", ".join(missing_genes),
            }
        )

        counts = pd.Series(y_sample).value_counts().sort_index()

        for label, count in counts.items():
            label_rows.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "spot_count": int(count),
                }
            )

    X_train = sparse.vstack(X_parts).tocsr()
    y_train = np.concatenate(y_parts)

    audit = pd.DataFrame(audit_rows)
    label_counts = pd.DataFrame(label_rows)

    return X_train, y_train, audit, label_counts


def train_model(X_train, y_train):
    model = Pipeline(
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
    )

    model.fit(X_train, y_train)

    return model


def save_metadata(selected_genes, X_train, y_train):
    metadata = {
        "model_name": MODEL_NAME,
        "feature_mode": FEATURE_MODE,
        "prediction_target": PREDICTION_TARGET,
        "neighbor_k": NEIGHBOR_K,
        "top_n_genes_per_region": TOP_N_GENES_PER_REGION,
        "stable_fraction_threshold": STABLE_FRACTION_THRESHOLD,
        "min_genes": MIN_GENES,
        "min_counts": MIN_COUNTS,
        "selected_gene_count": len(selected_genes),
        "training_spot_count": int(X_train.shape[0]),
        "training_feature_count": int(X_train.shape[1]),
        "classes": sorted(pd.Series(y_train).unique().tolist()),
    }

    with open(OUTPUT_FOLDER / "final_model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    pd.DataFrame([metadata]).to_csv(
        OUTPUT_FOLDER / "final_model_metadata.csv",
        index=False,
    )


def main():
    internal_adatas = load_all_internal_samples()

    selected_genes = discover_stable_genes_from_internal_data(
        sample_adatas=internal_adatas,
    )

    X_train, y_train, training_audit, label_counts = combine_internal_training_data(
        sample_adatas=internal_adatas,
        genes=selected_genes,
    )

    training_audit.to_csv(
        OUTPUT_FOLDER / "final_internal_training_feature_audit.csv",
        index=False,
    )

    label_counts.to_csv(
        OUTPUT_FOLDER / "final_internal_training_label_counts.csv",
        index=False,
    )

    print("\nTraining final 7-class model...")
    print("Training matrix:", X_train.shape)
    print("Training labels:", pd.Series(y_train).value_counts().sort_index().to_dict())

    model = train_model(
        X_train=X_train,
        y_train=y_train,
    )

    joblib.dump(
        model,
        OUTPUT_FOLDER / "final_7class_linear_svm_model.joblib",
    )

    save_metadata(
        selected_genes=selected_genes,
        X_train=X_train,
        y_train=y_train,
    )

    print("\nFinal model training completed.")
    print("Saved model folder:", OUTPUT_FOLDER)

    del X_train, y_train
    gc.collect()


if __name__ == "__main__":
    main()