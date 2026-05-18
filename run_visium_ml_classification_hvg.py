import gc
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DATA_FOLDER = Path("Data")

OUTPUT_ROOT = Path("outputs")
MODEL_OUTPUT_ROOT = OUTPUT_ROOT / "models"

ML_OUTPUT_FOLDER = MODEL_OUTPUT_ROOT / "visium_hvg_classification"
ML_FIGURE_FOLDER = ML_OUTPUT_FOLDER / "figures"

LABEL_COLUMN_SOURCE = "Region"
LABEL_COLUMN = "brain_region_label"

MIN_GENES = 200
MIN_COUNTS = 500
N_HVG = 2000

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

    summary = {
        "sample_id": sample_id,
        "spots_used": int(adata.n_obs),
        "genes_available": int(adata.n_vars),
        "labels_present": ", ".join(sorted(adata.obs[LABEL_COLUMN].astype(str).unique())),
    }

    print("Spots used:", adata.n_obs)
    print("Genes available:", adata.n_vars)
    print("Labels:", sorted(adata.obs[LABEL_COLUMN].astype(str).unique()))

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
        ML_OUTPUT_FOLDER / "ml_dataset_summary_all12.csv",
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
    test_sample: str,
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
            "fold": fold_index,
            "test_sample": test_sample,
            "gene": hvg_genes,
        }
    )

    feature_rows.to_csv(
        ML_OUTPUT_FOLDER / f"hvg_features_fold_{fold_index}_{test_sample}.csv",
        index=False,
    )

    del train_combined
    gc.collect()

    return hvg_genes


def extract_matrix(adata: sc.AnnData, genes: list[str]) -> tuple[np.ndarray | sparse.spmatrix, np.ndarray]:
    available_genes = [gene for gene in genes if gene in adata.var_names]

    if len(available_genes) != len(genes):
        missing_genes = sorted(set(genes).difference(set(available_genes)))
        raise ValueError(
            f"Some HVG genes are missing from a sample. Missing genes: {missing_genes[:20]}"
        )

    X = adata[:, genes].X

    if sparse.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)

    y = adata.obs[LABEL_COLUMN].astype(str).to_numpy()

    return X, y


def combine_fold_data(
    sample_adatas: dict[str, sc.AnnData],
    train_sample_ids: list[str],
    test_sample_id: str,
    hvg_genes: list[str],
) -> tuple[np.ndarray | sparse.spmatrix, np.ndarray, np.ndarray | sparse.spmatrix, np.ndarray]:
    X_train_parts = []
    y_train_parts = []

    for sample_id in train_sample_ids:
        X_sample, y_sample = extract_matrix(sample_adatas[sample_id], hvg_genes)
        X_train_parts.append(X_sample)
        y_train_parts.append(y_sample)

    X_test, y_test = extract_matrix(sample_adatas[test_sample_id], hvg_genes)

    if any(sparse.issparse(part) for part in X_train_parts):
        X_train = sparse.vstack(X_train_parts).tocsr()
    else:
        X_train = np.vstack(X_train_parts)

    y_train = np.concatenate(y_train_parts)

    return X_train, y_train, X_test, y_test


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], model_name: str) -> None:
    output_path = ML_FIGURE_FOLDER / f"confusion_matrix_{model_name}.png"

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, aspect="auto")

    plt.colorbar(label="Number of Spots")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]

            if value > 0:
                plt.text(
                    col_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def evaluate_models(sample_adatas: dict[str, sc.AnnData]) -> pd.DataFrame:
    models = get_models()
    sample_ids = sorted(sample_adatas.keys())

    all_labels = sorted(
        set(
            label
            for adata in sample_adatas.values()
            for label in adata.obs[LABEL_COLUMN].astype(str).unique()
        )
    )

    all_model_results = []
    all_hvg_feature_rows = []

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print("Training model:", model_name)
        print("=" * 80)

        y_true_all = []
        y_pred_all = []
        fold_rows = []

        for fold_index, test_sample_id in enumerate(sample_ids, start=1):
            train_sample_ids = [
                sample_id
                for sample_id in sample_ids
                if sample_id != test_sample_id
            ]

            print(f"Fold {fold_index}: testing on sample {test_sample_id}")

            hvg_genes = select_hvg_from_training_data(
                train_adatas=[sample_adatas[sample_id] for sample_id in train_sample_ids],
                fold_index=fold_index,
                test_sample=test_sample_id,
            )

            all_hvg_feature_rows.extend(
                [
                    {
                        "model": model_name,
                        "fold": fold_index,
                        "test_sample": test_sample_id,
                        "gene": gene,
                    }
                    for gene in hvg_genes
                ]
            )

            X_train, y_train, X_test, y_test = combine_fold_data(
                sample_adatas=sample_adatas,
                train_sample_ids=train_sample_ids,
                test_sample_id=test_sample_id,
                hvg_genes=hvg_genes,
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_accuracy = accuracy_score(y_test, y_pred)

            fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_index,
                    "test_sample": test_sample_id,
                    "train_sample_count": len(train_sample_ids),
                    "test_spot_count": len(y_test),
                    "hvg_gene_count": len(hvg_genes),
                    "accuracy": fold_accuracy,
                    "weighted_precision": fold_precision,
                    "weighted_recall": fold_recall,
                    "weighted_f1": fold_f1,
                }
            )

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

            del X_train, y_train, X_test, y_test
            gc.collect()

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        accuracy = accuracy_score(y_true_all, y_pred_all)

        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true_all,
            y_pred_all,
            average="weighted",
            zero_division=0,
        )

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true_all,
            y_pred_all,
            average="macro",
            zero_division=0,
        )

        all_model_results.append(
            {
                "model": model_name,
                "feature_mode": "hvg",
                "hvg_count_per_fold": N_HVG,
                "accuracy": accuracy,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "weighted_f1": weighted_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
            }
        )

        fold_df = pd.DataFrame(fold_rows)

        fold_df.to_csv(
            ML_OUTPUT_FOLDER / f"ml_fold_results_{model_name}.csv",
            index=False,
        )

        report_dict = classification_report(
            y_true_all,
            y_pred_all,
            labels=all_labels,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report_dict).transpose()

        report_df.to_csv(
            ML_OUTPUT_FOLDER / f"ml_classification_report_{model_name}.csv",
            index=True,
        )

        cm = confusion_matrix(
            y_true_all,
            y_pred_all,
            labels=all_labels,
        )

        cm_df = pd.DataFrame(
            cm,
            index=all_labels,
            columns=all_labels,
        )

        cm_df.to_csv(
            ML_OUTPUT_FOLDER / f"confusion_matrix_{model_name}.csv",
            index=True,
        )

        plot_confusion_matrix(
            matrix=cm,
            labels=all_labels,
            model_name=model_name,
        )

    results_df = pd.DataFrame(all_model_results).sort_values(
        "weighted_f1",
        ascending=False,
    )

    results_df.to_csv(
        ML_OUTPUT_FOLDER / "ml_model_comparison_summary.csv",
        index=False,
    )

    hvg_feature_df = pd.DataFrame(all_hvg_feature_rows)

    hvg_feature_df.to_csv(
        ML_OUTPUT_FOLDER / "hvg_features_all_models_all_folds.csv",
        index=False,
    )

    hvg_frequency_df = (
        hvg_feature_df
        .groupby("gene")
        .agg(
            selected_count=("fold", "count"),
            selected_model_count=("model", "nunique"),
            selected_fold_count=("fold", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["selected_count", "selected_model_count", "selected_fold_count"],
            ascending=False,
        )
    )

    hvg_frequency_df.to_csv(
        ML_OUTPUT_FOLDER / "hvg_feature_selection_frequency.csv",
        index=False,
    )

    return results_df


def check_outputs() -> None:
    expected_files = [
        "ml_dataset_summary_all12.csv",
        "ml_model_comparison_summary.csv",
        "hvg_features_all_models_all_folds.csv",
        "hvg_feature_selection_frequency.csv",
    ]

    model_names = [
        "logistic_regression",
        "random_forest",
        "linear_svm",
    ]

    for model_name in model_names:
        expected_files.extend(
            [
                f"ml_fold_results_{model_name}.csv",
                f"ml_classification_report_{model_name}.csv",
                f"confusion_matrix_{model_name}.csv",
            ]
        )

    expected_figures = [
        f"confusion_matrix_{model_name}.png"
        for model_name in model_names
    ]

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
        print("Missing expected HVG ML output files:")

        for file_path in missing_files:
            print("-", file_path)

        raise FileNotFoundError("Some expected HVG ML output files are missing.")

    print("No missing files. All expected HVG ML classification outputs were generated successfully.")


def main() -> None:
    sample_adatas, _ = load_all_samples()

    evaluate_models(sample_adatas)

    check_outputs()

    print("\nVisium HVG ML classification analysis completed.")
    print("Model outputs saved in:", ML_OUTPUT_FOLDER)


if __name__ == "__main__":
    main()