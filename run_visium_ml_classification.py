# Visium ML classification using leakage-safe stable recurrent signature genes.
# This version discovers stable genes inside each training fold only.
# The held-out test sample is never used during feature selection.

import gc
import math
from pathlib import Path

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

ML_OUTPUT_FOLDER = MODEL_OUTPUT_ROOT / "visium_stable_gene_classification_top20"
ML_FIGURE_FOLDER = ML_OUTPUT_FOLDER / "figures"

LABEL_COLUMN_SOURCE = "Region"
LABEL_COLUMN = "brain_region_label"

MIN_GENES = 200
MIN_COUNTS = 500
TOP_N_GENES_PER_REGION = 20
STABLE_FRACTION_THRESHOLD = 0.50

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


def discover_training_stable_genes(
    sample_adatas: dict[str, sc.AnnData],
    train_sample_ids: list[str],
    fold_index: int,
    test_sample_id: str,
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
        ML_OUTPUT_FOLDER / f"training_top_markers_fold_{fold_index}_{test_sample_id}.csv",
        index=False,
    )

    recurrence.to_csv(
        ML_OUTPUT_FOLDER / f"training_gene_recurrence_fold_{fold_index}_{test_sample_id}.csv",
        index=False,
    )

    stable_recurrence.to_csv(
        ML_OUTPUT_FOLDER / f"stable_gene_features_fold_{fold_index}_{test_sample_id}.csv",
        index=False,
    )

    stable_genes = sorted(stable_recurrence["gene"].astype(str).unique())

    if len(stable_genes) == 0:
        raise ValueError(
            f"No stable recurrent genes found in fold {fold_index} while testing {test_sample_id}. "
            "This means the threshold is too strict for this training split."
        )

    print(
        f"Fold {fold_index} stable genes discovered from training samples only:",
        len(stable_genes),
    )

    return stable_genes


def extract_gene_matrix(
    adata: sc.AnnData,
    genes: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    available_genes = [gene for gene in genes if gene in adata.var_names]
    missing_genes = [gene for gene in genes if gene not in adata.var_names]

    if len(available_genes) == 0:
        raise ValueError("None of the selected genes were found in this h5ad file.")

    extracted = adata[:, available_genes].X

    if sparse.issparse(extracted):
        extracted = extracted.toarray()
    else:
        extracted = np.asarray(extracted)

    full_matrix = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)

    gene_to_index = {
        gene: index
        for index, gene in enumerate(genes)
    }

    for local_index, gene in enumerate(available_genes):
        full_matrix[:, gene_to_index[gene]] = extracted[:, local_index]

    return full_matrix, available_genes, missing_genes


def combine_fold_data(
    sample_adatas: dict[str, sc.AnnData],
    train_sample_ids: list[str],
    test_sample_id: str,
    stable_genes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    X_train_parts = []
    y_train_parts = []
    feature_audit_rows = []

    for sample_id in train_sample_ids:
        X_sample, available_genes, missing_genes = extract_gene_matrix(
            sample_adatas[sample_id],
            stable_genes,
        )

        y_sample = sample_adatas[sample_id].obs[LABEL_COLUMN].astype(str).to_numpy()

        X_train_parts.append(X_sample)
        y_train_parts.append(y_sample)

        feature_audit_rows.append(
            {
                "sample_id": sample_id,
                "split": "train",
                "selected_genes": len(stable_genes),
                "available_genes": len(available_genes),
                "missing_genes": len(missing_genes),
                "missing_gene_names": ", ".join(missing_genes),
            }
        )

    X_test, available_test_genes, missing_test_genes = extract_gene_matrix(
        sample_adatas[test_sample_id],
        stable_genes,
    )

    y_test = sample_adatas[test_sample_id].obs[LABEL_COLUMN].astype(str).to_numpy()

    feature_audit_rows.append(
        {
            "sample_id": test_sample_id,
            "split": "test",
            "selected_genes": len(stable_genes),
            "available_genes": len(available_test_genes),
            "missing_genes": len(missing_test_genes),
            "missing_gene_names": ", ".join(missing_test_genes),
        }
    )

    X_train = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)

    feature_audit = pd.DataFrame(feature_audit_rows)

    return X_train, y_train, X_test, y_test, feature_audit


def get_models() -> dict:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
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
                ("scaler", StandardScaler()),
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


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    model_name: str,
) -> None:
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
            np.concatenate(
                [
                    sample_adatas[sample_id].obs[LABEL_COLUMN].astype(str).to_numpy()
                    for sample_id in sample_ids
                ]
            )
        )
    )

    results = []
    all_feature_audits = []
    all_feature_frequency_rows = []

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

            stable_genes = discover_training_stable_genes(
                sample_adatas=sample_adatas,
                train_sample_ids=train_sample_ids,
                fold_index=fold_index,
                test_sample_id=test_sample_id,
            )

            X_train, y_train, X_test, y_test, feature_audit = combine_fold_data(
                sample_adatas=sample_adatas,
                train_sample_ids=train_sample_ids,
                test_sample_id=test_sample_id,
                stable_genes=stable_genes,
            )

            feature_audit["model"] = model_name
            feature_audit["fold"] = fold_index
            feature_audit["test_sample"] = test_sample_id
            all_feature_audits.append(feature_audit)

            for gene in stable_genes:
                all_feature_frequency_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_index,
                        "test_sample": test_sample_id,
                        "gene": gene,
                    }
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
                    "stable_gene_count": len(stable_genes),
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

        results.append(
            {
                "model": model_name,
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

    results_df = pd.DataFrame(results).sort_values(
        "weighted_f1",
        ascending=False,
    )

    results_df.to_csv(
        ML_OUTPUT_FOLDER / "ml_model_comparison_summary.csv",
        index=False,
    )

    if all_feature_audits:
        feature_audit_df = pd.concat(all_feature_audits, ignore_index=True)
        feature_audit_df.to_csv(
            ML_OUTPUT_FOLDER / "ml_feature_availability_audit_all_models.csv",
            index=False,
        )

    if all_feature_frequency_rows:
        feature_frequency_source = pd.DataFrame(all_feature_frequency_rows)

        feature_frequency = (
            feature_frequency_source.groupby("gene")
            .agg(
                model_count=("model", "nunique"),
                fold_count=("fold", "nunique"),
                total_times_selected=("gene", "size"),
            )
            .reset_index()
            .sort_values(
                ["fold_count", "total_times_selected", "gene"],
                ascending=[False, False, True],
            )
        )

        feature_frequency_source.to_csv(
            ML_OUTPUT_FOLDER / "stable_gene_features_all_models_all_folds.csv",
            index=False,
        )

        feature_frequency.to_csv(
            ML_OUTPUT_FOLDER / "stable_gene_feature_selection_frequency.csv",
            index=False,
        )

    return results_df


def check_outputs() -> None:
    expected_files = [
        "ml_dataset_summary_all12.csv",
        "ml_model_comparison_summary.csv",
        "ml_feature_availability_audit_all_models.csv",
        "stable_gene_features_all_models_all_folds.csv",
        "stable_gene_feature_selection_frequency.csv",
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
        print("Missing expected ML output files:")

        for file_path in missing_files:
            print("-", file_path)

        raise FileNotFoundError("Some expected ML output files are missing.")

    print("No missing files. All expected ML classification outputs were generated successfully.")


def main() -> None:
    sample_adatas, _ = load_all_samples()

    evaluate_models(sample_adatas)

    check_outputs()

    print("\nLeakage-safe Visium stable-gene ML classification completed.")
    print("Model outputs saved in:", ML_OUTPUT_FOLDER)


if __name__ == "__main__":
    main()