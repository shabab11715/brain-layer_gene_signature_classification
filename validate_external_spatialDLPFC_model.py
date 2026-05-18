import gc
from pathlib import Path

import anndata as ad
import joblib
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmread
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")

import matplotlib.pyplot as plt


EXTERNAL_EXPORT_FOLDER = Path("Data") / "spatialDLPFC_labelled_export"
FINAL_MODEL_FOLDER = Path("outputs") / "models" / "final_7class_model"

OUTPUT_FOLDER = Path("outputs") / "models" / "external_spatialDLPFC_validation"
FIGURE_FOLDER = OUTPUT_FOLDER / "figures"

LABEL_COLUMN = "brain_region_label"

MIN_GENES = 200
MIN_COUNTS = 500
NEIGHBOR_K = 10

MODEL_PATH = FINAL_MODEL_FOLDER / "final_7class_linear_svm_model.joblib"
GENE_PATH = FINAL_MODEL_FOLDER / "final_selected_stable_genes.csv"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)


def make_unique_names(names):
    seen = {}
    unique_names = []

    for name in names:
        base_name = str(name)

        if base_name not in seen:
            seen[base_name] = 0
            unique_names.append(base_name)
        else:
            seen[base_name] += 1
            unique_names.append(f"{base_name}_{seen[base_name]}")

    return unique_names


def clean_label(label):
    label = str(label).strip()
    label = label.replace(" ", "")

    if label == "WhiteMatter":
        return "WM"

    return label


def load_final_model_and_genes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Final model not found: {MODEL_PATH}")

    if not GENE_PATH.exists():
        raise FileNotFoundError(f"Selected genes file not found: {GENE_PATH}")

    model = joblib.load(MODEL_PATH)
    genes = pd.read_csv(GENE_PATH)["gene"].astype(str).tolist()

    if len(genes) == 0:
        raise ValueError("Selected gene list is empty.")

    print("Loaded final model:", MODEL_PATH)
    print("Loaded selected genes:", len(genes))

    return model, genes


def load_external_sample(sample_id):
    sample_folder = EXTERNAL_EXPORT_FOLDER / sample_id

    expression_path = sample_folder / "expression_genes_by_spots.mtx"
    obs_path = sample_folder / "obs.csv"
    var_path = sample_folder / "var.csv"
    coords_path = sample_folder / "spatial_coords.csv"

    if not expression_path.exists():
        raise FileNotFoundError(expression_path)

    if not obs_path.exists():
        raise FileNotFoundError(obs_path)

    if not var_path.exists():
        raise FileNotFoundError(var_path)

    if not coords_path.exists():
        raise FileNotFoundError(coords_path)

    print("\n" + "=" * 80)
    print("Loading external sample:", sample_id)
    print("=" * 80)

    expression_genes_by_spots = mmread(expression_path).tocsr()
    expression_spots_by_genes = expression_genes_by_spots.T.tocsr()

    obs = pd.read_csv(obs_path, index_col=0)
    var = pd.read_csv(var_path)
    coords = pd.read_csv(coords_path)

    gene_symbols = var["gene_symbol"].astype(str).tolist()
    gene_symbols = make_unique_names(gene_symbols)

    adata = ad.AnnData(
        X=expression_spots_by_genes,
        obs=obs.copy(),
        var=pd.DataFrame(index=gene_symbols),
    )

    if "Region" not in adata.obs.columns:
        if "manual_layer_label" not in adata.obs.columns:
            raise ValueError(f"{sample_id} has no Region or manual_layer_label column.")

        adata.obs["Region"] = adata.obs["manual_layer_label"].apply(clean_label)

    adata.obs[LABEL_COLUMN] = adata.obs["Region"].apply(clean_label)
    adata.obs["sample_id"] = sample_id

    if "barcode" not in coords.columns:
        raise ValueError(f"{sample_id} spatial_coords.csv has no barcode column.")

    coords = coords.set_index("barcode")
    coords = coords.loc[adata.obs_names]

    if "pxl_col_in_fullres" in coords.columns and "pxl_row_in_fullres" in coords.columns:
        adata.obsm["spatial"] = coords[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    else:
        numeric_cols = coords.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            raise ValueError(f"{sample_id} does not have two numeric spatial coordinate columns.")

        adata.obsm["spatial"] = coords[numeric_cols[:2]].to_numpy()

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

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("External spots used:", adata.n_obs)
    print("External genes:", adata.n_vars)
    print(adata.obs[LABEL_COLUMN].value_counts().sort_index())

    return sample_id, adata


def load_external_samples():
    sample_file = EXTERNAL_EXPORT_FOLDER / "labelled_samples_found.csv"

    if not sample_file.exists():
        raise FileNotFoundError(sample_file)

    sample_table = pd.read_csv(sample_file)
    sample_ids = sorted(sample_table["sample_id"].astype(str).tolist())

    sample_adatas = {}

    for sample_id in sample_ids:
        loaded_sample_id, adata = load_external_sample(sample_id)
        sample_adatas[loaded_sample_id] = adata

    return sample_adatas


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


def smooth_predictions(predictions, neighbor_indices):
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


def calculate_metrics(y_true, y_pred):
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


def plot_confusion_matrix(matrix, labels, sample_name, prediction_type):
    output_path = FIGURE_FOLDER / f"confusion_matrix_external_{sample_name}_{prediction_type}.png"

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, aspect="auto")

    plt.colorbar(label="Number of Spots")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    plt.title(f"External spatialDLPFC Confusion Matrix: {sample_name} / {prediction_type}")
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


def evaluate_external_samples(model, external_adatas, selected_genes):
    all_labels = [
        "Layer1",
        "Layer2",
        "Layer3",
        "Layer4",
        "Layer5",
        "Layer6",
        "WM",
    ]

    summary_rows = []
    class_report_tables = []
    y_true_all = []
    y_raw_all = []
    y_smoothed_all = []
    feature_audit_rows = []

    for sample_id, adata in external_adatas.items():
        X_test, y_test, neighbor_indices, available_genes, missing_genes = build_neighborhood_feature_matrix(
            adata=adata,
            genes=selected_genes,
            neighbor_k=NEIGHBOR_K,
        )

        raw_predictions = model.predict(X_test)

        smoothed_predictions = smooth_predictions(
            predictions=raw_predictions,
            neighbor_indices=neighbor_indices,
        )

        prediction_sets = {
            "raw": raw_predictions,
            "smoothed": smoothed_predictions,
        }

        for prediction_type, predictions in prediction_sets.items():
            metrics = calculate_metrics(y_test, predictions)

            row = {
                "sample_id": sample_id,
                "prediction_type": prediction_type,
                "model": "linear_svm",
                "feature_mode": "stable_genes_top20",
                "neighbor_k": NEIGHBOR_K,
                "spot_count": len(y_test),
                "selected_genes": len(selected_genes),
                "available_genes": len(available_genes),
                "missing_genes": len(missing_genes),
            }

            row.update(metrics)
            summary_rows.append(row)

            report_dict = classification_report(
                y_test,
                predictions,
                labels=all_labels,
                output_dict=True,
                zero_division=0,
            )

            report_df = pd.DataFrame(report_dict).transpose()
            report_df.insert(0, "sample_id", sample_id)
            report_df.insert(1, "prediction_type", prediction_type)

            class_report_tables.append(report_df.reset_index(names="label"))

            report_df.to_csv(
                OUTPUT_FOLDER / f"external_classification_report_{sample_id}_{prediction_type}.csv",
                index=True,
            )

            cm = confusion_matrix(
                y_test,
                predictions,
                labels=all_labels,
            )

            cm_df = pd.DataFrame(
                cm,
                index=all_labels,
                columns=all_labels,
            )

            cm_df.to_csv(
                OUTPUT_FOLDER / f"confusion_matrix_external_{sample_id}_{prediction_type}.csv",
                index=True,
            )

            plot_confusion_matrix(
                matrix=cm,
                labels=all_labels,
                sample_name=sample_id,
                prediction_type=prediction_type,
            )

        y_true_all.extend(y_test)
        y_raw_all.extend(raw_predictions)
        y_smoothed_all.extend(smoothed_predictions)

        feature_audit_rows.append(
            {
                "sample_id": sample_id,
                "split": "external_test",
                "selected_genes": len(selected_genes),
                "available_genes": len(available_genes),
                "missing_genes": len(missing_genes),
                "feature_count_after_neighborhood": X_test.shape[1],
                "missing_gene_names": ", ".join(missing_genes),
            }
        )

    y_true_all = np.array(y_true_all)
    y_raw_all = np.array(y_raw_all)
    y_smoothed_all = np.array(y_smoothed_all)

    for prediction_type, predictions in {
        "raw": y_raw_all,
        "smoothed": y_smoothed_all,
    }.items():
        metrics = calculate_metrics(y_true_all, predictions)

        row = {
            "sample_id": "ALL_EXTERNAL_LABELLED",
            "prediction_type": prediction_type,
            "model": "linear_svm",
            "feature_mode": "stable_genes_top20",
            "neighbor_k": NEIGHBOR_K,
            "spot_count": len(y_true_all),
            "selected_genes": len(selected_genes),
            "available_genes": np.nan,
            "missing_genes": np.nan,
        }

        row.update(metrics)
        summary_rows.append(row)

        report_dict = classification_report(
            y_true_all,
            predictions,
            labels=all_labels,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report_dict).transpose()
        report_df.insert(0, "sample_id", "ALL_EXTERNAL_LABELLED")
        report_df.insert(1, "prediction_type", prediction_type)

        class_report_tables.append(report_df.reset_index(names="label"))

        report_df.to_csv(
            OUTPUT_FOLDER / f"external_classification_report_ALL_EXTERNAL_LABELLED_{prediction_type}.csv",
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
            OUTPUT_FOLDER / f"confusion_matrix_external_ALL_EXTERNAL_LABELLED_{prediction_type}.csv",
            index=True,
        )

        plot_confusion_matrix(
            matrix=cm,
            labels=all_labels,
            sample_name="ALL_EXTERNAL_LABELLED",
            prediction_type=prediction_type,
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        [
            "sample_id",
            "prediction_type",
        ]
    ).reset_index(drop=True)

    summary_df.to_csv(
        OUTPUT_FOLDER / "external_spatialDLPFC_validation_summary.csv",
        index=False,
    )

    class_report_df = pd.concat(class_report_tables, ignore_index=True)

    class_report_df.to_csv(
        OUTPUT_FOLDER / "external_spatialDLPFC_classwise_reports_combined.csv",
        index=False,
    )

    feature_audit_df = pd.DataFrame(feature_audit_rows)

    feature_audit_df.to_csv(
        OUTPUT_FOLDER / "external_feature_availability_audit.csv",
        index=False,
    )

    return summary_df


def save_text_summary(summary_df):
    combined_smoothed = summary_df[
        (summary_df["sample_id"] == "ALL_EXTERNAL_LABELLED")
        & (summary_df["prediction_type"] == "smoothed")
    ].iloc[0]

    summary_text = f"""
External spatialDLPFC Labelled Subset Validation

Final model:
Stable top-20 recurrent genes.
k=10 spatial neighborhood features.
Linear SVM.
Raw and smoothed predictions evaluated.

External validation data:
Three manually labelled spatialDLPFC samples:
- Br6522_ant
- Br6522_mid
- Br8667_post

Important limitation:
This is a limited external validation subset, not full external validation across all 30 spatialDLPFC samples.

Combined external smoothed result:
Accuracy: {combined_smoothed["accuracy"]:.4f}
Weighted F1: {combined_smoothed["weighted_f1"]:.4f}
Macro F1: {combined_smoothed["macro_f1"]:.4f}

Safe interpretation:
This result should be reported as limited external validation on the manually labelled spatialDLPFC subset.
"""

    with open(OUTPUT_FOLDER / "external_spatialDLPFC_validation_interpretation.txt", "w", encoding="utf-8") as file:
        file.write(summary_text)

    print(summary_text)


def main():
    model, selected_genes = load_final_model_and_genes()

    external_adatas = load_external_samples()

    summary_df = evaluate_external_samples(
        model=model,
        external_adatas=external_adatas,
        selected_genes=selected_genes,
    )

    save_text_summary(summary_df)

    print("\nExternal spatialDLPFC validation completed.")
    print("Outputs saved in:", OUTPUT_FOLDER)

    gc.collect()


if __name__ == "__main__":
    main()