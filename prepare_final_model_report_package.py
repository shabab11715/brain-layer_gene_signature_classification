from pathlib import Path
import json
import math

import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("outputs") / "reports" / "final_model_report_package_strict_detailed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_DIR = Path("outputs") / "models" / "final_7class_model"
INTERNAL_NEIGHBOR_DIR = Path("outputs") / "models" / "visium_7class_neighborhood_classification"
STABLE_TOP20_DIR = Path("outputs") / "models" / "visium_stable_gene_classification_top20"
EXTERNAL_DIR = Path("outputs") / "models" / "external_spatialDLPFC_validation"

LABEL_ORDER = [
    "Layer1",
    "Layer2",
    "Layer3",
    "Layer4",
    "Layer5",
    "Layer6",
    "WM",
]

EXPECTED_EXTERNAL_SAMPLES = [
    "Br6522_ant",
    "Br6522_mid",
    "Br8667_post",
]


def require_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return path


def read_csv(path):
    return pd.read_csv(require_file(path))


def read_csv_indexed(path):
    return pd.read_csv(require_file(path), index_col=0)


def read_json(path):
    with open(require_file(path), "r", encoding="utf-8") as file:
        return json.load(file)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def format_number(value, digits=4):
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def difference(new_value, old_value):
    return safe_float(new_value) - safe_float(old_value)


def select_row(df, filters, fallback_sort_column=None):
    selected = df.copy()

    for column, value in filters.items():
        if column not in selected.columns:
            raise ValueError(f"Column '{column}' not found while selecting row with filters {filters}")

        selected = selected[selected[column].astype(str) == str(value)]

    if len(selected) == 0:
        if fallback_sort_column and fallback_sort_column in df.columns:
            return df.sort_values(fallback_sort_column, ascending=False).iloc[0]

        raise ValueError(f"No row matched filters: {filters}")

    return selected.iloc[0]


def normalize_classification_report(df):
    df = df.copy()

    if "label" not in df.columns:
        first_column = df.columns[0]
        df = df.rename(columns={first_column: "label"})

    df["label"] = df["label"].astype(str)
    df = df[df["label"].isin(LABEL_ORDER)].copy()

    if "f1-score" in df.columns:
        df["f1-score"] = df["f1-score"].astype(float)

    if "precision" in df.columns:
        df["precision"] = df["precision"].astype(float)

    if "recall" in df.columns:
        df["recall"] = df["recall"].astype(float)

    if "support" in df.columns:
        df["support"] = df["support"].astype(float)

    df["label"] = pd.Categorical(df["label"], categories=LABEL_ORDER, ordered=True)
    df = df.sort_values("label").reset_index(drop=True)
    df["label"] = df["label"].astype(str)

    return df


def read_classification_report(path):
    return normalize_classification_report(read_csv(path))


def read_confusion_matrix(path):
    matrix = read_csv_indexed(path)

    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)

    matrix = matrix.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    matrix = matrix.fillna(0).astype(int)

    return matrix


def load_inputs():
    metadata = read_json(FINAL_MODEL_DIR / "final_model_metadata.json")

    selected_genes = read_csv(FINAL_MODEL_DIR / "final_selected_stable_genes.csv")

    final_training_audit = read_csv(FINAL_MODEL_DIR / "final_internal_training_feature_audit.csv")

    final_training_label_counts = read_csv(FINAL_MODEL_DIR / "final_internal_training_label_counts.csv")

    internal_summary = read_csv(
        INTERNAL_NEIGHBOR_DIR / "neighborhood_7class_combined_model_comparison_summary.csv"
    )

    internal_raw = select_row(
        internal_summary,
        {
            "feature_mode": "stable_genes",
            "neighbor_k": 10,
            "model": "linear_svm",
            "prediction_type": "raw",
        },
    )

    internal_smoothed = select_row(
        internal_summary,
        {
            "feature_mode": "stable_genes",
            "neighbor_k": 10,
            "model": "linear_svm",
            "prediction_type": "smoothed",
        },
    )

    stable_top20_summary = read_csv(
        STABLE_TOP20_DIR / "ml_model_comparison_summary.csv"
    )

    stable_top20_baseline = select_row(
        stable_top20_summary,
        {
            "model": "linear_svm",
        },
        fallback_sort_column="weighted_f1",
    )

    external_summary = read_csv(
        EXTERNAL_DIR / "external_spatialDLPFC_validation_summary.csv"
    )

    external_raw = select_row(
        external_summary,
        {
            "sample_id": "ALL_EXTERNAL_LABELLED",
            "prediction_type": "raw",
        },
    )

    external_smoothed = select_row(
        external_summary,
        {
            "sample_id": "ALL_EXTERNAL_LABELLED",
            "prediction_type": "smoothed",
        },
    )

    internal_raw_classwise = read_classification_report(
        INTERNAL_NEIGHBOR_DIR / "neighborhood_classification_report_stable_genes_k10_linear_svm_raw.csv"
    )

    internal_smoothed_classwise = read_classification_report(
        INTERNAL_NEIGHBOR_DIR / "neighborhood_classification_report_stable_genes_k10_linear_svm_smoothed.csv"
    )

    external_classwise_all = read_csv(
        EXTERNAL_DIR / "external_spatialDLPFC_classwise_reports_combined.csv"
    )

    external_classwise_all["sample_id"] = external_classwise_all["sample_id"].astype(str)
    external_classwise_all["prediction_type"] = external_classwise_all["prediction_type"].astype(str)
    external_classwise_all["label"] = external_classwise_all["label"].astype(str)

    external_raw_classwise = normalize_classification_report(
        external_classwise_all[
            (external_classwise_all["sample_id"] == "ALL_EXTERNAL_LABELLED")
            & (external_classwise_all["prediction_type"] == "raw")
            & (external_classwise_all["label"].isin(LABEL_ORDER))
        ].copy()
    )

    external_smoothed_classwise = normalize_classification_report(
        external_classwise_all[
            (external_classwise_all["sample_id"] == "ALL_EXTERNAL_LABELLED")
            & (external_classwise_all["prediction_type"] == "smoothed")
            & (external_classwise_all["label"].isin(LABEL_ORDER))
        ].copy()
    )

    external_sample_summary = external_summary[
        external_summary["sample_id"].astype(str).isin(EXPECTED_EXTERNAL_SAMPLES)
    ].copy()

    external_feature_audit = read_csv(
        EXTERNAL_DIR / "external_feature_availability_audit.csv"
    )

    internal_confusion_raw = read_confusion_matrix(
        INTERNAL_NEIGHBOR_DIR / "confusion_matrix_stable_genes_k10_linear_svm_raw.csv"
    )

    internal_confusion_smoothed = read_confusion_matrix(
        INTERNAL_NEIGHBOR_DIR / "confusion_matrix_stable_genes_k10_linear_svm_smoothed.csv"
    )

    external_confusion_raw = read_confusion_matrix(
        EXTERNAL_DIR / "confusion_matrix_external_ALL_EXTERNAL_LABELLED_raw.csv"
    )

    external_confusion_smoothed = read_confusion_matrix(
        EXTERNAL_DIR / "confusion_matrix_external_ALL_EXTERNAL_LABELLED_smoothed.csv"
    )

    return {
        "metadata": metadata,
        "selected_genes": selected_genes,
        "final_training_audit": final_training_audit,
        "final_training_label_counts": final_training_label_counts,
        "stable_top20_baseline": stable_top20_baseline,
        "internal_raw": internal_raw,
        "internal_smoothed": internal_smoothed,
        "external_raw": external_raw,
        "external_smoothed": external_smoothed,
        "internal_raw_classwise": internal_raw_classwise,
        "internal_smoothed_classwise": internal_smoothed_classwise,
        "external_raw_classwise": external_raw_classwise,
        "external_smoothed_classwise": external_smoothed_classwise,
        "external_classwise_all": external_classwise_all,
        "external_sample_summary": external_sample_summary,
        "external_feature_audit": external_feature_audit,
        "internal_confusion_raw": internal_confusion_raw,
        "internal_confusion_smoothed": internal_confusion_smoothed,
        "external_confusion_raw": external_confusion_raw,
        "external_confusion_smoothed": external_confusion_smoothed,
    }


def get_major_confusions(confusion_matrix, top_n=2):
    rows = []

    for true_label in LABEL_ORDER:
        true_total = int(confusion_matrix.loc[true_label].sum())
        correct = int(confusion_matrix.loc[true_label, true_label])

        off_diagonal = confusion_matrix.loc[true_label].copy()
        off_diagonal[true_label] = 0
        off_diagonal = off_diagonal.sort_values(ascending=False)

        confusions = []

        for predicted_label, count in off_diagonal.head(top_n).items():
            count = int(count)

            if count <= 0:
                continue

            rate = count / true_total if true_total else 0
            confusions.append(f"{predicted_label}: {count} ({rate:.1%})")

        rows.append(
            {
                "True layer": true_label,
                "True support": true_total,
                "Correct predictions": correct,
                "Correct rate": correct / true_total if true_total else float("nan"),
                "Main wrong predictions": "; ".join(confusions) if confusions else "None",
            }
        )

    return pd.DataFrame(rows)


def build_external_label_support_table(external_classwise_all):
    smoothed = external_classwise_all[
        (external_classwise_all["prediction_type"] == "smoothed")
        & (external_classwise_all["sample_id"].isin(EXPECTED_EXTERNAL_SAMPLES))
        & (external_classwise_all["label"].isin(LABEL_ORDER))
    ].copy()

    table = smoothed.pivot_table(
        index="sample_id",
        columns="label",
        values="support",
        aggfunc="first",
        fill_value=0,
    )

    table = table.reindex(index=EXPECTED_EXTERNAL_SAMPLES, columns=LABEL_ORDER, fill_value=0)
    table = table.reset_index()
    table.columns.name = None

    return table


def build_summary_tables(data):
    metadata = data["metadata"]
    baseline = data["stable_top20_baseline"]
    internal_raw = data["internal_raw"]
    internal_smoothed = data["internal_smoothed"]
    external_raw = data["external_raw"]
    external_smoothed = data["external_smoothed"]
    selected_genes = data["selected_genes"]
    external_feature_audit = data["external_feature_audit"]
    final_training_audit = data["final_training_audit"]

    final_model_summary = pd.DataFrame(
        [
            ["Final classifier", metadata.get("model_name", "linear_svm")],
            ["Feature mode", metadata.get("feature_mode", "stable_genes_top20")],
            ["Gene selection setting", f"Top {metadata.get('top_n_genes_per_region', 20)} genes per region"],
            ["Selected recurrent genes", metadata.get("selected_gene_count", len(selected_genes))],
            ["Spatial neighborhood", f"k = {metadata.get('neighbor_k', 10)} nearest spots"],
            ["Training sections", "12 original labelled Visium DLPFC sections"],
            ["Training spots", metadata.get("training_spot_count", "")],
            ["Training features", metadata.get("training_feature_count", "")],
            ["Target labels", ", ".join(metadata.get("classes", LABEL_ORDER))],
            ["Final saved model", "outputs/models/final_7class_model/final_7class_linear_svm_model.joblib"],
        ],
        columns=["Item", "Value"],
    )

    model_stage_performance = pd.DataFrame(
        [
            [
                "A",
                "Expression-only stable top-20 baseline",
                "No",
                "No",
                baseline["accuracy"],
                baseline["weighted_f1"],
                baseline["macro_f1"],
                "No-neighborhood baseline.",
            ],
            [
                "B",
                "Stable top-20 + k=10 neighborhood features",
                "Yes",
                "No",
                internal_raw["accuracy"],
                internal_raw["weighted_f1"],
                internal_raw["macro_f1"],
                "Neighborhood-feature model before smoothing.",
            ],
            [
                "C",
                "Stable top-20 + k=10 neighborhood features + smoothing",
                "Yes",
                "Yes",
                internal_smoothed["accuracy"],
                internal_smoothed["weighted_f1"],
                internal_smoothed["macro_f1"],
                "Best internal setting.",
            ],
        ],
        columns=[
            "Stage",
            "Setup",
            "Uses neighbor gene features",
            "Uses prediction smoothing",
            "Accuracy",
            "Weighted F1",
            "Macro F1",
            "Interpretation",
        ],
    )

    neighborhood_effect_breakdown = pd.DataFrame(
        [
            [
                "No-neighborhood baseline to raw neighborhood model",
                difference(internal_raw["accuracy"], baseline["accuracy"]),
                difference(internal_raw["weighted_f1"], baseline["weighted_f1"]),
                difference(internal_raw["macro_f1"], baseline["macro_f1"]),
                "Effect of adding neighborhood gene-expression features.",
            ],
            [
                "Raw neighborhood model to smoothed neighborhood model",
                difference(internal_smoothed["accuracy"], internal_raw["accuracy"]),
                difference(internal_smoothed["weighted_f1"], internal_raw["weighted_f1"]),
                difference(internal_smoothed["macro_f1"], internal_raw["macro_f1"]),
                "Effect of post-prediction smoothing.",
            ],
            [
                "No-neighborhood baseline to smoothed neighborhood model",
                difference(internal_smoothed["accuracy"], baseline["accuracy"]),
                difference(internal_smoothed["weighted_f1"], baseline["weighted_f1"]),
                difference(internal_smoothed["macro_f1"], baseline["macro_f1"]),
                "Total improvement of the final internal setting.",
            ],
        ],
        columns=[
            "Comparison",
            "Accuracy change",
            "Weighted F1 change",
            "Macro F1 change",
            "Interpretation",
        ],
    )

    internal_external_overall = pd.DataFrame(
        [
            [
                "Internal leave-one-sample-out",
                "Raw",
                internal_raw["accuracy"],
                internal_raw["weighted_f1"],
                internal_raw["macro_f1"],
                "Internal result before smoothing.",
            ],
            [
                "Internal leave-one-sample-out",
                "Smoothed",
                internal_smoothed["accuracy"],
                internal_smoothed["weighted_f1"],
                internal_smoothed["macro_f1"],
                "Best internal result.",
            ],
            [
                "External spatialDLPFC labelled subset",
                "Raw",
                external_raw["accuracy"],
                external_raw["weighted_f1"],
                external_raw["macro_f1"],
                "External result before smoothing.",
            ],
            [
                "External spatialDLPFC labelled subset",
                "Smoothed",
                external_smoothed["accuracy"],
                external_smoothed["weighted_f1"],
                external_smoothed["macro_f1"],
                "Limited external validation result.",
            ],
        ],
        columns=[
            "Validation source",
            "Prediction type",
            "Accuracy",
            "Weighted F1",
            "Macro F1",
            "Interpretation",
        ],
    )

    internal_class = data["internal_smoothed_classwise"][
        ["label", "precision", "recall", "f1-score", "support"]
    ].copy()

    external_class = data["external_smoothed_classwise"][
        ["label", "precision", "recall", "f1-score", "support"]
    ].copy()

    internal_class = internal_class.rename(
        columns={
            "precision": "Internal precision",
            "recall": "Internal recall",
            "f1-score": "Internal F1",
            "support": "Internal support",
        }
    )

    external_class = external_class.rename(
        columns={
            "precision": "External precision",
            "recall": "External recall",
            "f1-score": "External F1",
            "support": "External support",
        }
    )

    classwise = internal_class.merge(
        external_class,
        on="label",
        how="outer",
    )

    classwise["F1 change"] = classwise["External F1"] - classwise["Internal F1"]

    classwise["Interpretation"] = classwise["label"].map(
        {
            "Layer1": "Moderate external transfer. External support is from Br6522_ant and Br6522_mid only.",
            "Layer2": "Weakest external transfer. Low support and confusion with Layer1/Layer3 are important limitations.",
            "Layer3": "Moderate external transfer with noticeable drop from internal performance.",
            "Layer4": "Weak external transfer, but the drop is smaller because internal performance was already limited.",
            "Layer5": "Strongest external transfer among cortical layers.",
            "Layer6": "Strong external transfer and relatively stable across internal/external testing.",
            "WM": "Good external transfer, but external support is from Br6522_ant and Br6522_mid only.",
        }
    )

    classwise["label"] = pd.Categorical(classwise["label"], categories=LABEL_ORDER, ordered=True)
    classwise = classwise.sort_values("label").reset_index(drop=True)
    classwise["label"] = classwise["label"].astype(str)

    external_sample_performance = data["external_sample_summary"].copy()
    external_sample_performance = external_sample_performance[
        [
            "sample_id",
            "prediction_type",
            "spot_count",
            "accuracy",
            "weighted_f1",
            "macro_f1",
        ]
    ].copy()

    external_sample_performance["Interpretation"] = external_sample_performance.apply(
        lambda row: "Br8667_post lacks Layer1 and WM labels, so its macro score is not directly comparable to samples with all seven labels."
        if str(row["sample_id"]) == "Br8667_post"
        else "This sample contains all seven target labels.",
        axis=1,
    )

    external_label_support = build_external_label_support_table(data["external_classwise_all"])

    missing_genes = sorted(
        set(
            gene.strip()
            for value in external_feature_audit["missing_gene_names"].dropna().astype(str)
            for gene in value.split(",")
            if gene.strip()
        )
    )

    external_feature_availability = pd.DataFrame(
        [
            ["Selected final genes", int(external_feature_audit["selected_genes"].max())],
            ["Minimum genes available in external samples", int(external_feature_audit["available_genes"].min())],
            ["Maximum missing genes in external samples", int(external_feature_audit["missing_genes"].max())],
            ["Missing gene names", ", ".join(missing_genes) if missing_genes else "None"],
            ["Reporting note", "84 of 85 selected genes were available externally."],
        ],
        columns=["Item", "Value"],
    )

    internal_feature_check = pd.DataFrame(
        [
            ["Selected genes", int(final_training_audit["selected_genes"].max())],
            ["Minimum available genes in internal training sections", int(final_training_audit["available_genes"].min())],
            ["Maximum missing genes in internal training sections", int(final_training_audit["missing_genes"].max())],
            ["Training feature count", metadata.get("training_feature_count", "")],
            ["Expected feature count", int(metadata.get("selected_gene_count", len(selected_genes))) * 2],
            ["Validation check", "Passed if training feature count equals selected genes × 2 and missing genes = 0."],
        ],
        columns=["Item", "Value"],
    )

    internal_confusion_summary = get_major_confusions(data["internal_confusion_smoothed"])
    external_confusion_summary = get_major_confusions(data["external_confusion_smoothed"])

    weak_layer_analysis = pd.DataFrame(
        [
            [
                "Layer2",
                classwise.loc[classwise["label"] == "Layer2", "Internal support"].iloc[0],
                classwise.loc[classwise["label"] == "Layer2", "External support"].iloc[0],
                classwise.loc[classwise["label"] == "Layer2", "Internal F1"].iloc[0],
                classwise.loc[classwise["label"] == "Layer2", "External F1"].iloc[0],
                "Layer2 has lower internal support than Layer3 and is externally split between Layer1, Layer2, and Layer3.",
            ],
            [
                "Layer4",
                classwise.loc[classwise["label"] == "Layer4", "Internal support"].iloc[0],
                classwise.loc[classwise["label"] == "Layer4", "External support"].iloc[0],
                classwise.loc[classwise["label"] == "Layer4", "Internal F1"].iloc[0],
                classwise.loc[classwise["label"] == "Layer4", "External F1"].iloc[0],
                "Layer4 errors are concentrated toward neighboring layers, especially Layer3 and Layer5.",
            ],
            [
                "Layer5",
                classwise.loc[classwise["label"] == "Layer5", "Internal support"].iloc[0],
                classwise.loc[classwise["label"] == "Layer5", "External support"].iloc[0],
                classwise.loc[classwise["label"] == "Layer5", "Internal F1"].iloc[0],
                classwise.loc[classwise["label"] == "Layer5", "External F1"].iloc[0],
                "Layer5 shows stronger transfer than Layer2 and Layer4.",
            ],
        ],
        columns=[
            "Layer",
            "Internal support",
            "External support",
            "Internal F1",
            "External F1",
            "Interpretation",
        ],
    )

    return {
        "final_model_summary": final_model_summary,
        "model_stage_performance": model_stage_performance,
        "neighborhood_effect_breakdown": neighborhood_effect_breakdown,
        "internal_external_overall": internal_external_overall,
        "classwise_internal_external": classwise,
        "external_sample_performance": external_sample_performance,
        "external_label_support": external_label_support,
        "external_feature_availability": external_feature_availability,
        "internal_feature_check": internal_feature_check,
        "internal_confusion_summary": internal_confusion_summary,
        "external_confusion_summary": external_confusion_summary,
        "weak_layer_analysis": weak_layer_analysis,
    }


def save_tables(tables):
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def make_model_stage_plot(tables):
    df = tables["model_stage_performance"].copy()
    plot_df = df.set_index("Stage")[["Accuracy", "Weighted F1", "Macro F1"]].astype(float)

    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Stage Performance")
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()

    path = OUTPUT_DIR / "model_stage_performance.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def make_internal_external_plot(tables):
    df = tables["internal_external_overall"].copy()
    df["Group"] = df["Validation source"] + " (" + df["Prediction type"] + ")"
    plot_df = df.set_index("Group")[["Accuracy", "Weighted F1", "Macro F1"]].astype(float)

    ax = plot_df.plot(kind="bar", figsize=(11, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Internal and External Performance")
    ax.legend(loc="lower right")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    path = OUTPUT_DIR / "internal_external_performance.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def make_classwise_plot(tables):
    df = tables["classwise_internal_external"].copy()
    plot_df = df.set_index("label")[["Internal F1", "External F1"]].astype(float)

    ax = plot_df.plot(kind="bar", figsize=(10, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title("Class-wise Internal vs External F1")
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()

    path = OUTPUT_DIR / "classwise_internal_external_f1.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def make_external_sample_plot(tables):
    df = tables["external_sample_performance"].copy()
    df = df[df["prediction_type"] == "smoothed"].copy()

    required_columns = [
        "sample_id",
        "accuracy",
        "weighted_f1",
        "macro_f1",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns for external sample plot: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    plot_df = df.set_index("sample_id")[
        [
            "accuracy",
            "weighted_f1",
            "macro_f1",
        ]
    ].astype(float)

    plot_df = plot_df.rename(
        columns={
            "accuracy": "Accuracy",
            "weighted_f1": "Weighted F1",
            "macro_f1": "Macro F1",
        }
    )

    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("External Performance by Labelled spatialDLPFC Sample")
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()

    path = OUTPUT_DIR / "external_sample_performance.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def main():
    data = load_inputs()
    tables = build_summary_tables(data)

    save_tables(tables)

    plots = {
        "model_stage": make_model_stage_plot(tables),
        "internal_external": make_internal_external_plot(tables),
        "classwise": make_classwise_plot(tables),
        "external_sample": make_external_sample_plot(tables),
    }

    print("Final model data report package created.")
    print("Folder:", OUTPUT_DIR)
    print("CSV tables saved:", len(tables))
    print("PNG figures saved:", len(plots))


if __name__ == "__main__":
    main()