from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SAMPLE_IDS = [
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
]

OUTPUT_ROOT = Path("outputs")
COMPARISON_FOLDER = OUTPUT_ROOT / "all12_visium_comparison"
FIGURE_FOLDER = COMPARISON_FOLDER / "figures"

STABLE_FRACTION_THRESHOLD = 0.50
CORE_FRACTION_THRESHOLD = 0.75
MAX_HEATMAP_GENES_PER_LAYER = 10

COMPARISON_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)


def check_required_files() -> None:
    missing_files = []

    for sample_id in SAMPLE_IDS:
        file_path = (
            OUTPUT_ROOT
            / sample_id
            / f"top20_signature_genes_clean_{sample_id}.csv"
        )

        if not file_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print("-", file_path)

        raise FileNotFoundError("Some top20 signature gene files are missing.")

    print("All top20 clean signature gene files found.")


def load_top20_signature_tables() -> pd.DataFrame:
    all_tables = []

    required_columns = {
        "group",
        "names",
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
    }

    for sample_id in SAMPLE_IDS:
        file_path = (
            OUTPUT_ROOT
            / sample_id
            / f"top20_signature_genes_clean_{sample_id}.csv"
        )

        df = pd.read_csv(file_path)

        missing_columns = required_columns.difference(df.columns)

        if missing_columns:
            raise ValueError(
                f"{file_path} is missing required columns: {sorted(missing_columns)}"
            )

        df["sample_id"] = sample_id
        all_tables.append(df)

    combined_top20_df = pd.concat(all_tables, ignore_index=True)

    combined_top20_df.to_csv(
        COMPARISON_FOLDER / "combined_top20_signature_genes_clean_all12.csv",
        index=False,
    )

    print("Saved: combined_top20_signature_genes_clean_all12.csv")
    print("Combined rows:", combined_top20_df.shape[0])

    return combined_top20_df


def save_layer_sample_coverage(combined_top20_df: pd.DataFrame) -> pd.DataFrame:
    layer_sample_coverage = (
        combined_top20_df
        .groupby("group")["sample_id"]
        .nunique()
        .reset_index()
        .rename(columns={"sample_id": "sample_count_with_layer"})
        .sort_values("group")
        .reset_index(drop=True)
    )

    layer_sample_coverage.to_csv(
        COMPARISON_FOLDER / "layer_sample_coverage_all12.csv",
        index=False,
    )

    print("Saved: layer_sample_coverage_all12.csv")

    return layer_sample_coverage


def calculate_gene_recurrence(
    combined_top20_df: pd.DataFrame,
    layer_sample_coverage: pd.DataFrame,
) -> pd.DataFrame:
    recurrence_df = (
        combined_top20_df
        .groupby(["group", "names"])
        .agg(
            sample_count=("sample_id", "nunique"),
            sample_ids=("sample_id", lambda values: ", ".join(sorted(set(values)))),
            mean_score=("scores", "mean"),
            mean_logfoldchange=("logfoldchanges", "mean"),
            best_pvals_adj=("pvals_adj", "min"),
        )
        .reset_index()
    )

    recurrence_df = recurrence_df.merge(
        layer_sample_coverage,
        on="group",
        how="left",
    )

    recurrence_df["recurrence_fraction"] = (
        recurrence_df["sample_count"]
        / recurrence_df["sample_count_with_layer"]
    )

    def classify_recurrence(row: pd.Series) -> str:
        if row["recurrence_fraction"] >= CORE_FRACTION_THRESHOLD:
            return "core_recurrent"
        if row["recurrence_fraction"] >= STABLE_FRACTION_THRESHOLD:
            return "stable_recurrent"
        return "sample_specific_or_low_recurrence"

    recurrence_df["recurrence_category"] = recurrence_df.apply(
        classify_recurrence,
        axis=1,
    )

    recurrence_df = recurrence_df.sort_values(
        [
            "group",
            "recurrence_fraction",
            "sample_count",
            "mean_logfoldchange",
        ],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    recurrence_df.to_csv(
        COMPARISON_FOLDER / "all12_signature_gene_recurrence_by_layer.csv",
        index=False,
    )

    print("Saved: all12_signature_gene_recurrence_by_layer.csv")

    return recurrence_df


def save_stable_signature_genes(recurrence_df: pd.DataFrame) -> pd.DataFrame:
    stable_df = recurrence_df[
        recurrence_df["recurrence_fraction"] >= STABLE_FRACTION_THRESHOLD
    ].copy()

    stable_df = stable_df.sort_values(
        [
            "group",
            "recurrence_fraction",
            "sample_count",
            "mean_logfoldchange",
        ],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    stable_df.to_csv(
        COMPARISON_FOLDER / "stable_signature_genes_by_layer.csv",
        index=False,
    )

    print("Saved: stable_signature_genes_by_layer.csv")
    print("Stable recurrent genes:", stable_df.shape[0])

    return stable_df


def plot_stable_gene_count_by_layer(stable_df: pd.DataFrame) -> None:
    output_path = FIGURE_FOLDER / "stable_gene_count_by_layer.png"

    count_df = (
        stable_df
        .groupby("group")
        .size()
        .reset_index(name="stable_gene_count")
        .sort_values("group")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(count_df["group"], count_df["stable_gene_count"])
    plt.title("Stable Signature Gene Count by Brain Layer")
    plt.xlabel("Brain Layer")
    plt.ylabel("Number of Stable Recurrent Genes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def plot_signature_recurrence_heatmap(recurrence_df: pd.DataFrame) -> None:
    output_path = FIGURE_FOLDER / "all12_signature_gene_recurrence_heatmap.png"

    top_rows = []

    for group_name, group_df in recurrence_df.groupby("group"):
        group_top = group_df.sort_values(
            ["recurrence_fraction", "sample_count", "mean_logfoldchange"],
            ascending=[False, False, False],
        ).head(MAX_HEATMAP_GENES_PER_LAYER)

        top_rows.append(group_top)

    heatmap_source = pd.concat(top_rows, ignore_index=True)

    heatmap_matrix = (
        heatmap_source
        .pivot_table(
            index="names",
            columns="group",
            values="recurrence_fraction",
            aggfunc="max",
            fill_value=0,
        )
    )

    ordered_columns = sorted(heatmap_matrix.columns)
    heatmap_matrix = heatmap_matrix[ordered_columns]

    plt.figure(figsize=(10, max(6, heatmap_matrix.shape[0] * 0.25)))
    plt.imshow(heatmap_matrix.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Recurrence Fraction")
    plt.xticks(
        range(len(heatmap_matrix.columns)),
        heatmap_matrix.columns,
        rotation=45,
        ha="right",
    )
    plt.yticks(range(len(heatmap_matrix.index)), heatmap_matrix.index)
    plt.title("Top Recurrent Signature Genes Across 12 Visium Samples")
    plt.xlabel("Brain Layer")
    plt.ylabel("Gene")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def save_core_signature_genes(recurrence_df: pd.DataFrame) -> pd.DataFrame:
    core_df = recurrence_df[
        recurrence_df["recurrence_fraction"] >= CORE_FRACTION_THRESHOLD
    ].copy()

    core_df.to_csv(
        COMPARISON_FOLDER / "core_signature_genes_by_layer.csv",
        index=False,
    )

    print("Saved: core_signature_genes_by_layer.csv")
    print("Core recurrent genes:", core_df.shape[0])

    return core_df


def check_outputs() -> None:
    expected_files = [
        "combined_top20_signature_genes_clean_all12.csv",
        "layer_sample_coverage_all12.csv",
        "all12_signature_gene_recurrence_by_layer.csv",
        "stable_signature_genes_by_layer.csv",
        "core_signature_genes_by_layer.csv",
    ]

    expected_figures = [
        "all12_signature_gene_recurrence_heatmap.png",
    ]

    missing_files = []

    for file_name in expected_files:
        file_path = COMPARISON_FOLDER / file_name

        if not file_path.exists():
            missing_files.append(file_path)

    for file_name in expected_figures:
        file_path = FIGURE_FOLDER / file_name

        if not file_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print("Missing expected output files:")
        for file_path in missing_files:
            print("-", file_path)

        raise FileNotFoundError("Some expected recurrence output files are missing.")

    print("No missing files. All expected recurrence outputs were generated successfully.")


def main() -> None:
    check_required_files()

    combined_top20_df = load_top20_signature_tables()

    layer_sample_coverage = save_layer_sample_coverage(combined_top20_df)

    recurrence_df = calculate_gene_recurrence(
        combined_top20_df=combined_top20_df,
        layer_sample_coverage=layer_sample_coverage,
    )

    save_stable_signature_genes(recurrence_df)
    save_core_signature_genes(recurrence_df)

    plot_signature_recurrence_heatmap(recurrence_df)

    check_outputs()

    print("\nStable recurrent signature gene analysis completed.")
    print("Outputs saved in:", COMPARISON_FOLDER)
    print("Figures saved in:", FIGURE_FOLDER)


if __name__ == "__main__":
    main()