from pathlib import Path

import matplotlib
import pandas as pd

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

COMPARISON_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)


def check_required_files() -> None:
    missing_files = []

    for sample_id in SAMPLE_IDS:
        sample_folder = OUTPUT_ROOT / sample_id

        required_files = [
            sample_folder / f"input_h5ad_summary_{sample_id}.csv",
            sample_folder / f"filter_threshold_summary_{sample_id}.csv",
            sample_folder / f"final_clustering_summary_{sample_id}.csv",
            sample_folder / f"leiden_tuning_ranked_results_{sample_id}.csv",
            sample_folder / f"top20_signature_genes_clean_{sample_id}.csv",
            sample_folder / f"signature_genes_all_{sample_id}.csv",
            sample_folder / f"final_spot_cluster_assignments_{sample_id}.csv",
            sample_folder / f"top_signature_gene_spatial_maps_{sample_id}.csv",
        ]

        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(file_path)

    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print("-", file_path)

        raise FileNotFoundError("Some required files are missing. Fix these before continuing.")

    print("All required files found for all 12 samples.")


def combine_qc_summaries() -> pd.DataFrame:
    qc_rows = []

    for sample_id in SAMPLE_IDS:
        sample_folder = OUTPUT_ROOT / sample_id

        input_summary_path = sample_folder / f"input_h5ad_summary_{sample_id}.csv"
        filter_summary_path = sample_folder / f"filter_threshold_summary_{sample_id}.csv"

        input_summary = pd.read_csv(input_summary_path)
        filter_summary = pd.read_csv(filter_summary_path)

        light_filter_row = filter_summary[
            (filter_summary["min_genes"] == 200)
            & (filter_summary["min_counts"] == 500)
        ]

        if light_filter_row.empty:
            raise ValueError(
                f"Could not find min_genes=200 and min_counts=500 row for sample {sample_id}."
            )

        input_row = input_summary.iloc[0]
        filter_row = light_filter_row.iloc[0]

        qc_rows.append(
            {
                "sample_id": sample_id,
                "input_spots": int(input_row["n_spots"]),
                "input_genes": int(input_row["n_genes"]),
                "non_integer_fraction_sample": float(input_row["non_integer_fraction_sample"]),
                "filter_min_genes": int(filter_row["min_genes"]),
                "filter_min_counts": int(filter_row["min_counts"]),
                "spots_kept_after_filter": int(filter_row["spots_kept"]),
                "spots_removed_after_filter": int(filter_row["spots_removed"]),
                "percent_removed_after_filter": float(filter_row["percent_removed"]),
            }
        )

    qc_df = pd.DataFrame(qc_rows)

    qc_df.to_csv(
        COMPARISON_FOLDER / "combined_qc_summary_all12.csv",
        index=False,
    )

    return qc_df


def plot_qc_summary(qc_df: pd.DataFrame) -> None:
    plot_path = FIGURE_FOLDER / "all12_qc_spots_kept_removed.png"

    x_positions = range(len(qc_df))

    plt.figure(figsize=(12, 6))

    plt.bar(
        x_positions,
        qc_df["spots_kept_after_filter"],
        label="Spots Kept",
    )

    plt.bar(
        x_positions,
        qc_df["spots_removed_after_filter"],
        bottom=qc_df["spots_kept_after_filter"],
        label="Spots Removed",
    )

    plt.xticks(
        list(x_positions),
        qc_df["sample_id"].astype(str),
        rotation=45,
        ha="right",
    )

    plt.title("QC Filtering Summary Across 12 Visium Samples")
    plt.xlabel("Sample ID")
    plt.ylabel("Number of Spots")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("Saved:", plot_path)


def combine_final_clustering_summaries() -> pd.DataFrame:
    summary_tables = []

    for sample_id in SAMPLE_IDS:
        file_path = OUTPUT_ROOT / sample_id / f"final_clustering_summary_{sample_id}.csv"
        df = pd.read_csv(file_path)
        summary_tables.append(df)

    combined_df = pd.concat(summary_tables, ignore_index=True)

    combined_df.to_csv(
        COMPARISON_FOLDER / "combined_final_clustering_summary_all12.csv",
        index=False,
    )

    return combined_df


def plot_ari_nmi_summary(combined_df: pd.DataFrame) -> None:
    plot_path = FIGURE_FOLDER / "all12_ari_nmi_comparison.png"

    x_positions = range(len(combined_df))

    plt.figure(figsize=(12, 6))

    plt.bar(
        [x - 0.2 for x in x_positions],
        combined_df["ARI"],
        width=0.4,
        label="ARI",
    )

    plt.bar(
        [x + 0.2 for x in x_positions],
        combined_df["NMI"],
        width=0.4,
        label="NMI",
    )

    plt.xticks(
        list(x_positions),
        combined_df["sample_id"].astype(str),
        rotation=45,
        ha="right",
    )

    plt.title("ARI/NMI Comparison Across 12 Visium Samples")
    plt.xlabel("Sample ID")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("Saved:", plot_path)


def check_outputs() -> None:
    expected_files = [
        "combined_qc_summary_all12.csv",
        "combined_final_clustering_summary_all12.csv",
    ]

    expected_figures = [
        "all12_qc_spots_kept_removed.png",
        "all12_ari_nmi_comparison.png",
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
        print("Missing expected comparison outputs:")

        for file_path in missing_files:
            print("-", file_path)

        raise FileNotFoundError("Some expected comparison output files are missing.")

    print("No missing files. All expected all-12 comparison outputs were generated successfully.")


def main() -> None:
    check_required_files()

    qc_df = combine_qc_summaries()
    plot_qc_summary(qc_df)

    combined_df = combine_final_clustering_summaries()
    plot_ari_nmi_summary(combined_df)

    check_outputs()

    print("\nAll-12 Visium comparison completed.")
    print("Saved outputs:")
    print("-", COMPARISON_FOLDER / "combined_qc_summary_all12.csv")
    print("-", FIGURE_FOLDER / "all12_qc_spots_kept_removed.png")
    print("-", COMPARISON_FOLDER / "combined_final_clustering_summary_all12.csv")
    print("-", FIGURE_FOLDER / "all12_ari_nmi_comparison.png")


if __name__ == "__main__":
    main()