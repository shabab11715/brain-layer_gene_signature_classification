# Added for local reusable Visium H5AD pipeline
# Runs all .h5ad files inside the Data folder and saves outputs per sample.

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def create_folders(output_folder: Path, figure_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    figure_folder.mkdir(parents=True, exist_ok=True)


def get_nonzero_sample(adata: sc.AnnData, max_values: int = 100000) -> np.ndarray:
    if sparse.issparse(adata.X):
        nonzero_values = adata.X.data
    else:
        dense_sample = np.asarray(adata.X[: min(100, adata.n_obs), :])
        nonzero_values = dense_sample[dense_sample != 0]

    if len(nonzero_values) == 0:
        raise ValueError("Expression matrix contains no non-zero values.")

    return nonzero_values[: min(max_values, len(nonzero_values))]


def validate_raw_count_like_data(adata: sc.AnnData) -> float:
    value_sample = get_nonzero_sample(adata)
    non_integer_fraction = float(
        np.mean(np.abs(value_sample - np.round(value_sample)) > 1e-6)
    )

    if non_integer_fraction > 0.01:
        raise ValueError(
            "The expression matrix does not look like raw count data. "
            "This pipeline expects raw count-like values before normalization. "
            f"Non-integer fraction: {non_integer_fraction}"
        )

    return non_integer_fraction


def load_and_validate_h5ad(
    h5ad_path: Path,
    sample_id: str,
    label_column_source: str,
    label_column: str,
    spatial_image_key: str,
) -> tuple[sc.AnnData, str, float]:
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing input file: {h5ad_path}")

    if h5ad_path.stat().st_size == 0:
        raise ValueError(f"{h5ad_path} exists but is empty.")

    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()

    if label_column_source not in adata.obs.columns:
        raise ValueError(
            f"Required label column '{label_column_source}' was not found in adata.obs. "
            f"Available columns: {adata.obs.columns.tolist()}"
        )

    adata.obs[label_column] = adata.obs[label_column_source].copy()

    if "spatial" not in adata.obsm:
        raise ValueError("Spatial coordinates were not found in adata.obsm['spatial'].")

    if "spatial" not in adata.uns:
        raise ValueError("Spatial image metadata was not found in adata.uns['spatial'].")

    spatial_library_keys = list(adata.uns["spatial"].keys())

    if len(spatial_library_keys) == 0:
        raise ValueError("adata.uns['spatial'] exists but contains no library keys.")

    if sample_id in spatial_library_keys:
        spatial_library_id = sample_id
    else:
        spatial_library_id = spatial_library_keys[0]
        print(
            f"Warning: SAMPLE_ID '{sample_id}' was not found in adata.uns['spatial']. "
            f"Using first spatial library key: {spatial_library_id}"
        )

    if "images" not in adata.uns["spatial"][spatial_library_id]:
        raise ValueError(f"No image dictionary found for spatial library '{spatial_library_id}'.")

    available_image_keys = list(adata.uns["spatial"][spatial_library_id]["images"].keys())

    if spatial_image_key not in available_image_keys:
        raise ValueError(
            f"Required spatial image key '{spatial_image_key}' was not found for library "
            f"'{spatial_library_id}'. Available image keys: {available_image_keys}"
        )

    non_integer_fraction = validate_raw_count_like_data(adata)

    print("Loaded H5AD successfully.")
    print("Sample ID:", sample_id)
    print("File:", h5ad_path)
    print("Shape:", adata.shape)
    print("Spatial library ID:", spatial_library_id)
    print("Spatial image key:", spatial_image_key)
    print("Non-integer fraction:", non_integer_fraction)
    print("\nLabel counts:")
    print(adata.obs[label_column].value_counts(dropna=False))

    return adata, spatial_library_id, non_integer_fraction


def save_input_summary(
    output_folder: Path,
    sample_id: str,
    h5ad_path: Path,
    adata: sc.AnnData,
    label_column_source: str,
    label_column: str,
    spatial_library_id: str,
    non_integer_fraction: float,
) -> None:
    input_h5ad_summary = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "h5ad_file": str(h5ad_path),
                "n_spots": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "source_label_column": label_column_source,
                "standardized_label_column": label_column,
                "spatial_library_id": spatial_library_id,
                "has_spatial_coordinates": "spatial" in adata.obsm,
                "has_spatial_images": "spatial" in adata.uns,
                "non_integer_fraction_sample": non_integer_fraction,
                "input_data_interpretation": "raw_count_like",
            }
        ]
    )

    input_h5ad_summary.to_csv(
        output_folder / f"input_h5ad_summary_{sample_id}.csv",
        index=False,
    )


def save_spots_only_plot(adata: sc.AnnData, figure_folder: Path, sample_id: str) -> None:
    output_path = figure_folder / f"spatial_spots_only_total_counts_{sample_id}.png"

    coords = adata.obsm["spatial"]

    plt.figure(figsize=(7, 7))
    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=adata.obs["total_counts"],
        s=8,
    )

    plt.gca().invert_yaxis()
    plt.colorbar(label="Total Counts")
    plt.title(f"{sample_id} Spatial Spots Only Colored by Total Counts")
    plt.xlabel("spatial1")
    plt.ylabel("spatial2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def save_spatial_plot(
    adata: sc.AnnData,
    figure_folder: Path,
    file_name: str,
    library_id: str,
    img_key: str,
    title: str,
    color: str | None = None,
    spot_size: int = 80,
    alpha_img: float = 0.25,
) -> None:
    output_path = figure_folder / file_name

    # Added because Squidpy size is a scale factor, not the same as Scanpy spot_size.
    # Old Scanpy default spot_size=80 is treated here as Squidpy size=1.0.
    squidpy_size = max(float(spot_size) / 80.0, 0.05)

    plot_kwargs = {
        "adata": adata,
        "library_id": library_id,
        "img_res_key": img_key,
        "spatial_key": "spatial",
        "img": True,
        "img_alpha": alpha_img,
        "size": squidpy_size,
        "title": title,
    }

    if color is not None:
        plot_kwargs["color"] = color

    sq.pl.spatial_scatter(**plot_kwargs)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    print("Saved:", output_path)


def save_qc_histograms(adata: sc.AnnData, figure_folder: Path, sample_id: str) -> None:
    total_counts_path = figure_folder / f"qc_total_counts_distribution_{sample_id}.png"

    plt.figure(figsize=(7, 5))
    plt.hist(adata.obs["total_counts"], bins=50)
    plt.title(f"{sample_id} Distribution of Total Counts per Spot")
    plt.xlabel("Total Counts")
    plt.ylabel("Number of Spots")
    plt.tight_layout()
    plt.savefig(total_counts_path, dpi=300)
    plt.close()

    print("Saved:", total_counts_path)

    detected_genes_path = figure_folder / f"qc_detected_genes_distribution_{sample_id}.png"

    plt.figure(figsize=(7, 5))
    plt.hist(adata.obs["n_genes_by_counts"], bins=50)
    plt.title(f"{sample_id} Distribution of Detected Genes per Spot")
    plt.xlabel("Number of Genes Detected")
    plt.ylabel("Number of Spots")
    plt.tight_layout()
    plt.savefig(detected_genes_path, dpi=300)
    plt.close()

    print("Saved:", detected_genes_path)


def save_filter_threshold_summary(adata: sc.AnnData, output_folder: Path, sample_id: str) -> pd.DataFrame:
    total_spots = adata.n_obs

    threshold_tests = [
        {"min_genes": 200, "min_counts": 500},
        {"min_genes": 500, "min_counts": 1000},
        {"min_genes": 800, "min_counts": 1500},
        {"min_genes": 1000, "min_counts": 2000},
    ]

    results = []

    for test in threshold_tests:
        min_genes = test["min_genes"]
        min_counts = test["min_counts"]

        keep_mask = (
            (adata.obs["n_genes_by_counts"] >= min_genes)
            & (adata.obs["total_counts"] >= min_counts)
        )

        kept = int(keep_mask.sum())
        removed = int(total_spots - kept)

        results.append(
            {
                "sample_id": sample_id,
                "min_genes": min_genes,
                "min_counts": min_counts,
                "spots_kept": kept,
                "spots_removed": removed,
                "percent_removed": round((removed / total_spots) * 100, 2),
            }
        )

    filter_threshold_summary = pd.DataFrame(results)

    filter_threshold_summary.to_csv(
        output_folder / f"filter_threshold_summary_{sample_id}.csv",
        index=False,
    )

    return filter_threshold_summary


def apply_light_filter(adata: sc.AnnData) -> sc.AnnData:
    adata_qc = adata[
        (adata.obs["n_genes_by_counts"] >= 200)
        & (adata.obs["total_counts"] >= 500)
    ].copy()

    print("Original spots:", adata.n_obs)
    print("Filtered spots:", adata_qc.n_obs)
    print("Removed spots:", adata.n_obs - adata_qc.n_obs)

    return adata_qc


def save_kept_removed_plot(
    adata: sc.AnnData,
    adata_qc: sc.AnnData,
    figure_folder: Path,
    sample_id: str,
    spatial_library_id: str,
    spatial_image_key: str,
) -> None:
    removed_mask = ~adata.obs_names.isin(adata_qc.obs_names)

    adata_filter_check = adata.copy()
    adata_filter_check.obs["filter_status"] = "kept"
    adata_filter_check.obs.loc[removed_mask, "filter_status"] = "removed"

    save_spatial_plot(
        adata=adata_filter_check,
        figure_folder=figure_folder,
        file_name=f"filter_kept_vs_removed_{sample_id}.png",
        library_id=spatial_library_id,
        img_key=spatial_image_key,
        color="filter_status",
        title=f"{sample_id} Kept vs Removed Spots After Light Filter",
    )


def preprocess_for_clustering(adata_qc: sc.AnnData) -> None:
    adata_qc.raw = adata_qc.copy()

    sc.pp.normalize_total(adata_qc, target_sum=1e4)
    sc.pp.log1p(adata_qc)

    sc.pp.highly_variable_genes(
        adata_qc,
        flavor="seurat",
        n_top_genes=2000,
    )

    # Added zero_center=False to avoid sparse matrix densification warning
    sc.pp.scale(
        adata_qc,
        max_value=10,
        zero_center=False,
    )

    # Updated because use_highly_variable=True is deprecated
    sc.tl.pca(
        adata_qc,
        mask_var="highly_variable",
        svd_solver="arpack",
    )


def save_pca_plot(adata_qc: sc.AnnData, figure_folder: Path, sample_id: str) -> None:
    output_path = figure_folder / f"pca_total_counts_{sample_id}.png"

    sc.pl.pca(
        adata_qc,
        color="total_counts",
        show=False,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_path)


def save_umap_plot(
    adata_qc: sc.AnnData,
    figure_folder: Path,
    file_name: str,
    color: str,
    title: str,
) -> None:
    output_path = figure_folder / file_name

    if "X_umap" not in adata_qc.obsm:
        raise ValueError("UMAP coordinates are missing. Run sc.tl.umap before plotting.")

    sc.pl.umap(
        adata_qc,
        color=color,
        title=title,
        show=False,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_path)


def make_safe_file_part(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return safe_value.strip("_")


def save_top_signature_gene_spatial_maps(
    adata_de: sc.AnnData,
    top10_signature_genes_clean: pd.DataFrame,
    output_folder: Path,
    figure_folder: Path,
    sample_id: str,
    spatial_library_id: str,
    spatial_image_key: str,
) -> None:
    gene_figure_folder = figure_folder / "signature_gene_spatial_maps"
    gene_figure_folder.mkdir(parents=True, exist_ok=True)

    plotted_rows = []

    top_gene_per_group = (
        top10_signature_genes_clean
        .groupby("group", group_keys=False, observed=False)
        .head(1)
        .reset_index(drop=True)
    )

    for _, row in top_gene_per_group.iterrows():
        group_name = str(row["group"])
        gene_name = str(row["names"])

        if gene_name not in adata_de.var_names:
            continue

        safe_group = make_safe_file_part(group_name)
        safe_gene = make_safe_file_part(gene_name)

        file_name = f"top_signature_gene_{safe_group}_{safe_gene}_{sample_id}.png"

        save_spatial_plot(
            adata=adata_de,
            figure_folder=gene_figure_folder,
            file_name=file_name,
            library_id=spatial_library_id,
            img_key=spatial_image_key,
            color=gene_name,
            title=f"{sample_id} Top Signature Gene for {group_name}: {gene_name}",
        )

        plotted_rows.append(
            {
                "sample_id": sample_id,
                "group": group_name,
                "gene": gene_name,
                "figure_file": str(gene_figure_folder / file_name),
            }
        )

    plotted_df = pd.DataFrame(plotted_rows)

    plotted_df.to_csv(
        output_folder / f"top_signature_gene_spatial_maps_{sample_id}.csv",
        index=False,
    )

    print("Saved:", output_folder / f"top_signature_gene_spatial_maps_{sample_id}.csv")


def run_baseline_clustering(
    adata_qc: sc.AnnData,
    output_folder: Path,
    figure_folder: Path,
    sample_id: str,
    label_column: str,
    spatial_library_id: str,
    spatial_image_key: str,
) -> None:
    sc.pp.neighbors(
        adata_qc,
        n_neighbors=10,
        n_pcs=30,
    )

    # Added igraph Leiden backend to avoid FutureWarning and improve clustering speed
    sc.tl.leiden(
        adata_qc,
        resolution=0.5,
        key_added="leiden_clusters",
        random_state=42,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )

    save_spatial_plot(
        adata=adata_qc,
        figure_folder=figure_folder,
        file_name=f"baseline_leiden_clusters_{sample_id}.png",
        library_id=spatial_library_id,
        img_key=spatial_image_key,
        color="leiden_clusters",
        title=f"{sample_id} Baseline Leiden Clusters on Tissue",
    )

    valid_mask = adata_qc.obs[label_column].notna()

    true_labels = adata_qc.obs.loc[valid_mask, label_column]
    predicted_clusters = adata_qc.obs.loc[valid_mask, "leiden_clusters"]

    baseline_ari = adjusted_rand_score(true_labels, predicted_clusters)
    baseline_nmi = normalized_mutual_info_score(true_labels, predicted_clusters)

    baseline_clustering_summary = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "cluster_column": "leiden_clusters",
                "backend": "igraph",
                "n_neighbors": 10,
                "n_pcs": 30,
                "resolution": 0.5,
                "num_clusters": int(adata_qc.obs["leiden_clusters"].nunique()),
                "ARI": baseline_ari,
                "NMI": baseline_nmi,
                "notes": "Baseline igraph Leiden clustering before parameter tuning.",
            }
        ]
    )

    baseline_clustering_summary.to_csv(
        output_folder / f"baseline_clustering_summary_{sample_id}.csv",
        index=False,
    )

    comparison_table = pd.crosstab(
        adata_qc.obs.loc[valid_mask, label_column],
        adata_qc.obs.loc[valid_mask, "leiden_clusters"],
    )

    comparison_table.to_csv(
        output_folder / f"baseline_label_vs_cluster_table_{sample_id}.csv",
        index_label="brain_layer",
    )


def run_leiden_tuning(
    adata_qc: sc.AnnData,
    output_folder: Path,
    sample_id: str,
    label_column: str,
) -> pd.DataFrame:
    results = []

    valid_mask = adata_qc.obs[label_column].notna()
    true_labels = adata_qc.obs.loc[valid_mask, label_column]

    n_neighbors_list = [10, 15, 20, 30]
    n_pcs_list = [20, 30, 40, 50]
    resolution_list = [0.4, 0.6, 0.8, 1.0]

    for n_neighbors in n_neighbors_list:
        for n_pcs in n_pcs_list:
            print(f"Running igraph Leiden: n_neighbors={n_neighbors}, n_pcs={n_pcs}")

            sc.pp.neighbors(
                adata_qc,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
            )

            for resolution in resolution_list:
                cluster_key = f"leiden_igraph_n{n_neighbors}_pc{n_pcs}_r{resolution}"

                sc.tl.leiden(
                    adata_qc,
                    resolution=resolution,
                    key_added=cluster_key,
                    random_state=42,
                    flavor="igraph",
                    n_iterations=2,
                    directed=False,
                )

                predicted_clusters = adata_qc.obs.loc[valid_mask, cluster_key]

                ari = adjusted_rand_score(true_labels, predicted_clusters)
                nmi = normalized_mutual_info_score(true_labels, predicted_clusters)
                num_clusters = int(adata_qc.obs[cluster_key].nunique())

                results.append(
                    {
                        "sample_id": sample_id,
                        "backend": "igraph",
                        "n_neighbors": n_neighbors,
                        "n_pcs": n_pcs,
                        "resolution": resolution,
                        "num_clusters": num_clusters,
                        "ARI": ari,
                        "NMI": nmi,
                        "cluster_key": cluster_key,
                    }
                )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_folder / f"leiden_tuning_results_{sample_id}.csv",
        index=False,
    )

    return results_df


def select_best_clustering(
    adata_qc: sc.AnnData,
    results_df: pd.DataFrame,
    output_folder: Path,
    sample_id: str,
    label_column: str,
) -> tuple[str, pd.Series, float, float]:
    valid_mask = adata_qc.obs[label_column].notna()
    true_labels = adata_qc.obs.loc[valid_mask, label_column]

    candidate_comparison_df = results_df.copy()

    candidate_comparison_df = candidate_comparison_df.sort_values(
        ["ARI", "NMI"],
        ascending=False,
    ).reset_index(drop=True)

    candidate_comparison_df.to_csv(
        output_folder / f"leiden_tuning_ranked_results_{sample_id}.csv",
        index=False,
    )

    best_candidate = candidate_comparison_df.iloc[0]
    best_cluster_key = str(best_candidate["cluster_key"])

    if best_cluster_key not in adata_qc.obs.columns:
        raise KeyError(f"Selected cluster column not found: {best_cluster_key}")

    adata_qc.obs["best_leiden_cluster"] = adata_qc.obs[best_cluster_key]

    sc.pp.neighbors(
        adata_qc,
        n_neighbors=int(best_candidate["n_neighbors"]),
        n_pcs=int(best_candidate["n_pcs"]),
    )

    sc.tl.umap(adata_qc, random_state=42)

    best_predicted_clusters = adata_qc.obs.loc[valid_mask, "best_leiden_cluster"]

    final_ari = adjusted_rand_score(true_labels, best_predicted_clusters)
    final_nmi = normalized_mutual_info_score(true_labels, best_predicted_clusters)

    if abs(final_ari - float(best_candidate["ARI"])) > 1e-10:
        raise ValueError("Final ARI does not match the selected best candidate ARI.")

    if abs(final_nmi - float(best_candidate["NMI"])) > 1e-10:
        raise ValueError("Final NMI does not match the selected best candidate NMI.")

    return best_cluster_key, best_candidate, final_ari, final_nmi

def remove_extra_tuning_cluster_columns(adata_qc: sc.AnnData) -> None:
    tuning_cluster_columns = [
        column
        for column in adata_qc.obs.columns
        if column.startswith("leiden_igraph_n")
    ]

    if len(tuning_cluster_columns) > 0:
        adata_qc.obs.drop(columns=tuning_cluster_columns, inplace=True)

        print(
            "Removed temporary Leiden tuning columns from final saved AnnData:",
            len(tuning_cluster_columns),
        )

def save_final_clustering_outputs(
    adata_qc: sc.AnnData,
    output_folder: Path,
    figure_folder: Path,
    sample_id: str,
    label_column: str,
    spatial_library_id: str,
    spatial_image_key: str,
    best_cluster_key: str,
    best_candidate: pd.Series,
    final_ari: float,
    final_nmi: float,
) -> None:
    save_spatial_plot(
        adata=adata_qc,
        figure_folder=figure_folder,
        file_name=f"final_best_leiden_clusters_{sample_id}.png",
        library_id=spatial_library_id,
        img_key=spatial_image_key,
        color="best_leiden_cluster",
        title=f"{sample_id} Final Best Leiden Clustering on Tissue",
    )

    save_umap_plot(
        adata_qc=adata_qc,
        figure_folder=figure_folder,
        file_name=f"umap_best_leiden_cluster_{sample_id}.png",
        color="best_leiden_cluster",
        title=f"{sample_id} UMAP Colored by Tuned Leiden Cluster",
    )

    save_umap_plot(
        adata_qc=adata_qc,
        figure_folder=figure_folder,
        file_name=f"umap_ground_truth_labels_{sample_id}.png",
        color=label_column,
        title=f"{sample_id} UMAP Colored by Ground Truth Brain Layer",
    )

    valid_mask = adata_qc.obs[label_column].notna()

    final_comparison_table = pd.crosstab(
        adata_qc.obs.loc[valid_mask, label_column],
        adata_qc.obs.loc[valid_mask, "best_leiden_cluster"],
    )

    final_clustering_summary = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "selected_cluster_column": "best_leiden_cluster",
                "original_cluster_key": best_cluster_key,
                "backend": best_candidate["backend"],
                "n_neighbors": int(best_candidate["n_neighbors"]),
                "n_pcs": int(best_candidate["n_pcs"]),
                "resolution": float(best_candidate["resolution"]),
                "num_clusters": int(best_candidate["num_clusters"]),
                "ARI": final_ari,
                "NMI": final_nmi,
                "notes": (
                    "Label-informed parameter selection from the igraph Leiden tuning grid. "
                    "Report this as tuned clustering, not purely unsupervised discovery."
                ),
            }
        ]
    )

    final_clustering_summary.to_csv(
        output_folder / f"final_clustering_summary_{sample_id}.csv",
        index=False,
    )

    final_comparison_table.to_csv(
        output_folder / f"final_label_vs_cluster_table_{sample_id}.csv",
        index_label="brain_layer",
    )

    coords = adata_qc.obsm["spatial"]

    spot_cluster_assignments = adata_qc.obs[
        [
            label_column,
            "best_leiden_cluster",
            "total_counts",
            "n_genes_by_counts",
        ]
    ].copy()

    spot_cluster_assignments["spatial1"] = coords[:, 0]
    spot_cluster_assignments["spatial2"] = coords[:, 1]

    spot_cluster_assignments.to_csv(
        output_folder / f"final_spot_cluster_assignments_{sample_id}.csv",
        index_label="barcode",
    )

    remove_extra_tuning_cluster_columns(adata_qc)

    adata_qc.write_h5ad(
        output_folder / f"adata_qc_final_{sample_id}.h5ad"
    )


def run_signature_gene_analysis(
    adata: sc.AnnData,
    output_folder: Path,
    figure_folder: Path,
    sample_id: str,
    label_column: str,
    spatial_library_id: str,
    spatial_image_key: str,
) -> None:
    adata_de = adata[
        (adata.obs["n_genes_by_counts"] >= 200)
        & (adata.obs["total_counts"] >= 500)
    ].copy()

    adata_de.raw = None

    if label_column not in adata_de.obs.columns:
        raise ValueError(f"{label_column} was not found in adata_de.obs.")

    adata_de = adata_de[adata_de.obs[label_column].notna()].copy()

    sc.pp.normalize_total(adata_de, target_sum=1e4)
    sc.pp.log1p(adata_de)

    sc.tl.rank_genes_groups(
        adata_de,
        groupby=label_column,
        method="wilcoxon",
        use_raw=False,
    )

    signature_results = sc.get.rank_genes_groups_df(
        adata_de,
        group=None,
    )

    signature_results.to_csv(
        output_folder / f"signature_genes_all_{sample_id}.csv",
        index=False,
    )

    signature_results_clean = signature_results[
        ~signature_results["names"].astype(str).str.startswith("MT-")
    ].copy()

    top10_signature_genes_clean = (
        signature_results_clean
        .groupby("group", observed=False)
        .head(10)
        .reset_index(drop=True)
    )

    expected_marker_groups = sorted(signature_results["group"].unique())

    clean_marker_counts = (
        top10_signature_genes_clean
        .groupby("group", observed=False)
        .size()
        .reindex(expected_marker_groups, fill_value=0)
    )

    if (clean_marker_counts < 10).any():
        print("Warning: Some groups have fewer than 10 clean non-mitochondrial signature genes.")
        print(clean_marker_counts)

    top10_signature_genes_clean.to_csv(
        output_folder / f"top10_signature_genes_clean_{sample_id}.csv",
        index=False,
    )

    save_top_signature_gene_spatial_maps(
        adata_de=adata_de,
        top10_signature_genes_clean=top10_signature_genes_clean,
        output_folder=output_folder,
        figure_folder=figure_folder,
        sample_id=sample_id,
        spatial_library_id=spatial_library_id,
        spatial_image_key=spatial_image_key,
    )


def check_expected_outputs(output_folder: Path, figure_folder: Path, sample_id: str) -> None:
    expected_output_files = [
        f"input_h5ad_summary_{sample_id}.csv",
        f"filter_threshold_summary_{sample_id}.csv",
        f"baseline_clustering_summary_{sample_id}.csv",
        f"baseline_label_vs_cluster_table_{sample_id}.csv",
        f"leiden_tuning_results_{sample_id}.csv",
        f"leiden_tuning_ranked_results_{sample_id}.csv",
        f"final_clustering_summary_{sample_id}.csv",
        f"final_label_vs_cluster_table_{sample_id}.csv",
        f"final_spot_cluster_assignments_{sample_id}.csv",
        f"adata_qc_final_{sample_id}.h5ad",
        f"signature_genes_all_{sample_id}.csv",
        f"top10_signature_genes_clean_{sample_id}.csv",
        f"top_signature_gene_spatial_maps_{sample_id}.csv",
    ]

    expected_figure_files = [
        f"spatial_spots_only_total_counts_{sample_id}.png",
        f"tissue_spot_overlay_{sample_id}.png",
        f"spatial_total_counts_{sample_id}.png",
        f"qc_total_counts_distribution_{sample_id}.png",
        f"qc_detected_genes_distribution_{sample_id}.png",
        f"filtered_spots_total_counts_{sample_id}.png",
        f"filter_kept_vs_removed_{sample_id}.png",
        f"pca_total_counts_{sample_id}.png",
        f"baseline_leiden_clusters_{sample_id}.png",
        f"ground_truth_labels_{sample_id}.png",
        f"final_best_leiden_clusters_{sample_id}.png",
        f"umap_best_leiden_cluster_{sample_id}.png",
        f"umap_ground_truth_labels_{sample_id}.png",
    ]

    existing_output_files = os.listdir(output_folder)
    existing_figure_files = os.listdir(figure_folder)

    missing_output_files = [
        file_name for file_name in expected_output_files
        if file_name not in existing_output_files
    ]

    missing_figure_files = [
        file_name for file_name in expected_figure_files
        if file_name not in existing_figure_files
    ]

    signature_gene_map_csv = output_folder / f"top_signature_gene_spatial_maps_{sample_id}.csv"

    missing_signature_gene_map_files = []

    if signature_gene_map_csv.exists():
        signature_gene_map_df = pd.read_csv(signature_gene_map_csv)

        if "figure_file" in signature_gene_map_df.columns:
            for figure_file in signature_gene_map_df["figure_file"].dropna():
                if not Path(figure_file).exists():
                    missing_signature_gene_map_files.append(figure_file)

    if missing_output_files or missing_figure_files or missing_signature_gene_map_files:
        print("\nMissing expected output files:")
        for file_name in missing_output_files:
            print("-", file_name)

        print("\nMissing expected figure files:")
        for file_name in missing_figure_files:
            print("-", file_name)

        print("\nMissing signature gene spatial map files:")
        for file_name in missing_signature_gene_map_files:
            print("-", file_name)

        raise FileNotFoundError("Some expected output or figure files are missing.")

    print("No missing files. All expected outputs and figures were generated successfully.")


def process_sample(
    h5ad_path: Path,
    output_root: Path,
    label_column_source: str,
    label_column: str,
    spatial_image_key: str,
) -> dict:
    sample_id = h5ad_path.stem

    output_folder = output_root / sample_id
    figure_folder = output_folder / "figures"

    create_folders(output_folder, figure_folder)

    print("\n" + "=" * 80)
    print(f"Processing sample: {sample_id}")
    print("=" * 80)

    adata, spatial_library_id, non_integer_fraction = load_and_validate_h5ad(
        h5ad_path=h5ad_path,
        sample_id=sample_id,
        label_column_source=label_column_source,
        label_column=label_column,
        spatial_image_key=spatial_image_key,
    )

    save_input_summary(
        output_folder=output_folder,
        sample_id=sample_id,
        h5ad_path=h5ad_path,
        adata=adata,
        label_column_source=label_column_source,
        label_column=label_column,
        spatial_library_id=spatial_library_id,
        non_integer_fraction=non_integer_fraction,
    )

    sc.pp.calculate_qc_metrics(adata, inplace=True)

    save_spots_only_plot(adata, figure_folder, sample_id)

    save_spatial_plot(
        adata=adata,
        figure_folder=figure_folder,
        file_name=f"tissue_spot_overlay_{sample_id}.png",
        library_id=spatial_library_id,
        img_key=spatial_image_key,
        color=None,
        title=f"{sample_id} Tissue Image with Spatial Spot Overlay",
        spot_size=4,
        alpha_img=0.5,
    )

    save_spatial_plot(
        adata=adata,
        figure_folder=figure_folder,
        file_name=f"spatial_total_counts_{sample_id}.png",
        library_id=spatial_library_id,
        img_key=spatial_image_key,
        color="total_counts",
        title=f"{sample_id} Spatial Spots Colored by Total Counts",
    )

    save_qc_histograms(adata, figure_folder, sample_id)

    save_filter_threshold_summary(adata, output_folder, sample_id)

    adata_qc = apply_light_filter(adata)

    save_spatial_plot(
        adata=adata_qc,
        figure_folder=figure_folder,
        file_name=f"filtered_spots_total_counts_{sample_id}.png",
        library_id=spatial_library_id,
        img_key=spatial_image_key,
        color="total_counts",
        title=f"{sample_id} Filtered Spots Colored by Total Counts",
    )

    save_kept_removed_plot(
        adata=adata,
        adata_qc=adata_qc,
        figure_folder=figure_folder,
        sample_id=sample_id,
        spatial_library_id=spatial_library_id,
        spatial_image_key=spatial_image_key,
    )

    preprocess_for_clustering(adata_qc)

    save_pca_plot(adata_qc, figure_folder, sample_id)

    run_baseline_clustering(
        adata_qc=adata_qc,
        output_folder=output_folder,
        figure_folder=figure_folder,
        sample_id=sample_id,
        label_column=label_column,
        spatial_library_id=spatial_library_id,
        spatial_image_key=spatial_image_key,
    )

    save_spatial_plot(
        adata=adata_qc,
        figure_folder=figure_folder,
        file_name=f"ground_truth_labels_{sample_id}.png",
        library_id=spatial_library_id,
        img_key=spatial_image_key,
        color=label_column,
        title=f"{sample_id} Ground Truth Brain-Layer Labels",
    )

    results_df = run_leiden_tuning(
        adata_qc=adata_qc,
        output_folder=output_folder,
        sample_id=sample_id,
        label_column=label_column,
    )

    best_cluster_key, best_candidate, final_ari, final_nmi = select_best_clustering(
        adata_qc=adata_qc,
        results_df=results_df,
        output_folder=output_folder,
        sample_id=sample_id,
        label_column=label_column,
    )

    save_final_clustering_outputs(
        adata_qc=adata_qc,
        output_folder=output_folder,
        figure_folder=figure_folder,
        sample_id=sample_id,
        label_column=label_column,
        spatial_library_id=spatial_library_id,
        spatial_image_key=spatial_image_key,
        best_cluster_key=best_cluster_key,
        best_candidate=best_candidate,
        final_ari=final_ari,
        final_nmi=final_nmi,
    )

    run_signature_gene_analysis(
        adata=adata,
        output_folder=output_folder,
        figure_folder=figure_folder,
        sample_id=sample_id,
        label_column=label_column,
        spatial_library_id=spatial_library_id,
        spatial_image_key=spatial_image_key,
    )

    check_expected_outputs(output_folder, figure_folder, sample_id)

    return {
        "sample_id": sample_id,
        "final_ari": final_ari,
        "final_nmi": final_nmi,
        "best_cluster_key": best_cluster_key,
        "backend": best_candidate["backend"],
        "status": "completed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reusable Visium H5AD spatial transcriptomics pipeline locally."
    )

    parser.add_argument(
        "--input-dir",
        default="Data",
        help="Folder containing .h5ad files. Default: Data",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Folder where outputs will be saved. Default: outputs",
    )

    parser.add_argument(
        "--sample-id",
        default=None,
        help="Optional single sample ID to process, for example 151508. If omitted, all .h5ad files are processed.",
    )

    parser.add_argument(
        "--label-column-source",
        default="Region",
        help="Source label column inside adata.obs. Default: Region",
    )

    parser.add_argument(
        "--label-column",
        default="brain_region_label",
        help="Standardized label column to create/use in the pipeline. Default: brain_region_label",
    )

    parser.add_argument(
        "--spatial-image-key",
        default="hires",
        help="Spatial image key to use for tissue plots. Default: hires",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    if args.sample_id:
        h5ad_files = [input_dir / f"{args.sample_id}.h5ad"]
    else:
        h5ad_files = sorted(input_dir.glob("*.h5ad"))

    if len(h5ad_files) == 0:
        raise FileNotFoundError(f"No .h5ad files found inside: {input_dir}")

    run_results = []

    for h5ad_path in h5ad_files:
        result = process_sample(
            h5ad_path=h5ad_path,
            output_root=output_root,
            label_column_source=args.label_column_source,
            label_column=args.label_column,
            spatial_image_key=args.spatial_image_key,
        )

        run_results.append(result)

    run_summary_df = pd.DataFrame(run_results)

    output_root.mkdir(parents=True, exist_ok=True)

    run_summary_path = output_root / "run_summary_all_samples.csv"

    run_summary_df.to_csv(run_summary_path, index=False)

    print("\n" + "=" * 80)
    print("Pipeline finished.")
    print("Run summary saved:", run_summary_path)
    print(run_summary_df)


if __name__ == "__main__":
    main()
