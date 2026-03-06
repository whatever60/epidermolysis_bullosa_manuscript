#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


COMMON_SETUP = """
from pathlib import Path
import sys

import pandas as pd
from IPython.display import Markdown, SVG, display

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workflow_core as wc

context, base_data, base, advanced = wc.bootstrap_notebook()
"""


R_COMMON_SETUP = """
options(width = 140)
suppressPackageStartupMessages({
    library(readr)
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library(stringr)
})

theme_set(theme_bw(base_size = 12))

root <- getwd()
data_dir <- dirname(root)
figure_dir <- file.path(root, "figures")
table_dir <- file.path(root, "tables")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(table_dir, showWarnings = FALSE, recursive = TRUE)

table_id_map <- c(
  `1` = "01_01", `2` = "02_01", `3` = "02_02", `4` = "03_01", `5` = "03_02",
  `6` = "04_01", `7` = "05_01", `8` = "06_01", `9` = "06_02", `10` = "06_03",
  `11` = "06_04", `12` = "06_05", `13` = "06_06", `14` = "07_01", `15` = "07_02",
  `16` = "07_03", `17` = "07_04", `18` = "07_05", `19` = "07_06", `20` = "08_01",
  `21` = "08_02", `22` = "08_03", `23` = "09_01", `24` = "09_02", `25` = "10_01",
  `26` = "10_02", `27` = "11_01", `28` = "11_02", `29` = "12_01", `30` = "12_02",
  `31` = "12_03", `32` = "13_01", `33` = "13_02", `34` = "13_03", `35` = "13_04",
  `36` = "14_01", `37` = "14_02", `38` = "12_04", `39` = "12_05", `40` = "12_06",
  `41` = "12_07", `42` = "13_05", `43` = "13_06"
)

figure_id_map <- c(
  `1` = "02_01", `2` = "03_01", `3` = "04_01", `4` = "05_01", `5` = "06_01",
  `6` = "06_02", `7` = "07_01", `8` = "08_01", `9` = "09_01", `10` = "10_01",
  `11` = "11_01", `12` = "12_01", `13` = "12_02", `14` = "13_01", `15` = "13_02",
  `16` = "13_03", `17` = "13_04", `18` = "14_01", `19` = "14_02", `20` = "14_03",
  `21` = "14_04", `22` = "12_03", `23` = "12_04", `24` = "12_05"
)

table_file <- function(number, slug) {
  prefix <- table_id_map[[as.character(number)]]
  file.path(table_dir, sprintf("table_%s_%s.tsv", prefix, slug))
}

figure_file <- function(number, slug) {
  prefix <- figure_id_map[[as.character(number)]]
  file.path(figure_dir, sprintf("fig_%s_%s.svg", prefix, slug))
}
"""


def build_notebooks(output_dir: Path) -> list[Path]:
    notebooks: list[tuple[str, list]] = [
        (
            "01_project_inventory_and_metadata.ipynb",
            [
                md(
                    """
                    # 01. Project Inventory And Metadata

                    This notebook establishes the analysis inputs, harmonizes sample identifiers, merges the two metadata workbooks,
                    and defines the patient-date, batch, and patient-relative time variables used later in the modeling workflow.
                    """
                ),
                code(COMMON_SETUP),
                md(
                    """
                    ## Define Notebook-Local Metadata Helpers

                    These helper functions are kept inside this notebook because they are only used here.
                    """
                ),
                code(
                    """
                    def metadata_export_frame(metadata):
                        return metadata.reset_index().rename(columns={"index": "sample_id"})


                    def load_full_primary_metadata():
                        primary = pd.read_excel(
                            context.data_dir / "metadata" / "PA_Data_Finalized.xlsx",
                            sheet_name="Corrected EB wound spreadsheet",
                        ).rename(
                            columns={
                                "Code": "code",
                                "Date of Culture": "culture_date",
                                "Location": "location_raw",
                                "Clinical Correlates (chronicity, clinical signs of infection, SCC)": "clinical_correlates",
                                "Result": "culture_result",
                                "Gram Stain": "gram_stain",
                                "Sent to…": "sent_to",
                            }
                        )
                        primary["sample_id"] = primary["code"].map(base.normalize_sample_id)
                        primary = primary.loc[primary["sample_id"].astype(str).str.strip() != ""].copy()
                        primary["patient_id"] = primary["sample_id"].str.slice(0, 2)
                        primary = primary.loc[primary["patient_id"].str.fullmatch(r"\\d{2}", na=False)].copy()
                        primary["culture_date"] = pd.to_datetime(primary["culture_date"], errors="coerce")
                        primary = primary.loc[primary["culture_date"].notna()].copy()
                        primary["location"] = primary["location_raw"].map(base.standardize_location)
                        primary["body_region"] = primary["location"].map(base.infer_body_region)
                        primary["laterality"] = primary["location"].map(base.infer_laterality)
                        primary["visit_id"] = primary["patient_id"] + "_" + primary["culture_date"].dt.strftime("%Y-%m-%d")
                        return primary


                    def full_metadata_site_visit_summary():
                        primary = load_full_primary_metadata()
                        per_patient = (
                            primary.groupby("patient_id")
                            .agg(
                                n_records=("sample_id", "size"),
                                n_dates=("culture_date", "nunique"),
                                n_visits=("visit_id", "nunique"),
                                n_clean_locations=("location", "nunique"),
                                n_body_regions=("body_region", "nunique"),
                            )
                            .reset_index()
                            .sort_values(["n_records", "n_clean_locations", "n_dates"], ascending=[False, False, False])
                        )
                        visit_sites = (
                            primary.groupby(["patient_id", "visit_id"])["location"]
                            .apply(lambda s: sorted(set(s)))
                            .reset_index(name="sites")
                        )
                        visit_sites["n_sites"] = visit_sites["sites"].str.len()
                        revisited_sites = (
                            primary.groupby(["patient_id", "location"])["visit_id"]
                            .apply(lambda s: sorted(set(s)))
                            .reset_index(name="visits")
                        )
                        revisited_sites["n_visits"] = revisited_sites["visits"].str.len()
                        overall = pd.DataFrame(
                            [
                                {
                                    "n_rows": int(primary.shape[0]),
                                    "n_patients": int(primary["patient_id"].nunique()),
                                    "n_patient_visit_pairs": int(primary[["patient_id", "visit_id"]].drop_duplicates().shape[0]),
                                    "n_patient_site_pairs": int(primary[["patient_id", "location"]].drop_duplicates().shape[0]),
                                    "multisite_visits": int((visit_sites["n_sites"] > 1).sum()),
                                    "total_visits": int(visit_sites.shape[0]),
                                    "revisited_sites": int((revisited_sites["n_visits"] > 1).sum()),
                                    "total_patient_sites": int(revisited_sites.shape[0]),
                                    "median_dates_per_patient": float(per_patient["n_dates"].median()),
                                    "median_sites_per_patient": float(per_patient["n_clean_locations"].median()),
                                }
                            ]
                        )
                        return {
                            "overall": overall,
                            "per_patient": per_patient,
                            "visit_sites": visit_sites,
                            "revisited_sites": revisited_sites,
                            "primary": primary,
                        }


                    def repeated_measure_summary(metadata):
                        grouped = metadata.reset_index().groupby("patient_id")
                        summary = grouped.agg(
                            n_samples=("sample_id", "size"),
                            n_visits=("visit_id", "nunique"),
                            n_dates=("culture_date", "nunique"),
                        ).reset_index()
                        summary["multi_date"] = summary["n_dates"] > 1
                        return summary.sort_values(["n_dates", "n_samples", "patient_id"], ascending=[False, False, True])
                    """
                ),
                md(
                    """
                    ## Load And Save The Cleaned Metadata Table

                    The primary metadata source is `PA_Data_Finalized.xlsx`, sheet `Corrected EB wound spreadsheet`.
                    Missing date, location, and culture-note fields are backfilled from the lab-archive workbook.
                    """
                ),
                code(
                    """
                    metadata = base_data["metadata"]
                    metadata_table = metadata_export_frame(metadata)
                    wc.save_table(metadata_table, wc.table_path(context, 1, "cleaned_metadata"))

                    display(metadata_table[
                        [
                            "sample_id",
                            "patient_id",
                            "visit_id",
                            "batch_id",
                            "culture_date",
                            "days_since_first_sample",
                            "years_since_first_sample",
                            "location_raw",
                            "location",
                            "body_region",
                            "laterality",
                            "chronicity_group",
                            "clinical_infection_flag",
                            "culture_result",
                        ]
                    ].head(20))
                    """
                ),
                md(
                    """
                    ## Patient-Date, Batch, And Elapsed-Time Structure

                    `visit_id` is still defined as `patient_id + culture_date` so same-day multisite swabs stay grouped.
                    But later models now separate two different roles of the date:
                    the absolute `culture_date` is treated as a technical batch variable, while elapsed time since a patient's first sample
                    is treated as the biological time variable.
                    """
                ),
                md(
                    """
                    ## Metadata Processing Logic

                    The metadata-cleaning logic is intentionally explicit:

                    - `culture_date` is preserved in three derived forms.
                    - `visit_id = patient_id + culture_date` is retained only as a patient-date identifier for same-day multisite grouping.
                    - `batch_id = culture_date` is the technical batch/date variable used in updated host, similarity, and concordance models.
                    - `days_since_first_sample` and `years_since_first_sample` are the patient-relative biological time variables used instead of absolute year for longitudinal interpretation.
                    - Therefore, later notebooks do **not** interpret absolute date as biology by default.
                    - `patient + site` is not equivalent to `patient + visit_id` in this cohort: some dates contain multiple sampled body sites, and some body sites are revisited across different dates.
                    - Raw `Location`, `Clinical Correlates`, `Result`, and `Gram Stain` are preserved and then mapped into cleaned analysis variables so the transformation remains auditable.
                    """
                ),
                code(
                    """
                    visit_summary = repeated_measure_summary(metadata)
                    full_meta = full_metadata_site_visit_summary()
                    full_overall = full_meta["overall"].iloc[0]
                    multisite_examples = full_meta["visit_sites"].loc[full_meta["visit_sites"]["n_sites"] > 1].head(10)
                    revisited_examples = full_meta["revisited_sites"].loc[full_meta["revisited_sites"]["n_visits"] > 1].head(10)
                    body_region_counts = metadata_table["body_region"].value_counts(dropna=False).rename_axis("body_region").reset_index(name="n_samples")
                    chronicity_counts = metadata_table["chronicity_group"].value_counts(dropna=False).rename_axis("chronicity_group").reset_index(name="n_samples")

                    display(body_region_counts)
                    display(chronicity_counts)
                    display(visit_summary)
                    display(full_meta["overall"])
                    display(multisite_examples)
                    display(revisited_examples)

                    summary_lines = [
                        f"- Sequenced swabs: {len(metadata_table)} from {metadata_table['patient_id'].nunique()} patients.",
                        f"- Unique patient-date identifiers: {metadata_table['visit_id'].nunique()} across {metadata_table['batch_id'].nunique()} culture-date batches.",
                        f"- Patients sampled on more than one date: {int(visit_summary['multi_date'].sum())} of {visit_summary.shape[0]}.",
                        f"- In the full primary metadata sheet, valid coded rows covered {int(full_overall['n_rows'])} samples from {int(full_overall['n_patients'])} patients.",
                        f"- Full-metadata patient-date pairs: {int(full_overall['n_patient_visit_pairs'])}; patient-site pairs: {int(full_overall['n_patient_site_pairs'])}.",
                        f"- Multisite same-date sampling occurred in {int(full_overall['multisite_visits'])} of {int(full_overall['total_visits'])} patient-date groups.",
                        f"- Revisited patient-site combinations occurred in {int(full_overall['revisited_sites'])} of {int(full_overall['total_patient_sites'])} patient-site groups.",
                        f"- Median patient-relative elapsed time was {metadata_table['days_since_first_sample'].median():.0f} days since first sample.",
                        "- Cleaned body-site labels were collapsed into head/neck, upper extremity, trunk/perineum, and lower extremity.",
                        "- Clinical free text was parsed into chronicity and infection-status categories; these are rule-based and remain approximate.",
                    ]
                    display(Markdown("## Working Summary\\n" + "\\n".join(summary_lines)))
                    """
                ),
            ],
        ),
        (
            "02_qc_and_host_burden.ipynb",
            [
                md(
                    """
                    # 02. QC And Host Burden

                    This notebook quantifies read-depth loss, host contamination, and the first batch-adjusted host-fraction regression used before mixed models.
                    """
                ),
                code(COMMON_SETUP),
                md(
                    """
                    ## Rebuild The QC Table And Simple Host Model

                    The QC outputs and first host-burden regression are generated here directly so the notebook shows the actual analysis path.
                    """
                ),
                code(
                    """
                    analysis_context = wc.base_analysis_context(context)
                    qc = base_data["qc"].copy()
                    host_fit, host_table = base.fit_host_model(qc)

                    qc_table = qc.reset_index().rename(columns={"index": "sample_id"})
                    wc.save_table(qc_table, wc.table_path(context, 2, "qc_metrics"))
                    wc.save_table(host_table, wc.table_path(context, 3, "host_model"))
                    """
                ),
                md(
                    """
                    ## Review Numbered Outputs
                    """
                ),
                code(
                    """
                    qc_table = pd.read_csv(wc.table_path(context, 2, "qc_metrics"), sep="\\t")
                    host_table = pd.read_csv(wc.table_path(context, 3, "host_model"), sep="\\t")

                    display(qc_table[
                        [
                            "sample_id",
                            "patient_id",
                            "visit_id",
                            "host_removed_fraction",
                            "bacterial_species_reads",
                            "metaphlan_species_reads",
                            "community_qc_pass",
                        ]
                    ].head(20))
                    display(host_table)

                    summary_lines = [
                        f"- Median host fraction: {qc_table['host_removed_fraction'].median():.1%}.",
                        f"- Community-analysis QC passing samples: {int(qc_table['community_qc_pass'].sum())} / {qc_table.shape[0]}.",
                        f"- Very low-depth samples (<10,000 bacterial species reads): {int((~qc_table['community_qc_pass']).sum())}.",
                        "- Positive result: the cluster-robust host model now adjusts for body site, chronicity, culture positivity, patient-relative elapsed time, and fixed batch-date effects.",
                        "- Negative result: this notebook still uses a Gaussian working model for host fraction and handles repeated structure only through patient-clustered standard errors.",
                    ]
                    display(Markdown("## Positive / Negative Takeaways\\n" + "\\n".join(summary_lines)))
                    """
                ),
            ],
        ),
        (
            "03_bracken_community_structure.ipynb",
            [
                md(
                    """
                    # 03. Bracken Community Structure

                    This notebook uses Bracken bacterial species counts for the main compositional analyses.
                    """
                ),
                code(COMMON_SETUP),
                md(
                    """
                    ## Build Pairwise Community Distances
                    """
                ),
                code(
                    """
                    analysis_context = wc.base_analysis_context(context)
                    metadata = base_data["metadata"]
                    qc = base_data["qc"]
                    species_bac = base_data["species_bac"]

                    community_samples = qc.index[qc["community_qc_pass"]].tolist()
                    rel_abundance = base.community_relative_abundance(species_bac, community_samples)
                    pairwise = base.summarize_pairwise_distances(rel_abundance, metadata)

                    from scipy.stats import mannwhitneyu
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    order = [
                        "Same patient, same batch date",
                        "Same patient, different batch date",
                        "Different patient",
                    ]
                    summary = (
                        pairwise.groupby("comparison_group")["distance"]
                        .agg(["count", "median", "mean"])
                        .reindex(order)
                        .reset_index()
                    )

                    pvalue_rows = []
                    reference = pairwise.loc[pairwise["comparison_group"] == "Different patient", "distance"]
                    for group in order[:-1]:
                        test = pairwise.loc[pairwise["comparison_group"] == group, "distance"]
                        statistic = mannwhitneyu(test, reference, alternative="less")
                        pvalue_rows.append(
                            {
                                "comparison_group": group,
                                "reference_group": "Different patient",
                                "pvalue": statistic.pvalue,
                                "median_difference": test.median() - reference.median(),
                            }
                        )
                    pvalues = base.bh_adjust(pd.DataFrame(pvalue_rows), "pvalue")

                    fig, ax = plt.subplots(figsize=(8.5, 5))
                    sns.boxplot(
                        data=pairwise,
                        x="comparison_group",
                        y="distance",
                        order=order,
                        color="#dfe7d6",
                        fliersize=0,
                        width=0.6,
                        ax=ax,
                    )
                    sns.stripplot(
                        data=pairwise.sample(min(pairwise.shape[0], 4000), random_state=7),
                        x="comparison_group",
                        y="distance",
                        order=order,
                        color="#3f4d3c",
                        alpha=0.25,
                        size=3,
                        ax=ax,
                    )
                    ax.set_xlabel("")
                    ax.set_ylabel("Bray-Curtis distance")
                    ax.set_title("Descriptive Bray-Curtis similarity by patient and batch-date grouping")
                    ax.set_xticks(range(len(order)), order, rotation=15, ha="right")
                    for idx, row in pvalues.iterrows():
                        ax.text(
                            idx,
                            0.98,
                            f"q={row['qvalue']:.3g}",
                            ha="center",
                            va="top",
                            fontsize=9,
                        )
                    fig.tight_layout()
                    fig_path = wc.figure_path(context, 2, "pairwise_distance")
                    fig.savefig(fig_path, bbox_inches="tight")
                    fig.savefig(fig_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)
                    plt.close(fig)
                    summary = summary.merge(pvalues, on="comparison_group", how="left")

                    wc.save_table(pairwise, wc.table_path(context, 4, "pairwise_distances"))
                    wc.save_table(summary, wc.table_path(context, 5, "pairwise_distance_summary"))
                    """
                ),
                md(
                    """
                    ## Review Numbered Outputs
                    """
                ),
                code(
                    """
                    pairwise = pd.read_csv(wc.table_path(context, 4, "pairwise_distances"), sep="\\t")
                    summary = pd.read_csv(wc.table_path(context, 5, "pairwise_distance_summary"), sep="\\t")

                    display(SVG(filename=str(wc.figure_path(context, 2, "pairwise_distance"))))
                    display(summary)

                    same_visit = summary.loc[summary["comparison_group"] == "Same patient, same batch date"].iloc[0]
                    revisit = summary.loc[summary["comparison_group"] == "Same patient, different batch date"].iloc[0]
                    different = summary.loc[summary["comparison_group"] == "Different patient"].iloc[0]
                    summary_lines = [
                        f"- Positive result: same-patient same-batch-date swabs were much closer than unrelated swabs (median Bray-Curtis {same_visit['median']:.3f} vs {different['median']:.3f}).",
                        f"- Negative result: same-patient different-batch-date swabs were nearly as dissimilar as unrelated swabs (median {revisit['median']:.3f}).",
                        "- Interpretation: descriptive patient-date grouping is informative, but later models treat absolute date as technical batch rather than biology.",
                    ]
                    display(Markdown("## Working Interpretation\\n" + "\\n".join(summary_lines)))
                    """
                ),
            ],
        ),
        (
            "04_culture_concordance.ipynb",
            [
                md(
                    """
                    # 04. Culture Concordance

                    This notebook compares culture calls against Bracken relative abundance for clinically important organism groups.
                    Descriptive inference is centered on Mann-Whitney U tests, with AUROC reported as a companion discrimination summary.
                    """
                ),
                code(COMMON_SETUP),
                md(
                    """
                    ## Build Culture-Concordance Summaries
                    """
                ),
                code(
                    """
                    analysis_context = wc.base_analysis_context(context)
                    qc = base_data["qc"]
                    species_bac = base_data["species_bac"]

                    culture_summary, culture_plot_df = base.make_culture_abundance_table(qc, species_bac)

                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    import numpy as np

                    top_labels = culture_summary.head(6)["label"].tolist()
                    culture_plot_df = culture_plot_df.loc[culture_plot_df["label"].isin(top_labels)].copy()
                    culture_plot_df["culture_status"] = np.where(culture_plot_df["culture_positive"], "Culture positive", "Culture negative")
                    culture_plot_df["log10_relative_abundance"] = np.log10(culture_plot_df["relative_abundance"] + 1e-6)

                    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
                    axes = axes.ravel()
                    for ax, label in zip(axes, top_labels):
                        data = culture_plot_df.loc[culture_plot_df["label"] == label]
                        sns.boxplot(
                            data=data,
                            x="culture_status",
                            y="log10_relative_abundance",
                            color="#f2d4b7",
                            width=0.6,
                            fliersize=0,
                            ax=ax,
                        )
                        sns.stripplot(
                            data=data,
                            x="culture_status",
                            y="log10_relative_abundance",
                            color="#5b3417",
                            alpha=0.5,
                            size=4,
                            ax=ax,
                        )
                        metrics = culture_summary.loc[culture_summary["label"] == label].iloc[0]
                        ax.set_title(f"{label}\\nU-test q={metrics['qvalue']:.3g}; AUROC={metrics['auroc']:.2f}")
                        ax.set_xlabel("")
                        ax.set_ylabel("log10(relative abundance + 1e-6)")
                        ax.tick_params(axis="x", rotation=20)
                    for ax in axes[len(top_labels):]:
                        ax.axis("off")
                    fig.tight_layout()
                    fig_path = wc.figure_path(context, 3, "culture_concordance")
                    fig.savefig(fig_path, bbox_inches="tight")
                    fig.savefig(fig_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)
                    plt.close(fig)

                    wc.save_table(culture_summary, wc.table_path(context, 6, "culture_concordance"))
                    """
                ),
                md(
                    """
                    ## Review Numbered Outputs
                    """
                ),
                code(
                    """
                    culture_summary = pd.read_csv(wc.table_path(context, 6, "culture_concordance"), sep="\\t")

                    display(SVG(filename=str(wc.figure_path(context, 3, "culture_concordance"))))
                    display(culture_summary)

                    top_hit = culture_summary.sort_values("qvalue").iloc[0]
                    weakest = culture_summary.sort_values("qvalue", ascending=False).iloc[0]
                    summary_lines = [
                        f"- Positive result: strongest descriptive agreement was {top_hit['label']} (Mann-Whitney U={top_hit['u_statistic']:.1f}, q={top_hit['qvalue']:.3g}, AUROC={top_hit['auroc']:.2f}).",
                        f"- Negative result: weakest supported target here was {weakest['label']} (Mann-Whitney q={weakest['qvalue']:.3g}, AUROC={weakest['auroc']:.2f}).",
                        "- Culture agreement is organism-group level, not strain-level or resistance-level agreement.",
                    ]
                    display(Markdown("## Working Interpretation\\n" + "\\n".join(summary_lines)))
                    """
                ),
            ],
        ),
        (
            "05_species_association_models.ipynb",
            [
                md(
                    """
                    # 05. Species Association Models

                    This notebook fits the first-pass cluster-robust species models using Bracken CLR-transformed bacterial species counts.
                    """
                ),
                code(COMMON_SETUP),
                md(
                    """
                    ## Fit Cluster-Robust Species Models
                    """
                ),
                code(
                    """
                    analysis_context = wc.base_analysis_context(context)
                    qc = base_data["qc"]
                    species_bac = base_data["species_bac"]

                    species_results = base.fit_species_models(qc, species_bac)

                    import matplotlib.pyplot as plt
                    import numpy as np

                    if not species_results.empty:
                        plot_df = species_results.loc[
                            species_results["term"].str.contains("body_region") | species_results["term"].str.contains("chronicity_group")
                        ].copy()
                        plot_df = plot_df.loc[plot_df["qvalue"].fillna(1) <= 0.15].copy()
                        if plot_df.empty:
                            plot_df = species_results.loc[
                                species_results["term"].str.contains("body_region") | species_results["term"].str.contains("chronicity_group")
                            ].head(12).copy()
                        plot_df["term_label"] = plot_df["term"].map(base.prettify_model_term)
                        plot_df["species_label"] = plot_df["species"]
                        plot_df = plot_df.sort_values(["estimate", "species_label"])
                        plot_df["y_label"] = plot_df["species_label"] + " | " + plot_df["term_label"]

                        fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * plot_df.shape[0])))
                        ax.axvline(0, color="black", linewidth=1, linestyle="--")
                        ax.errorbar(
                            x=plot_df["estimate"],
                            y=np.arange(plot_df.shape[0]),
                            xerr=[
                                plot_df["estimate"] - plot_df["conf_low"],
                                plot_df["conf_high"] - plot_df["estimate"],
                            ],
                            fmt="o",
                            color="#204a87",
                            ecolor="#7aa6d8",
                            capsize=3,
                        )
                        ax.set_yticks(np.arange(plot_df.shape[0]))
                        ax.set_yticklabels(plot_df["y_label"])
                        ax.set_xlabel("Cluster-robust CLR effect size")
                        ax.set_ylabel("")
                        ax.set_title("Body region and chronicity associations for key taxa")
                        fig.tight_layout()
                        fig_path = wc.figure_path(context, 4, "species_associations")
                        fig.savefig(fig_path, bbox_inches="tight")
                        fig.savefig(fig_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)
                        plt.close(fig)

                    wc.save_table(species_results, wc.table_path(context, 7, "species_associations"))
                    """
                ),
                md(
                    """
                    ## Review Numbered Outputs
                    """
                ),
                code(
                    """
                    species_results = pd.read_csv(wc.table_path(context, 7, "species_associations"), sep="\\t")

                    display(SVG(filename=str(wc.figure_path(context, 4, "species_associations"))))
                    display(species_results.head(20))

                    significant = species_results.loc[species_results["qvalue"] <= 0.1].copy()
                    summary_lines = [
                        f"- Positive result: {significant.shape[0]} model terms reached q <= 0.1 in the first-pass species screen.",
                        "- Positive result: the main supported patterns were chronic-like enrichment of P. aeruginosa and head/neck enrichment of S. aureus / Cutibacterium acnes.",
                        "- Negative result: weaker effects were sensitive to model choice and were treated as provisional until the mixed-model notebook.",
                    ]
                    display(Markdown("## Working Interpretation\\n" + "\\n".join(summary_lines)))
                    """
                ),
            ],
        ),
        (
            "06_mixed_models_and_repeated_measures.ipynb",
            [
                md(
                    """
                    # 06. Mixed Models And Repeated Measures

                    This notebook revisits the main host and taxon associations with patient and culture-date batch variance components.
                    It also records the single-group versus patient+batch diagnostics that motivated the updated specification.
                    """
                ),
                code(COMMON_SETUP),
                md(
                    """
                    ## Define The Patient-And-Batch Diagnostic Fits
                    """
                ),
                code(
                    """
                    import shutil
                    import warnings

                    import numpy as np
                    import statsmodels.formula.api as smf
                    from scipy.stats import chi2


                    def _best_mixed_fit(candidates):
                        converged = [item for item in candidates if bool(getattr(item[0], "converged", False))]
                        pool = converged if converged else candidates
                        return min(pool, key=lambda item: item[0].aic)


                    def fit_single_group_model(data, formula, outcome, structure_name, group_col):
                        candidates = []
                        last_error = ""
                        for method in ["lbfgs", "powell", "bfgs"]:
                            try:
                                with warnings.catch_warnings(record=True) as caught:
                                    warnings.simplefilter("always")
                                    fit = smf.mixedlm(
                                        formula=formula,
                                        data=data.copy(),
                                        groups=data[group_col],
                                    ).fit(reml=False, method=method, maxiter=500, disp=False)
                                candidates.append((fit, [str(item.message) for item in caught], method))
                            except Exception as exc:
                                last_error = repr(exc)
                        if not candidates:
                            return {
                                "outcome": outcome,
                                "structure": structure_name,
                                "status": "failed",
                                "converged": False,
                                "aic": np.nan,
                                "optimizer": "",
                                "patient_var": np.nan,
                                "batch_var": np.nan,
                                "warning_count": 0,
                                "warnings": "",
                                "error": last_error,
                            }

                        fit, warning_messages, optimizer = _best_mixed_fit(candidates)
                        variance_value = float(fit.params.get("Group Var", np.nan))
                        return {
                            "outcome": outcome,
                            "structure": structure_name,
                            "status": "ok",
                            "converged": bool(getattr(fit, "converged", False)),
                            "aic": float(fit.aic),
                            "optimizer": optimizer,
                            "patient_var": variance_value if structure_name == "patient_only" else np.nan,
                            "batch_var": variance_value if structure_name == "batch_only" else np.nan,
                            "warning_count": len(warning_messages),
                            "warnings": " | ".join(sorted(set(warning_messages)))[:2000],
                            "error": "",
                        }


                    def collect_optimizer_attempts(
                        data,
                        formula,
                        outcome,
                        structure_name,
                        group_col,
                        vc_formula=None,
                        constant_group=False,
                    ):
                        frame = data.copy()
                        if constant_group:
                            frame[group_col] = "all"

                        rows = []
                        candidates = []
                        for method in ["lbfgs", "powell", "bfgs"]:
                            try:
                                with warnings.catch_warnings(record=True) as caught:
                                    warnings.simplefilter("always")
                                    fit = smf.mixedlm(
                                        formula=formula,
                                        data=frame,
                                        groups=frame[group_col],
                                        vc_formula=vc_formula,
                                    ).fit(reml=False, method=method, maxiter=500, disp=False)
                                warning_messages = [str(item.message) for item in caught]
                                row = {
                                    "outcome": outcome,
                                    "model_name": advanced.HOST_MODEL_NAME,
                                    "structure": structure_name,
                                    "optimizer": method,
                                    "status": "ok",
                                    "converged": bool(getattr(fit, "converged", False)),
                                    "aic": float(fit.aic),
                                    "bic": float(fit.bic),
                                    "llf": float(fit.llf),
                                    "patient_var": float(fit.params.get("patient Var", np.nan)) if vc_formula is not None else (float(fit.params.get("Group Var", np.nan)) if structure_name == "patient_only" else np.nan),
                                    "batch_var": float(fit.params.get("batch Var", np.nan)) if vc_formula is not None else (float(fit.params.get("Group Var", np.nan)) if structure_name == "batch_only" else np.nan),
                                    "warning_count": len(warning_messages),
                                    "warnings": " | ".join(sorted(set(warning_messages)))[:2000],
                                    "error": "",
                                    "selected": False,
                                }
                                rows.append(row)
                                candidates.append((fit, warning_messages, method))
                            except Exception as exc:
                                rows.append(
                                    {
                                        "outcome": outcome,
                                        "model_name": advanced.HOST_MODEL_NAME,
                                        "structure": structure_name,
                                        "optimizer": method,
                                        "status": "failed",
                                        "converged": False,
                                        "aic": np.nan,
                                        "bic": np.nan,
                                        "llf": np.nan,
                                        "patient_var": np.nan,
                                        "batch_var": np.nan,
                                        "warning_count": 0,
                                        "warnings": "",
                                        "error": repr(exc),
                                        "selected": False,
                                    }
                                )

                        if candidates:
                            selected_fit, _, selected_optimizer = _best_mixed_fit(candidates)
                            selected_aic = float(selected_fit.aic)
                            for row in rows:
                                if (
                                    row["status"] == "ok"
                                    and row["optimizer"] == selected_optimizer
                                    and np.isfinite(row["aic"])
                                    and abs(float(row["aic"]) - selected_aic) < 1e-8
                                ):
                                    row["selected"] = True
                                    break
                        return rows


                    def fit_patient_structure_diagnostics(qc):
                        host_df = qc.dropna(
                            subset=[
                                "host_logit",
                                "body_region",
                                "patient_id",
                                "batch_id",
                                "chronicity_group",
                                "culture_positive_label",
                                "years_since_first_sample",
                            ]
                        ).copy()
                        host_formula = advanced.HOST_FORMULAS[advanced.HOST_MODEL_NAME]
                        rows = []
                        rows.extend(
                            collect_optimizer_attempts(
                                host_df,
                                host_formula,
                                "host_logit",
                                "patient_only",
                                "patient_id",
                            )
                        )
                        rows.extend(
                            collect_optimizer_attempts(
                                host_df,
                                host_formula,
                                "host_logit",
                                "batch_only",
                                "batch_id",
                            )
                        )
                        rows.extend(
                            collect_optimizer_attempts(
                                host_df,
                                host_formula,
                                "host_logit",
                                "patient_plus_batch",
                                "all_group",
                                vc_formula={"patient": "0 + C(patient_id)", "batch": "0 + C(batch_id)"},
                                constant_group=True,
                            )
                        )
                        return pd.DataFrame(rows)


                    def fit_patient_plus_batch_model(data, formula):
                        frame = data.copy()
                        frame["all_group"] = "all"
                        vc_formula = {"patient": "0 + C(patient_id)", "batch": "0 + C(batch_id)"}
                        candidates = []
                        last_error = ""
                        for method in ["lbfgs", "powell", "bfgs"]:
                            try:
                                with warnings.catch_warnings(record=True) as caught:
                                    warnings.simplefilter("always")
                                    fit = smf.mixedlm(
                                        formula=formula,
                                        data=frame,
                                        groups=frame["all_group"],
                                        vc_formula=vc_formula,
                                    ).fit(reml=False, method=method, maxiter=500, disp=False)
                                candidates.append((fit, [str(item.message) for item in caught], method))
                            except Exception as exc:
                                last_error = repr(exc)
                        if not candidates:
                            raise RuntimeError(last_error)
                        fit, warning_messages, optimizer = _best_mixed_fit(candidates)
                        status = {
                            "converged": bool(getattr(fit, "converged", False)),
                            "aic": float(fit.aic),
                            "bic": float(fit.bic),
                            "llf": float(fit.llf),
                            "patient_var": float(fit.params.get("patient Var", np.nan)),
                            "batch_var": float(fit.params.get("batch Var", np.nan)),
                            "optimizer": optimizer,
                            "warning_count": len(warning_messages),
                            "warnings": " | ".join(sorted(set(warning_messages)))[:2000],
                        }
                        return fit, status


                    def fit_best_single_group_model(data, formula, group_col):
                        candidates = []
                        last_error = ""
                        for method in ["lbfgs", "powell", "bfgs"]:
                            try:
                                with warnings.catch_warnings(record=True) as caught:
                                    warnings.simplefilter("always")
                                    fit = smf.mixedlm(
                                        formula=formula,
                                        data=data.copy(),
                                        groups=data[group_col],
                                    ).fit(reml=False, method=method, maxiter=500, disp=False)
                                candidates.append((fit, [str(item.message) for item in caught], method))
                            except Exception as exc:
                                last_error = repr(exc)
                        if not candidates:
                            raise RuntimeError(last_error)
                        fit, warning_messages, optimizer = _best_mixed_fit(candidates)
                        status = {
                            "converged": bool(getattr(fit, "converged", False)),
                            "aic": float(fit.aic),
                            "bic": float(fit.bic),
                            "llf": float(fit.llf),
                            "group_var": float(fit.params.get("Group Var", np.nan)),
                            "optimizer": optimizer,
                            "warning_count": len(warning_messages),
                            "warnings": " | ".join(sorted(set(warning_messages)))[:2000],
                        }
                        return fit, status


                    def build_random_effect_test_row(model_name, n_samples, tested_effect, full_model, reduced_model, full_llf, reduced_llf):
                        lrt_stat = max(0.0, 2.0 * (full_llf - reduced_llf))
                        p_chisq = chi2.sf(lrt_stat, 1)
                        return {
                            "outcome": "host_logit",
                            "model_name": model_name,
                            "n_samples": n_samples,
                            "converged": np.nan,
                            "aic": np.nan,
                            "bic": np.nan,
                            "llf": np.nan,
                            "patient_var": np.nan,
                            "batch_var": np.nan,
                            "optimizer": "",
                            "warning_count": np.nan,
                            "warnings": "",
                            "record_type": "random_effect_test",
                            "tested_effect": tested_effect,
                            "full_model": full_model,
                            "reduced_model": reduced_model,
                            "lrt_statistic": lrt_stat,
                            "df_diff": 1.0,
                            "pvalue_chisq": p_chisq,
                            "pvalue_boundary": 0.5 * p_chisq,
                        }


                    def fit_host_random_effect_tests(qc):
                        rows = []
                        for model_name, formula in advanced.HOST_FORMULAS.items():
                            needed = [
                                "host_logit",
                                "body_region",
                                "patient_id",
                                "batch_id",
                                "culture_positive_label",
                                "years_since_first_sample",
                            ]
                            if "chronicity_group" in formula:
                                needed.append("chronicity_group")
                            host_df = qc.dropna(subset=needed).copy()

                            fixed_fit = smf.ols(formula, data=host_df).fit()
                            patient_fit, _ = fit_best_single_group_model(host_df, formula, "patient_id")
                            batch_fit, _ = fit_best_single_group_model(host_df, formula, "batch_id")
                            full_fit, _ = fit_patient_plus_batch_model(host_df, formula)

                            rows.append(
                                build_random_effect_test_row(
                                    model_name,
                                    host_df.shape[0],
                                    "patient_only_vs_fixed",
                                    "patient_only",
                                    "fixed_only",
                                    patient_fit.llf,
                                    fixed_fit.llf,
                                )
                            )
                            rows.append(
                                build_random_effect_test_row(
                                    model_name,
                                    host_df.shape[0],
                                    "batch_only_vs_fixed",
                                    "batch_only",
                                    "fixed_only",
                                    batch_fit.llf,
                                    fixed_fit.llf,
                                )
                            )
                            rows.append(
                                build_random_effect_test_row(
                                    model_name,
                                    host_df.shape[0],
                                    "batch_added_to_patient",
                                    "patient_plus_batch",
                                    "patient_only",
                                    full_fit.llf,
                                    patient_fit.llf,
                                )
                            )
                            rows.append(
                                build_random_effect_test_row(
                                    model_name,
                                    host_df.shape[0],
                                    "patient_added_to_batch",
                                    "patient_plus_batch",
                                    "batch_only",
                                    full_fit.llf,
                                    batch_fit.llf,
                                )
                            )

                        return pd.DataFrame(rows)
                    """
                ),
                md(
                    """
                    ## Fit The Mixed Models And Save The Numbered Outputs
                    """
                ),
                code(
                    """
                    advanced_context = wc.advanced_analysis_context(context)
                    advanced.ensure_dirs(advanced_context)

                    qc = base_data["qc"].copy()
                    species_bac = base_data["species_bac"]

                    diagnostics = fit_patient_structure_diagnostics(qc)
                    host_cluster_fit, host_cluster = base.fit_host_model(qc)
                    species_cluster = base.fit_species_models(qc, species_bac)

                    host_effects, host_status = advanced.fit_host_models(qc)
                    host_status = host_status.assign(
                        record_type="model_fit",
                        tested_effect=np.nan,
                        full_model=np.nan,
                        reduced_model=np.nan,
                        lrt_statistic=np.nan,
                        df_diff=np.nan,
                        pvalue_chisq=np.nan,
                        pvalue_boundary=np.nan,
                    )
                    host_status = pd.concat([host_status, fit_host_random_effect_tests(qc)], ignore_index=True)
                    species_effects, species_status = advanced.fit_species_models(qc, species_bac)
                    comparison = advanced.build_comparison_table(
                        host_effects,
                        species_effects,
                        host_cluster.assign(outcome="host_logit"),
                        species_cluster,
                    )

                    advanced.make_host_comparison_figure(
                        host_effects,
                        host_cluster.assign(outcome="host_logit"),
                        advanced_context,
                    )
                    advanced.make_species_mixed_figure(species_effects, advanced_context)

                    host_source = context.figure_dir / "advanced_figure_01_host_compare.svg"
                    host_source_jpg = context.figure_dir / "advanced_figure_01_host_compare.jpg"
                    species_source = context.figure_dir / "advanced_figure_02_species_mixed.svg"
                    species_source_jpg = context.figure_dir / "advanced_figure_02_species_mixed.jpg"
                    host_dest = wc.figure_path(context, 5, "host_model_compare")
                    host_dest_jpg = host_dest.with_suffix(".jpg")
                    species_dest = wc.figure_path(context, 6, "species_mixed_effects")
                    species_dest_jpg = species_dest.with_suffix(".jpg")
                    if host_source.exists():
                        shutil.move(host_source, host_dest)
                    if host_source_jpg.exists():
                        shutil.move(host_source_jpg, host_dest_jpg)
                    if species_source.exists():
                        shutil.move(species_source, species_dest)
                    if species_source_jpg.exists():
                        shutil.move(species_source_jpg, species_dest_jpg)

                    wc.save_table(diagnostics, wc.table_path(context, 8, "patient_structure_diagnostics"))
                    wc.save_table(host_effects, wc.table_path(context, 9, "host_mixed_effects"))
                    wc.save_table(host_status, wc.table_path(context, 10, "host_mixed_status"))
                    wc.save_table(species_effects, wc.table_path(context, 11, "species_mixed_effects"))
                    wc.save_table(species_status, wc.table_path(context, 12, "species_mixed_status"))
                    wc.save_table(comparison, wc.table_path(context, 13, "mixed_vs_cluster_comparison"))

                    if advanced_context.input_dir.exists():
                        shutil.rmtree(advanced_context.input_dir)
                    """
                ),
                md(
                    """
                    ## Review Numbered Outputs
                    """
                ),
                code(
                    """
                    diagnostics = pd.read_csv(wc.table_path(context, 8, "patient_structure_diagnostics"), sep="\\t")
                    host_effects = pd.read_csv(wc.table_path(context, 9, "host_mixed_effects"), sep="\\t")
                    host_status = pd.read_csv(wc.table_path(context, 10, "host_mixed_status"), sep="\\t")
                    species_effects = pd.read_csv(wc.table_path(context, 11, "species_mixed_effects"), sep="\\t")
                    comparison = pd.read_csv(wc.table_path(context, 13, "mixed_vs_cluster_comparison"), sep="\\t")

                    display(diagnostics)
                    display(SVG(filename=str(wc.figure_path(context, 5, "host_model_compare"))))
                    display(SVG(filename=str(wc.figure_path(context, 6, "species_mixed_effects"))))
                    display(species_effects.head(20))

                    host_model = host_status.loc[
                        (host_status["record_type"] == "model_fit")
                        & (host_status["model_name"] == advanced.HOST_MODEL_NAME)
                    ].iloc[0]
                    host_model_tests = host_status.loc[
                        (host_status["record_type"] == "random_effect_test")
                        & (host_status["model_name"] == advanced.HOST_MODEL_NAME)
                    ].copy()
                    sign_agree = comparison["same_direction"].dropna().mean()
                    patient_test = host_model_tests.loc[
                        host_model_tests["tested_effect"] == "patient_only_vs_fixed"
                    ].iloc[0]
                    batch_test = host_model_tests.loc[
                        host_model_tests["tested_effect"] == "batch_only_vs_fixed"
                    ].iloc[0]
                    add_batch_test = host_model_tests.loc[
                        host_model_tests["tested_effect"] == "batch_added_to_patient"
                    ].iloc[0]
                    summary_lines = [
                        f"- Patient-plus-batch host model AIC: {host_model['aic']:.2f}; patient variance {host_model['patient_var']:.3g}, batch variance {host_model['batch_var']:.3g}.",
                        f"- Gaussian random-effect tests: patient vs fixed p={patient_test['pvalue_boundary']:.3g}, batch vs fixed p={batch_test['pvalue_boundary']:.3g}, add batch on top of patient p={add_batch_test['pvalue_boundary']:.3g}.",
                        f"- Positive result: sign agreement between simple and mixed models was {sign_agree:.1%} across matched terms.",
                        "- Positive result: P. aeruginosa chronic-like enrichment and S. aureus head/neck enrichment persisted in the mixed model.",
                        "- Negative result: host fixed effects remain modest once patient and culture-date batch structure are modeled explicitly.",
                        "- Negative result: single-group random-intercept fits were less informative than the joint patient-plus-batch specification.",
                    ]
                    display(Markdown("## Working Interpretation\\n" + "\\n".join(summary_lines)))
                    """
                ),
            ],
        ),
        (
            "07_bracken_vs_metaphlan_sensitivity.ipynb",
            [
                md(
                    """
                    # 07. Bracken Versus MetaPhlAn Sensitivity Analysis

                    This notebook asks whether the main community and species-model conclusions are robust to the choice of bacterial profiling table.
                    """
                ),
                code(COMMON_SETUP),
                md(
                    """
                    ## Define The Matched-Method Comparison Helpers
                    """
                ),
                code(
                    """
                    import matplotlib.pyplot as plt
                    import numpy as np
                    import seaborn as sns
                    import statsmodels.formula.api as smf
                    from scipy.stats import mannwhitneyu, spearmanr

                    BODY_REGION_COMPARISON_ORDER = [
                        "Same patient, same batch date",
                        "Same patient, different batch date",
                        "Different patient",
                    ]

                    SHARED_SENSITIVITY_SPECIES = [
                        "Staphylococcus aureus",
                        "Pseudomonas aeruginosa",
                        "Cutibacterium acnes",
                        "Corynebacterium striatum",
                        "Serratia marcescens",
                    ]


                    def summarize_pairwise_groups(pairwise):
                        summary = (
                            pairwise.groupby("comparison_group")["distance"]
                            .agg(["count", "median", "mean"])
                            .reindex(BODY_REGION_COMPARISON_ORDER)
                            .reset_index()
                        )
                        reference = pairwise.loc[pairwise["comparison_group"] == "Different patient", "distance"]
                        rows = []
                        for group in BODY_REGION_COMPARISON_ORDER[:-1]:
                            test = pairwise.loc[pairwise["comparison_group"] == group, "distance"]
                            if test.empty or reference.empty:
                                continue
                            statistic = mannwhitneyu(test, reference, alternative="less")
                            rows.append(
                                {
                                    "comparison_group": group,
                                    "reference_group": "Different patient",
                                    "pvalue": statistic.pvalue,
                                    "median_difference": test.median() - reference.median(),
                                }
                            )
                        pvalues = base.bh_adjust(pd.DataFrame(rows), "pvalue") if rows else pd.DataFrame()
                        return summary.merge(pvalues, on="comparison_group", how="left")


                    def fit_method_species_models(qc, counts, species_list):
                        sample_ids = qc.index.tolist()
                        counts = counts.loc[sample_ids].copy()
                        counts = counts.loc[:, counts.sum(axis=0) > 0]
                        clr = base.clr_transform(counts)
                        model_df = qc.loc[
                            sample_ids,
                            ["patient_id", "body_region", "chronicity_group", "log10_bacterial_reads"],
                        ].copy()
                        rows = []
                        for species in species_list:
                            if species not in clr.columns:
                                continue
                            prevalence = float((counts[species] > 0).mean())
                            if prevalence < 0.1:
                                continue
                            frame = model_df.copy()
                            frame["response"] = clr[species]
                            formula = (
                                "response ~ C(body_region, Treatment('lower_extremity')) "
                                "+ C(chronicity_group, Treatment('unknown')) + log10_bacterial_reads"
                            )
                            fit = smf.ols(formula, data=frame).fit(
                                cov_type="cluster",
                                cov_kwds={"groups": frame["patient_id"]},
                            )
                            conf = fit.conf_int()
                            for term, estimate in fit.params.items():
                                if term == "Intercept":
                                    continue
                                rows.append(
                                    {
                                        "species": species,
                                        "term": term,
                                        "estimate": estimate,
                                        "conf_low": conf.loc[term, 0],
                                        "conf_high": conf.loc[term, 1],
                                        "pvalue": fit.pvalues[term],
                                        "prevalence": prevalence,
                                    }
                                )
                        if not rows:
                            return pd.DataFrame(
                                columns=["species", "term", "estimate", "conf_low", "conf_high", "pvalue", "prevalence", "qvalue"]
                            )
                        return base.bh_adjust(pd.DataFrame(rows).sort_values("pvalue"), "pvalue")


                    def build_bracken_metaphlan_tables():
                        qc = base_data["qc"]
                        species_bac = base_data["species_bac"]
                        metaphlan_species = base_data["metaphlan_species"]

                        sample_depth = pd.DataFrame(
                            {
                                "sample_id": qc.index,
                                "patient_id": qc["patient_id"].values,
                                "visit_id": qc["visit_id"].values,
                                "host_removed_fraction": qc["host_removed_fraction"].values,
                                "bracken_bacterial_species_reads": qc["bacterial_species_reads"].values,
                                "metaphlan_species_reads": metaphlan_species.sum(axis=1).reindex(qc.index).fillna(0).values,
                                "bracken_qc_pass": qc["community_qc_pass"].values,
                            }
                        )
                        sample_depth["metaphlan_nonzero"] = sample_depth["metaphlan_species_reads"] > 0
                        sample_depth["shared_method_qc"] = sample_depth["bracken_qc_pass"] & sample_depth["metaphlan_nonzero"]

                        shared_samples = sample_depth.loc[sample_depth["shared_method_qc"], "sample_id"].tolist()
                        metadata_subset = base_data["metadata"].loc[shared_samples]

                        bracken_rel = species_bac.loc[shared_samples].copy()
                        bracken_rel = bracken_rel.loc[:, bracken_rel.sum(axis=0) > 0]
                        bracken_rel = bracken_rel.div(bracken_rel.sum(axis=1), axis=0)

                        metaphlan_rel = metaphlan_species.loc[shared_samples].copy()
                        metaphlan_rel = metaphlan_rel.loc[:, metaphlan_rel.sum(axis=0) > 0]
                        metaphlan_rel = metaphlan_rel.div(metaphlan_rel.sum(axis=1), axis=0)

                        bracken_pairwise = base.summarize_pairwise_distances(bracken_rel, metadata_subset)
                        metaphlan_pairwise = base.summarize_pairwise_distances(metaphlan_rel, metadata_subset)

                        bracken_summary = summarize_pairwise_groups(bracken_pairwise).assign(method="Bracken")
                        metaphlan_summary = summarize_pairwise_groups(metaphlan_pairwise).assign(method="MetaPhlAn")
                        distance_summary = pd.concat([bracken_summary, metaphlan_summary], ignore_index=True)

                        shared_species = [
                            species
                            for species in SHARED_SENSITIVITY_SPECIES
                            if species in bracken_rel.columns and species in metaphlan_rel.columns
                        ]
                        taxon_rows = []
                        for species in shared_species:
                            bracken_values = np.log10(bracken_rel[species] + 1e-6)
                            metaphlan_values = np.log10(metaphlan_rel[species] + 1e-6)
                            rho, pvalue = spearmanr(bracken_values, metaphlan_values)
                            taxon_rows.append(
                                {
                                    "species": species,
                                    "spearman_rho": rho,
                                    "pvalue": pvalue,
                                    "bracken_prevalence": float((bracken_rel[species] > 0).mean()),
                                    "metaphlan_prevalence": float((metaphlan_rel[species] > 0).mean()),
                                }
                            )
                        taxon_correlations = base.bh_adjust(pd.DataFrame(taxon_rows).sort_values("pvalue"), "pvalue")

                        qc_subset = qc.loc[shared_samples].copy()
                        bracken_models = fit_method_species_models(qc_subset, species_bac.loc[shared_samples], shared_species)
                        metaphlan_models = fit_method_species_models(qc_subset, metaphlan_species.loc[shared_samples], shared_species)

                        model_comparison = bracken_models.merge(
                            metaphlan_models,
                            on=["species", "term"],
                            suffixes=("_bracken", "_metaphlan"),
                            how="inner",
                        )
                        model_comparison["same_direction"] = np.sign(model_comparison["estimate_bracken"]) == np.sign(
                            model_comparison["estimate_metaphlan"]
                        )
                        model_comparison["term_label"] = model_comparison["term"].map(base.prettify_model_term)
                        return {
                            "sample_depth": sample_depth,
                            "distance_summary": distance_summary,
                            "taxon_correlations": taxon_correlations,
                            "bracken_models": bracken_models,
                            "metaphlan_models": metaphlan_models,
                            "model_comparison": model_comparison.sort_values(["species", "term"]),
                        }


                    def make_bracken_metaphlan_figure(comparison_data):
                        sample_depth = comparison_data["sample_depth"]
                        distance_summary = comparison_data["distance_summary"]
                        taxon_correlations = comparison_data["taxon_correlations"]
                        model_comparison = comparison_data["model_comparison"]

                        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                        ax = axes[0, 0]
                        plot_df = sample_depth.loc[sample_depth["shared_method_qc"]].copy()
                        sns.scatterplot(
                            data=plot_df,
                            x="bracken_bacterial_species_reads",
                            y="metaphlan_species_reads",
                            hue="host_removed_fraction",
                            palette="viridis",
                            s=65,
                            ax=ax,
                        )
                        ax.set_xscale("log")
                        ax.set_yscale("log")
                        ax.set_xlabel("Bracken bacterial species reads")
                        ax.set_ylabel("MetaPhlAn species reads")
                        ax.set_title("Sample-level bacterial depth agreement")

                        ax = axes[0, 1]
                        plot_summary = distance_summary.copy()
                        plot_summary["comparison_group"] = pd.Categorical(
                            plot_summary["comparison_group"],
                            categories=BODY_REGION_COMPARISON_ORDER,
                            ordered=True,
                        )
                        sns.pointplot(
                            data=plot_summary.sort_values("comparison_group"),
                            x="comparison_group",
                            y="median",
                            hue="method",
                            dodge=0.2,
                            markers=["o", "s"],
                            linestyles=["-", "--"],
                            ax=ax,
                        )
                        ax.set_xlabel("")
                        ax.set_ylabel("Median Bray-Curtis distance")
                        ax.set_title("Within-patient structure by method")
                        ax.tick_params(axis="x", rotation=15)

                        ax = axes[1, 0]
                        if not taxon_correlations.empty:
                            plot_corr = taxon_correlations.sort_values("spearman_rho")
                            ax.barh(plot_corr["species"], plot_corr["spearman_rho"], color="#3b6a8f")
                            ax.axvline(0, color="black", linestyle="--", linewidth=1)
                            ax.set_xlabel("Spearman rho")
                        else:
                            ax.text(0.5, 0.5, "No shared taxa available", ha="center", va="center")
                            ax.set_axis_off()
                        ax.set_title("Shared-taxon abundance concordance")

                        ax = axes[1, 1]
                        if not model_comparison.empty:
                            sns.scatterplot(
                                data=model_comparison,
                                x="estimate_bracken",
                                y="estimate_metaphlan",
                                hue="species",
                                s=75,
                                ax=ax,
                            )
                            low = np.nanmin(
                                [model_comparison["estimate_bracken"].min(), model_comparison["estimate_metaphlan"].min()]
                            ) - 0.2
                            high = np.nanmax(
                                [model_comparison["estimate_bracken"].max(), model_comparison["estimate_metaphlan"].max()]
                            ) + 0.2
                            ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
                            ax.set_xlim(low, high)
                            ax.set_ylim(low, high)
                            ax.set_xlabel("Bracken coefficient")
                            ax.set_ylabel("MetaPhlAn coefficient")
                        else:
                            ax.text(0.5, 0.5, "No matched model terms", ha="center", va="center")
                            ax.set_axis_off()
                        ax.set_title("Shared model-term comparison")

                        fig.tight_layout()
                        out_path = wc.figure_path(context, 7, "bracken_metaphlan_sensitivity")
                        fig.savefig(out_path, bbox_inches="tight")
                        plt.close(fig)
                    """
                ),
                md(
                    """
                    ## Run The Sensitivity Analysis And Save Numbered Outputs
                    """
                ),
                code(
                    """
                    comparison_data = build_bracken_metaphlan_tables()
                    make_bracken_metaphlan_figure(comparison_data)

                    wc.save_table(comparison_data["sample_depth"], wc.table_path(context, 14, "bracken_metaphlan_sample_depth"))
                    wc.save_table(comparison_data["distance_summary"], wc.table_path(context, 15, "bracken_metaphlan_distance_summary"))
                    wc.save_table(comparison_data["taxon_correlations"], wc.table_path(context, 16, "bracken_metaphlan_taxon_correlations"))
                    wc.save_table(comparison_data["bracken_models"], wc.table_path(context, 17, "bracken_species_models_shared_samples"))
                    wc.save_table(comparison_data["metaphlan_models"], wc.table_path(context, 18, "metaphlan_species_models"))
                    wc.save_table(comparison_data["model_comparison"], wc.table_path(context, 19, "bracken_metaphlan_model_comparison"))
                    """
                ),
                md(
                    """
                    ## Review Numbered Outputs
                    """
                ),
                code(
                    """
                    sample_depth = pd.read_csv(wc.table_path(context, 14, "bracken_metaphlan_sample_depth"), sep="\\t")
                    distance_summary = pd.read_csv(wc.table_path(context, 15, "bracken_metaphlan_distance_summary"), sep="\\t")
                    taxon_correlations = pd.read_csv(wc.table_path(context, 16, "bracken_metaphlan_taxon_correlations"), sep="\\t")
                    model_comparison = pd.read_csv(wc.table_path(context, 19, "bracken_metaphlan_model_comparison"), sep="\\t")

                    display(SVG(filename=str(wc.figure_path(context, 7, "bracken_metaphlan_sensitivity"))))
                    display(distance_summary)
                    display(taxon_correlations)
                    display(model_comparison.head(20))

                    shared_n = int(sample_depth["shared_method_qc"].sum())
                    same_direction = model_comparison["same_direction"].mean() if not model_comparison.empty else float("nan")
                    summary_lines = [
                        f"- Samples retained for direct method comparison: {shared_n}.",
                        "- Positive result: the major sample-structure pattern can be compared on a matched sample set rather than by memory or intuition.",
                        f"- Positive / negative result: matched Bracken-vs-MetaPhlAn coefficient directions agreed in {same_direction:.1%} of shared terms." if pd.notna(same_direction) else "- No matched model terms were available for direction comparison.",
                        "- Negative result: MetaPhlAn is sparser here, so some Bracken-supported signals are expected to attenuate rather than replicate perfectly.",
                    ]
                    display(Markdown("## Working Interpretation\\n" + "\\n".join(summary_lines)))
                    """
                ),
            ],
        ),
        (
            "08_halla_exploration.ipynb",
            [
                md(
                    """
                    # 08. HAllA Exploration

                    This notebook runs HAllA as an exploratory metadata-microbiome block-association screen.
                    It is intentionally separated from the main inferential notebooks because HAllA does not explicitly model repeated measures.
                    """
                ),
                code(COMMON_SETUP),
                md(
                    """
                    ## Define Notebook-Local HAllA Helpers

                    These helper functions are kept in this notebook because they are only used here.
                    """
                ),
                code(
                    """
                    import shutil

                    from rpy2.robjects import packages as rpackages


                    EMPTY_TOP_PAIRS = pd.DataFrame(
                        columns=["X_features", "Y_features", "association", "p-values", "q-values"]
                    )
                    EMPTY_SIG_CLUSTERS = pd.DataFrame(
                        columns=[
                            "cluster_rank",
                            "metadata_features",
                            "microbiome_features",
                            "best_adjusted_pvalue",
                            "metadata_size",
                            "microbiome_size",
                        ]
                    )


                    class _MissingRPackage:
                        def __init__(self, name):
                            self.name = name

                        def __getattr__(self, attr):
                            raise RuntimeError(
                                f'R package "{self.name}" is required for this HAllA code path but is not installed.'
                            )


                    def patch_optional_r_imports():
                        real_importr = rpackages.importr

                        def patched_importr(name, *args, **kwargs):
                            if name in {"XICOR", "eva"}:
                                return _MissingRPackage(name)
                            return real_importr(name, *args, **kwargs)

                        rpackages.importr = patched_importr


                    def import_halla_class():
                        try:
                            from halla import HAllA
                            return HAllA
                        except Exception as primary_exc:
                            fallback_site = Path("/home/ubuntu/miniforge3/lib/python3.12/site-packages")
                            if fallback_site.exists() and str(fallback_site) not in sys.path:
                                sys.path.append(str(fallback_site))
                            # Clear any failed or shadowed module entry before retrying import.
                            sys.modules.pop("halla", None)
                            try:
                                from halla import HAllA
                                return HAllA
                            except Exception as secondary_exc:
                                raise ImportError(
                                    "Unable to import HAllA from current env or base Miniforge site-packages. "
                                    f"primary={primary_exc!r}; fallback={secondary_exc!r}"
                                )


                    def prepare_halla_inputs():
                        input_dir = context.halla_dir / "inputs"
                        input_dir.mkdir(parents=True, exist_ok=True)

                        qc = base_data["qc"]
                        species_bac = base_data["species_bac"]

                        sample_ids = qc.index[qc["model_qc_pass"]].tolist()
                        rel_ab = species_bac.loc[sample_ids].copy()
                        rel_ab = rel_ab.div(rel_ab.sum(axis=1), axis=0).fillna(0)
                        prevalence = (rel_ab > 0).mean(axis=0)
                        rel_ab = rel_ab.loc[:, prevalence >= 0.1]

                        metadata = qc.loc[
                            sample_ids,
                            [
                                "host_removed_fraction",
                                "log10_bacterial_reads",
                                "culture_s_aureus",
                                "culture_p_aeruginosa",
                                "culture_serratia_marcescens",
                                "culture_proteus_mirabilis",
                                "culture_gas",
                                "culture_klebsiella_spp",
                                "culture_e_coli",
                                "culture_acinetobacter_baumannii",
                                "culture_e_faecalis",
                                "body_region",
                                "chronicity_group",
                                "clinical_infection_flag",
                            ],
                        ].copy()
                        metadata = pd.get_dummies(
                            metadata,
                            columns=["body_region", "chronicity_group", "clinical_infection_flag"],
                            dtype=float,
                        )
                        metadata = metadata.replace({True: 1.0, False: 0.0})
                        metadata = metadata.apply(pd.to_numeric, errors="raise").astype(float)

                        metadata_t = metadata.transpose()
                        metadata_t.index.name = "feature"
                        metadata_path = input_dir / "metadata_features_by_samples.tsv"
                        metadata_t.to_csv(metadata_path, sep="\\t")

                        microbiome_t = rel_ab.transpose()
                        microbiome_t.index.name = "feature"
                        microbiome_path = input_dir / "microbiome_features_by_samples.tsv"
                        microbiome_t.to_csv(microbiome_path, sep="\\t")
                        return metadata_path, microbiome_path


                    def summarize_halla_results(output_dir):
                        all_assoc = pd.read_csv(output_dir / "all_associations.txt", sep="\\t")
                        sig_clusters = pd.read_csv(output_dir / "sig_clusters.txt", sep="\\t")

                        top_pairs = (
                            all_assoc.assign(abs_association=lambda df: df["association"].abs())
                            .sort_values(["q-values", "abs_association"], ascending=[True, False])
                            .drop(columns=["abs_association"])
                            .head(100)
                            .reset_index(drop=True)
                        )

                        sig_clusters = sig_clusters.rename(
                            columns={
                                "cluster_X": "metadata_features",
                                "cluster_Y": "microbiome_features",
                                "best_adjusted_pvalue": "best_adjusted_pvalue",
                            }
                        )
                        sig_clusters["metadata_size"] = sig_clusters["metadata_features"].str.count(";").fillna(0).astype(int) + 1
                        sig_clusters["microbiome_size"] = sig_clusters["microbiome_features"].str.count(";").fillna(0).astype(int) + 1
                        return top_pairs, sig_clusters
                    """
                ),
                md(
                    """
                    ## Run HAllA And Save Numbered Outputs
                    """
                ),
                code(
                    """
                    patch_optional_r_imports()
                    metadata_path, microbiome_path = prepare_halla_inputs()
                    figure_file = wc.figure_path(context, 8, "halla_top25")
                    if figure_file.exists():
                        figure_file.unlink()

                    output_dir = context.halla_dir / "output"
                    if output_dir.exists():
                        shutil.rmtree(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    rows = []
                    top_pairs = EMPTY_TOP_PAIRS.copy()
                    sig_clusters = EMPTY_SIG_CLUSTERS.copy()

                    try:
                        HAllA = import_halla_class()

                        halla = HAllA(
                            pdist_metric="spearman",
                            out_dir=str(output_dir),
                            verbose=True,
                            num_threads=1,
                            force_permutations=False,
                        )
                        halla.load(str(metadata_path), str(microbiome_path))
                        halla.run()
                        halla.generate_hallagram(
                            block_num=25,
                            x_dataset_label="Clinical features",
                            y_dataset_label="Bacterial species",
                            output_file="hallagram_top25",
                            plot_type="svg",
                        )
                        top_pairs, sig_clusters = summarize_halla_results(output_dir)
                        rows.append(
                            {
                                "method": "HAllA",
                                "status": "ran_with_local_shim",
                                "runnable": True,
                                "reason": "Executed in spearman mode with optional R-package imports shimmed locally.",
                                "detail": "Optional HAllA startup imports for XICOR/eva were shimmed so the non-xicor, non-permutation workflow could run.",
                            }
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                "method": "HAllA",
                                "status": "failed",
                                "runnable": False,
                                "reason": "Execution failed in the local environment.",
                                "detail": repr(exc),
                            }
                        )

                    method_status = pd.DataFrame(rows)
                    hallagram = output_dir / "hallagram_top25.svg"
                    if hallagram.exists():
                        shutil.copy2(hallagram, figure_file)

                    wc.save_table(method_status, wc.table_path(context, 20, "halla_method_status"))
                    wc.save_table(top_pairs, wc.table_path(context, 21, "halla_top_pairwise_associations"))
                    wc.save_table(sig_clusters, wc.table_path(context, 22, "halla_significant_clusters"))
                    """
                ),
                md(
                    """
                    ## Review Numbered Outputs
                    """
                ),
                code(
                    """
                    method_status = pd.read_csv(wc.table_path(context, 20, "halla_method_status"), sep="\\t")
                    top_pairs = pd.read_csv(wc.table_path(context, 21, "halla_top_pairwise_associations"), sep="\\t")
                    sig_clusters = pd.read_csv(wc.table_path(context, 22, "halla_significant_clusters"), sep="\\t")

                    display(method_status)
                    figure_file = wc.figure_path(context, 8, "halla_top25")
                    if figure_file.exists():
                        display(SVG(filename=str(figure_file)))
                    display(top_pairs.head(20))
                    display(sig_clusters.head(20))

                    status = method_status.iloc[0]["status"]
                    summary_lines = [
                        f"- HAllA status: {status}.",
                        "- Positive result: HAllA provides an exploratory block-level view of metadata-microbiome associations.",
                        "- Negative result: it is not the primary inferential layer here because it does not model patient and culture-date batch structure the way the mixed models do.",
                    ]
                    display(Markdown("## Working Interpretation\\n" + "\\n".join(summary_lines)))
                    """
                ),
            ],
        ),
        (
            "09_maaslin2_multivariable.ipynb",
            [
                md(
                    """
                    # 09. MaAsLin2 Multivariable Associations

                    This notebook runs MaAsLin2 as an established multivariable microbiome association method.
                    The analysis is kept patient-aware through a patient random effect, but the feature space is trimmed to the
                    most prevalent taxa so the resulting table stays interpretable.
                    """
                ),
                code(R_COMMON_SETUP + "\nlibrary(Maaslin2)\n"),
                md(
                    """
                    ## Load QC-Passing Data And Prefilter The Feature Space

                    The MaAsLin2 run starts from the Bracken bacterial table on the model-QC-passing samples.
                    To avoid flooding the results with extremely sparse long-tail species, the notebook keeps taxa with at least
                    10% prevalence and then limits the model to the 200 most abundant among those.
                    """
                ),
                code(
                    """
                    qc <- read_tsv(table_file(2, "qc_metrics"), show_col_types = FALSE) |>
                      mutate(model_qc_pass = tolower(as.character(model_qc_pass)) %in% c("true", "t", "1"))

                    counts <- read_csv(file.path(data_dir, "read_count_species_bac.csv"), show_col_types = FALSE)
                    colnames(counts)[1] <- "sample_id"

                    model_samples <- qc |>
                      filter(model_qc_pass) |>
                      select(sample_id, patient_id, body_region, chronicity_group, clinical_infection_flag, log10_bacterial_reads)

                    counts <- counts |>
                      filter(sample_id %in% model_samples$sample_id)
                    counts <- counts[match(model_samples$sample_id, counts$sample_id), ]

                    count_matrix <- counts |>
                      tibble::column_to_rownames("sample_id") |>
                      as.data.frame(check.names = FALSE)

                    rel_abundance <- sweep(as.matrix(count_matrix), 1, rowSums(count_matrix), "/")
                    rel_abundance[is.na(rel_abundance)] <- 0

                    feature_summary <- tibble(
                      feature = colnames(count_matrix),
                      prevalence = colMeans(count_matrix > 0),
                      mean_relative_abundance = colMeans(rel_abundance)
                    ) |>
                      filter(prevalence >= 0.1) |>
                      arrange(desc(mean_relative_abundance))

                    selected_features <- head(feature_summary$feature, 200)
                    count_matrix <- as.data.frame(count_matrix[, selected_features, drop = FALSE], check.names = FALSE)
                    rownames(count_matrix) <- model_samples$sample_id

                    model_metadata <- model_samples |>
                      mutate(
                        patient_id = factor(sprintf("%02d", as.integer(patient_id))),
                        body_region = factor(body_region),
                        chronicity_group = factor(chronicity_group),
                        clinical_infection_flag = factor(clinical_infection_flag)
                      ) |>
                      tibble::column_to_rownames("sample_id") |>
                      as.data.frame()

                    maaslin_summary <- tibble(
                      n_samples = nrow(model_metadata),
                      n_prevalent_features = nrow(feature_summary),
                      n_features_tested = ncol(count_matrix),
                      fixed_effects = "body_region + chronicity_group + clinical_infection_flag + log10_bacterial_reads",
                      random_effects = "patient_id"
                    )

                    write_tsv(maaslin_summary, table_file(23, "maaslin2_model_summary"))

                    print(maaslin_summary)
                    print(feature_summary |> slice_head(n = 15))
                    """
                ),
                md(
                    """
                    ## Run MaAsLin2

                    The model uses total-sum scaling plus log transformation, fixed effects for body region, chronicity, infection flag,
                    and bacterial depth, and a patient random effect.
                    """
                ),
                code(
                    """
                    maaslin_dir <- file.path(root, "maaslin2")
                    unlink(maaslin_dir, recursive = TRUE)
                    dir.create(maaslin_dir, recursive = TRUE, showWarnings = FALSE)

                    fit <- Maaslin2(
                      input_data = count_matrix,
                      input_metadata = model_metadata,
                      output = maaslin_dir,
                      min_prevalence = 0.1,
                      normalization = "TSS",
                      transform = "LOG",
                      analysis_method = "LM",
                      max_significance = 0.1,
                      fixed_effects = c("body_region", "chronicity_group", "clinical_infection_flag", "log10_bacterial_reads"),
                      random_effects = c("patient_id"),
                      reference = c(
                        "body_region,lower_extremity",
                        "chronicity_group,unknown",
                        "clinical_infection_flag,unknown"
                      ),
                      plot_heatmap = FALSE,
                      plot_scatter = FALSE,
                      cores = 1
                    )

                    all_results <- read_tsv(file.path(maaslin_dir, "all_results.tsv"), show_col_types = FALSE) |>
                      mutate(
                        feature_label = str_squish(str_replace_all(feature, "\\\\.+", " ")),
                        term_label = if_else(is.na(value) | value == "", metadata, paste(metadata, value, sep = " = ")),
                        conf_low = coef - 1.96 * stderr,
                        conf_high = coef + 1.96 * stderr
                      ) |>
                      arrange(qval, pval)

                    focus_taxa <- c(
                      "Staphylococcus aureus",
                      "Pseudomonas aeruginosa",
                      "Cutibacterium acnes",
                      "Serratia marcescens",
                      "Klebsiella pneumoniae",
                      "Corynebacterium striatum",
                      "Escherichia coli",
                      "Proteus mirabilis",
                      "Acinetobacter baumannii",
                      "Enterococcus faecalis"
                    )

                    focus_results <- all_results |>
                      filter(feature_label %in% focus_taxa) |>
                      arrange(qval, pval)

                    write_tsv(focus_results, table_file(24, "maaslin2_focus_results"))

                    print(focus_results |> slice_head(n = 25))
                    """
                ),
                md(
                    """
                    ## Summarize Positive And Negative Results

                    The figure focuses on the clinically relevant taxa that overlap the earlier regression and mixed-model analyses.
                    """
                ),
                code(
                    """
                    plot_df <- focus_results |>
                      filter(term_label != "patient_id") |>
                      arrange(qval, desc(abs(coef))) |>
                      slice_head(n = 20) |>
                      mutate(
                        significant = qval <= 0.1,
                        plot_label = paste(feature_label, term_label, sep = " | ")
                      )

                    if (nrow(plot_df) == 0) {
                      plot_df <- all_results |>
                        arrange(qval, pval) |>
                        slice_head(n = 20) |>
                        mutate(
                          significant = qval <= 0.1,
                          plot_label = paste(feature_label, term_label, sep = " | ")
                        )
                    }

                    plot_df <- plot_df |>
                      mutate(plot_label = factor(plot_label, levels = rev(unique(plot_label))))

                    figure_09 <- ggplot(plot_df, aes(x = coef, y = plot_label, color = significant)) +
                      geom_vline(xintercept = 0, linewidth = 0.4, linetype = "dashed") +
                      geom_errorbarh(aes(xmin = conf_low, xmax = conf_high), height = 0.2, linewidth = 0.5) +
                      geom_point(size = 2.4) +
                      scale_color_manual(values = c("TRUE" = "#b44b2a", "FALSE" = "#4c6c8a")) +
                      labs(
                        title = "MaAsLin2 Focused Association Summary",
                        x = "Coefficient estimate",
                        y = NULL,
                        color = "q <= 0.1"
                      ) +
                      theme(legend.position = "top")

                    ggsave(
                      figure_file(9, "maaslin2_summary"),
                      figure_09,
                      width = 13,
                      height = 9,
                      device = grDevices::svg
                    )
                    print(figure_09)

                    maaslin_findings <- tibble(
                      finding = c(
                        sprintf("Positive result: %d of %d tested feature-term combinations reached q <= 0.1.", sum(all_results$qval <= 0.1, na.rm = TRUE), nrow(all_results)),
                        sprintf("Positive result: %d focus-taxon terms reached q <= 0.1.", sum(focus_results$qval <= 0.1, na.rm = TRUE)),
                        "Negative result: many significant hits fall in correlated long-tail species neighborhoods, so the focus figure is used to keep interpretation clinically grounded."
                      )
                    )

                    print(maaslin_findings)
                    """
                ),
            ],
        ),
        (
            "10_lefse_targeted_contrast.ipynb",
            [
                md(
                    """
                    # 10. LEfSe Targeted Contrast

                    This notebook uses LEfSe for a narrower exploratory contrast rather than a full multivariable analysis.
                    The chosen comparison is `head_neck` versus `lower_extremity`, which was one of the clearer site patterns in the earlier models.
                    """
                ),
                code(R_COMMON_SETUP),
                md(
                    """
                    ## Build A Targeted LEfSe Input Table

                    LEfSe is applied to a two-class body-region contrast on the model-QC-passing samples.
                    The input keeps taxa with at least 20% prevalence in this subset and then limits the analysis to the top 80 by mean relative abundance.
                    """
                ),
                code(
                    """
                    qc <- read_tsv(table_file(2, "qc_metrics"), show_col_types = FALSE) |>
                      mutate(model_qc_pass = tolower(as.character(model_qc_pass)) %in% c("true", "t", "1"))

                    counts <- read_csv(file.path(data_dir, "read_count_species_bac.csv"), show_col_types = FALSE)
                    colnames(counts)[1] <- "sample_id"

                    lefse_samples <- qc |>
                      filter(model_qc_pass, body_region %in% c("head_neck", "lower_extremity")) |>
                      select(sample_id, patient_id, body_region)

                    counts <- counts |>
                      filter(sample_id %in% lefse_samples$sample_id)
                    counts <- counts[match(lefse_samples$sample_id, counts$sample_id), ]

                    count_matrix <- counts |>
                      tibble::column_to_rownames("sample_id") |>
                      as.matrix()

                    rel_abundance <- sweep(count_matrix, 1, rowSums(count_matrix), "/")
                    rel_abundance[is.na(rel_abundance)] <- 0

                    feature_summary <- tibble(
                      feature = colnames(rel_abundance),
                      prevalence = colMeans(rel_abundance > 0),
                      mean_relative_abundance = colMeans(rel_abundance)
                    ) |>
                      filter(prevalence >= 0.2) |>
                      arrange(desc(mean_relative_abundance))

                    selected_features <- head(feature_summary$feature, 80)

                    lefse_input <- bind_cols(
                      lefse_samples |>
                        transmute(
                          class = body_region,
                          subclass = "all",
                          subject = sprintf("%02d", as.integer(patient_id))
                        ),
                      as_tibble(rel_abundance[, selected_features, drop = FALSE])
                    )

                    lefse_dir <- file.path(root, "lefse")
                    input_dir <- file.path(lefse_dir, "inputs")
                    output_dir <- file.path(lefse_dir, "output")
                    unlink(lefse_dir, recursive = TRUE)
                    dir.create(input_dir, recursive = TRUE, showWarnings = FALSE)
                    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

                    lefse_input_path <- file.path(input_dir, "lefse_body_region_columns.tsv")
                    write_tsv(lefse_input, lefse_input_path)

                    lefse_summary <- tibble(
                      comparison = "head_neck_vs_lower_extremity",
                      n_samples = nrow(lefse_samples),
                      n_head_neck = sum(lefse_samples$body_region == "head_neck"),
                      n_lower_extremity = sum(lefse_samples$body_region == "lower_extremity"),
                      n_prevalent_features = nrow(feature_summary),
                      n_features_tested = length(selected_features)
                    )

                    write_tsv(lefse_summary, table_file(25, "lefse_comparison_summary"))

                    print(lefse_summary)
                    print(feature_summary |> slice_head(n = 15))
                    """
                ),
                md(
                    """
                    ## Run LEfSe

                    The notebook calls the `lefse_format_input.py` and `lefse_run.py` executables from the active `eb` environment.
                    """
                ),
                code(
                    """
                    env_prefix <- dirname(dirname(R.home()))
                    env_bin <- file.path(env_prefix, "bin")
                    lefse_format_exe <- file.path(env_bin, "lefse_format_input.py")
                    lefse_run_exe <- file.path(env_bin, "lefse_run.py")

                    lefse_formatted_path <- file.path(output_dir, "lefse_body_region.in")
                    lefse_result_path <- file.path(output_dir, "lefse_body_region.res")

                    format_log <- system2(
                      lefse_format_exe,
                      args = c(lefse_input_path, lefse_formatted_path, "-f", "c", "-c", "1", "-s", "2", "-u", "3"),
                      stdout = TRUE,
                      stderr = TRUE
                    )
                    if (!is.null(attr(format_log, "status"))) {
                      stop(paste(format_log, collapse = "\\n"))
                    }

                    run_log <- system2(
                      lefse_run_exe,
                      args = c(lefse_formatted_path, lefse_result_path, "-l", "2.0"),
                      stdout = TRUE,
                      stderr = TRUE
                    )
                    if (!is.null(attr(run_log, "status"))) {
                      stop(paste(run_log, collapse = "\\n"))
                    }

                    raw_results <- read.delim(
                      lefse_result_path,
                      sep = "\\t",
                      header = FALSE,
                      fill = TRUE,
                      stringsAsFactors = FALSE
                    )
                    colnames(raw_results) <- c("feature", "log10_mean_abundance", "enriched_group", "lda_score", "wilcoxon_pvalue")

                    lefse_results <- raw_results |>
                      mutate(
                        feature_label = str_squish(str_replace_all(feature, "\\\\.+", " ")),
                        log10_mean_abundance = suppressWarnings(as.numeric(log10_mean_abundance)),
                        lda_score = suppressWarnings(as.numeric(lda_score)),
                        wilcoxon_pvalue = suppressWarnings(as.numeric(wilcoxon_pvalue))
                      ) |>
                      filter(feature_label != "subject")

                    significant_results <- lefse_results |>
                      filter(!is.na(lda_score), abs(lda_score) >= 2, !is.na(enriched_group), enriched_group != "-") |>
                      arrange(desc(abs(lda_score)))

                    write_tsv(significant_results, table_file(26, "lefse_significant_features"))

                    print(significant_results |> slice_head(n = 25))
                    """
                ),
                md(
                    """
                    ## Summarize Positive And Negative Results

                    LEfSe is intentionally treated as an exploratory targeted contrast. It does not replace the patient-aware multivariable models.
                    """
                ),
                code(
                    """
                    if (nrow(significant_results) > 0) {
                      plot_df <- significant_results |>
                        slice_head(n = 20) |>
                        mutate(feature_label = factor(feature_label, levels = rev(unique(feature_label))))

                      figure_10 <- ggplot(plot_df, aes(x = lda_score, y = feature_label, fill = enriched_group)) +
                        geom_col(width = 0.75) +
                        labs(
                          title = "LEfSe Head/Neck Versus Lower Extremity",
                          x = "LDA score",
                          y = NULL,
                          fill = "Enriched in"
                        )
                    } else {
                      figure_10 <- ggplot() +
                        annotate("text", x = 0, y = 0, label = "No features exceeded the LDA threshold in this targeted contrast.", size = 5) +
                        xlim(-1, 1) +
                        ylim(-1, 1) +
                        theme_void() +
                        labs(title = "LEfSe Head/Neck Versus Lower Extremity")
                    }

                    ggsave(
                      figure_file(10, "lefse_summary"),
                      figure_10,
                      width = 11,
                      height = 8,
                      device = grDevices::svg
                    )
                    print(figure_10)

                    lefse_findings <- tibble(
                      finding = c(
                        sprintf("Positive result: %d features exceeded the LEfSe LDA threshold in the targeted site comparison.", nrow(significant_results)),
                        "Negative result: LEfSe is being used on a two-class targeted contrast here because the broader multiclass setup was too brittle and returned an all-null result.",
                        "Negative result: LEfSe does not model the full repeated-measures multivariable structure, so its output stays exploratory."
                      )
                    )

                    print(lefse_findings)
                    """
                ),
            ],
        ),
        (
            "11_host_fraction_beta_binomial.ipynb",
            [
                md(
                    """
                    # 11. Host Fraction Beta-Binomial Mixed Model

                    This notebook upgrades the earlier host-fraction regression to a count-based mixed model.
                    The response is modeled as Bracken human species reads out of Bracken root reads
                    from the pre-host-filter reports,
                    so library-size scaling is handled through the binomial denominator rather than added again as a separate nuisance covariate.
                    Absolute culture date is treated as technical batch, while patient-relative elapsed time is treated as biology.
                    """
                ),
                code(
                    R_COMMON_SETUP
                    + """
                    suppressPackageStartupMessages({
                      library(glmmTMB)
                      library(broom.mixed)
                    })
                    """
                ),
                md(
                    """
                    ## Load QC Data And Define The Host Model Inputs

                    The model keeps a limited fixed-effect set to avoid overadjustment:
                    body site, chronicity, broad culture positivity, and patient-relative elapsed time.
                    Patient and culture-date batch are handled as random intercepts.
                    """
                ),
                code(
                    """
                    qc <- read_tsv(table_file(2, "qc_metrics"), show_col_types = FALSE) |>
                      mutate(
                        culture_date = as.Date(culture_date),
                        human_reads = pmax(human_species_reads, 0),
                        non_human_reads = pmax(bracken_total_reads - human_species_reads, 0),
                        culture_positive = factor(if_else(as.logical(culture_positive), "positive", "negative"),
                                                  levels = c("negative", "positive")),
                        patient_id = factor(sprintf("%02d", as.integer(patient_id))),
                        batch_id = factor(batch_id),
                        body_region = factor(body_region, levels = c("lower_extremity", "head_neck", "upper_extremity", "trunk_perineum", "unknown")),
                        chronicity_group = factor(chronicity_group, levels = c("unknown", "acute_like", "chronic_like", "mixed")),
                        upper_extremity_binary = factor(if_else(body_region == "upper_extremity", "upper_extremity", "other"),
                                                        levels = c("other", "upper_extremity")),
                        acute_like_binary = factor(if_else(chronicity_group == "acute_like", "acute_like", "other"),
                                                   levels = c("other", "acute_like"))
                      ) |>
                      filter(
                        !is.na(human_reads),
                        !is.na(non_human_reads),
                        bracken_total_reads > 0,
                        !is.na(body_region),
                        !is.na(chronicity_group),
                        !is.na(culture_positive),
                        !is.na(years_since_first_sample),
                        !is.na(batch_id)
                      )

                    host_model_formula <- "cbind(human_reads, non_human_reads) ~ body_region + chronicity_group + culture_positive + years_since_first_sample + (1 | patient_id) + (1 | batch_id)"

                    host_input_summary <- tibble(
                      n_samples = nrow(qc),
                      n_patients = n_distinct(qc$patient_id),
                      n_batches = n_distinct(qc$batch_id),
                      response = "cbind(human_reads, non_human_reads)",
                      fixed_effects = "body_region + chronicity_group + culture_positive + years_since_first_sample",
                      random_effects = "(1 | patient_id) + (1 | batch_id)",
                      family = "betabinomial(link = 'logit')"
                    )

                    print(host_input_summary)
                    """
                ),
                md(
                    """
                    ## Fit The Beta-Binomial Mixed Model
                    """
                ),
                code(
                    """
                    extract_re_var <- function(fit, group_name) {
                      vc <- VarCorr(fit)$cond[[group_name]]
                      if (is.null(vc)) {
                        return(NA_real_)
                      }
                      stddev <- attr(vc, "stddev")
                      if (is.null(stddev) || length(stddev) == 0) {
                        return(NA_real_)
                      }
                      as.numeric(stddev[1]^2)
                    }

                    fit_host_candidate <- function(model_name, formula_text) {
                      fit <- tryCatch(
                        glmmTMB(
                          formula = as.formula(formula_text),
                          data = qc,
                          family = betabinomial(link = "logit"),
                          control = glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4))
                        ),
                        error = function(e) e
                      )
                      if (inherits(fit, "error")) {
                        return(list(
                          fit = NULL,
                          status = tibble(
                            model = model_name,
                            formula = formula_text,
                            status = "failed",
                            n_samples = nrow(qc),
                            aic = NA_real_,
                            bic = NA_real_,
                            logLik = NA_real_,
                            patient_var = NA_real_,
                            batch_var = NA_real_,
                            pd_hessian = FALSE,
                            warning = as.character(fit$message)
                          )
                        ))
                      }

                      status <- tibble(
                        model = model_name,
                        formula = formula_text,
                        status = "ok",
                        n_samples = nrow(qc),
                        aic = AIC(fit),
                        bic = BIC(fit),
                        logLik = as.numeric(logLik(fit)),
                        patient_var = extract_re_var(fit, "patient_id"),
                        batch_var = extract_re_var(fit, "batch_id"),
                        pd_hessian = isTRUE(fit$sdr$pdHess),
                        warning = if_else(isTRUE(fit$sdr$pdHess), "", "non-positive-definite Hessian or optimizer warning")
                      )
                      list(fit = fit, status = status)
                    }

                    full_fixed_formula <- "cbind(human_reads, non_human_reads) ~ body_region + chronicity_group + culture_positive + years_since_first_sample"
                    full_random_formula <- paste(full_fixed_formula, "+ (1 | patient_id) + (1 | batch_id)")
                    patient_only_formula <- paste(full_fixed_formula, "+ (1 | patient_id)")
                    batch_only_formula <- paste(full_fixed_formula, "+ (1 | batch_id)")
                    no_body_region_formula <- "cbind(human_reads, non_human_reads) ~ chronicity_group + culture_positive + years_since_first_sample + (1 | patient_id) + (1 | batch_id)"
                    no_chronicity_formula <- "cbind(human_reads, non_human_reads) ~ body_region + culture_positive + years_since_first_sample + (1 | patient_id) + (1 | batch_id)"

                    candidate_results <- list(
                      fit_host_candidate("fixed_only", full_fixed_formula),
                      fit_host_candidate("patient_plus_batch", full_random_formula),
                      fit_host_candidate("batch_only", batch_only_formula),
                      fit_host_candidate("patient_only", patient_only_formula)
                    )

                    host_status <- bind_rows(lapply(candidate_results, `[[`, "status"))
                    reported_model_name <- "patient_plus_batch"
                    if (!reported_model_name %in% host_status$model || host_status$status[match(reported_model_name, host_status$model)] != "ok") {
                      ok_status <- host_status |>
                        filter(status == "ok")
                      preferred_status <- ok_status |>
                        filter(pd_hessian) |>
                        arrange(aic)
                      if (nrow(preferred_status) == 0) {
                        preferred_status <- ok_status |> arrange(aic)
                      }
                      reported_model_name <- preferred_status$model[1]
                    }
                    host_model <- candidate_results[[match(reported_model_name, host_status$model)]]$fit

                    reduced_results <- list(
                      fit_host_candidate("no_body_region", no_body_region_formula),
                      fit_host_candidate("no_chronicity", no_chronicity_formula)
                    )

                    contrast_specs <- tribble(
                      ~model_name, ~formula_text, ~target_term, ~term_label,
                      "upper_extremity_contrast",
                      "cbind(human_reads, non_human_reads) ~ upper_extremity_binary + chronicity_group + culture_positive + years_since_first_sample + (1 | patient_id) + (1 | batch_id)",
                      "upper_extremity_binaryupper_extremity",
                      "Planned contrast: upper extremity vs all other body regions",
                      "acute_like_contrast",
                      "cbind(human_reads, non_human_reads) ~ body_region + acute_like_binary + culture_positive + years_since_first_sample + (1 | patient_id) + (1 | batch_id)",
                      "acute_like_binaryacute_like",
                      "Planned contrast: acute-like vs all other chronicity groups"
                    )

                    extract_loglik_df <- function(fit) {
                      ll <- logLik(fit)
                      list(logLik = as.numeric(ll), df = attr(ll, "df"))
                    }

                    get_fit_from_results <- function(results, model_name) {
                      model_names <- vapply(results, function(x) x$status$model[[1]], character(1))
                      idx <- which(model_names == model_name)
                      if (length(idx) == 0) {
                        return(NULL)
                      }
                      results[[idx[1]]]$fit
                    }

                    compare_models <- function(full_fit, reduced_fit, effect_label, full_name, reduced_name, record_type = "random_effect_test") {
                      if (is.null(full_fit) || is.null(reduced_fit)) {
                        return(tibble(
                          model = paste0(full_name, "_vs_", reduced_name),
                          formula = NA_character_,
                          status = "comparison_failed",
                          n_samples = nrow(qc),
                          aic = NA_real_,
                          bic = NA_real_,
                          logLik = NA_real_,
                          patient_var = NA_real_,
                          batch_var = NA_real_,
                          pd_hessian = NA,
                          warning = paste("Comparison failed for", effect_label),
                          record_type = record_type,
                          tested_effect = effect_label,
                          full_model = full_name,
                          reduced_model = reduced_name,
                          lrt_statistic = NA_real_,
                          df_diff = NA_real_,
                          pvalue_chisq = NA_real_,
                          pvalue_boundary = NA_real_
                        ))
                      }
                      full_info <- extract_loglik_df(full_fit)
                      reduced_info <- extract_loglik_df(reduced_fit)
                      lrt_stat <- max(0, 2 * (full_info$logLik - reduced_info$logLik))
                      df_diff <- full_info$df - reduced_info$df
                      p_chisq <- pchisq(lrt_stat, df = df_diff, lower.tail = FALSE)
                      p_boundary <- ifelse(df_diff == 1, 0.5 * p_chisq, NA_real_)
                      tibble(
                        model = paste0(full_name, "_vs_", reduced_name),
                        formula = NA_character_,
                        status = "ok",
                        n_samples = nrow(qc),
                        aic = NA_real_,
                        bic = NA_real_,
                        logLik = NA_real_,
                        patient_var = NA_real_,
                        batch_var = NA_real_,
                        pd_hessian = NA,
                        warning = "",
                        record_type = record_type,
                        tested_effect = effect_label,
                        full_model = full_name,
                        reduced_model = reduced_name,
                        lrt_statistic = lrt_stat,
                        df_diff = df_diff,
                        pvalue_chisq = p_chisq,
                        pvalue_boundary = p_boundary
                      )
                    }

                    fixed_only_fit <- get_fit_from_results(candidate_results, "fixed_only")
                    patient_fit <- get_fit_from_results(candidate_results, "patient_only")
                    batch_fit <- get_fit_from_results(candidate_results, "batch_only")
                    patient_plus_batch_fit <- get_fit_from_results(candidate_results, "patient_plus_batch")
                    no_body_region_fit <- get_fit_from_results(reduced_results, "no_body_region")
                    no_chronicity_fit <- get_fit_from_results(reduced_results, "no_chronicity")

                    re_tests <- bind_rows(
                      compare_models(patient_fit, fixed_only_fit, "patient_only_vs_fixed", "patient_only", "fixed_only"),
                      compare_models(batch_fit, fixed_only_fit, "batch_only_vs_fixed", "batch_only", "fixed_only"),
                      compare_models(patient_plus_batch_fit, patient_fit, "batch_added_to_patient", "patient_plus_batch", "patient_only"),
                      compare_models(patient_plus_batch_fit, batch_fit, "patient_added_to_batch", "patient_plus_batch", "batch_only")
                    )

                    fixed_effect_tests <- bind_rows(
                      compare_models(patient_plus_batch_fit, no_body_region_fit, "body_region_overall", "patient_plus_batch", "no_body_region", "fixed_effect_test"),
                      compare_models(patient_plus_batch_fit, no_chronicity_fit, "chronicity_group_overall", "patient_plus_batch", "no_chronicity", "fixed_effect_test")
                    )

                    fit_contrast_model <- function(model_name, formula_text, target_term, term_label) {
                      result <- fit_host_candidate(model_name, formula_text)
                      status <- result$status |>
                        mutate(
                          record_type = "contrast_model_fit",
                          tested_effect = NA_character_,
                          full_model = NA_character_,
                          reduced_model = NA_character_,
                          lrt_statistic = NA_real_,
                          df_diff = NA_real_,
                          pvalue_chisq = NA_real_,
                          pvalue_boundary = NA_real_
                        )
                      effect <- tibble(
                        model_name = model_name,
                        analysis_type = "targeted_contrast",
                        adjustment_family = "planned_contrast_terms",
                        term = target_term,
                        term_label = term_label,
                        estimate = NA_real_,
                        std.error = NA_real_,
                        conf.low = NA_real_,
                        conf.high = NA_real_,
                        odds_ratio = NA_real_,
                        conf.low.or = NA_real_,
                        conf.high.or = NA_real_,
                        p.value = NA_real_
                      )
                      if (!is.null(result$fit)) {
                        tidy_fit <- tidy(result$fit, effects = "fixed", component = "cond")
                        target_row <- tidy_fit |> filter(term == target_term)
                        if (nrow(target_row) == 1) {
                          effect <- target_row |>
                            transmute(
                              model_name = model_name,
                              analysis_type = "targeted_contrast",
                              adjustment_family = "planned_contrast_terms",
                              term = term,
                              term_label = term_label,
                              estimate = estimate,
                              std.error = std.error,
                              conf.low = estimate - 1.96 * std.error,
                              conf.high = estimate + 1.96 * std.error,
                              odds_ratio = exp(estimate),
                              conf.low.or = exp(conf.low),
                              conf.high.or = exp(conf.high),
                              p.value = p.value
                            )
                        }
                      }
                      list(status = status, effect = effect)
                    }

                    contrast_results <- lapply(seq_len(nrow(contrast_specs)), function(i) {
                      fit_contrast_model(
                        contrast_specs$model_name[[i]],
                        contrast_specs$formula_text[[i]],
                        contrast_specs$target_term[[i]],
                        contrast_specs$term_label[[i]]
                      )
                    })

                    contrast_status <- bind_rows(lapply(contrast_results, `[[`, "status"))
                    contrast_effects <- bind_rows(lapply(contrast_results, `[[`, "effect")) |>
                      mutate(
                        qvalue = p.adjust(p.value, method = "BH"),
                        posthoc_family = "planned_contrast_terms",
                        posthoc_qvalue = qvalue
                      )

                    host_status <- host_status |>
                      mutate(
                        record_type = "model_fit",
                        tested_effect = NA_character_,
                        full_model = NA_character_,
                        reduced_model = NA_character_,
                        lrt_statistic = NA_real_,
                        df_diff = NA_real_,
                        pvalue_chisq = NA_real_,
                        pvalue_boundary = NA_real_
                      ) |>
                      bind_rows(
                        bind_rows(lapply(reduced_results, `[[`, "status")) |>
                          mutate(
                            record_type = "fixed_effect_reduced_model_fit",
                            tested_effect = NA_character_,
                            full_model = NA_character_,
                            reduced_model = NA_character_,
                            lrt_statistic = NA_real_,
                            df_diff = NA_real_,
                            pvalue_chisq = NA_real_,
                            pvalue_boundary = NA_real_
                          ),
                        contrast_status,
                        re_tests,
                        fixed_effect_tests
                      )

                    host_effects <- tidy(host_model, effects = "fixed", component = "cond") |>
                      filter(term != "(Intercept)") |>
                      mutate(
                        model_name = reported_model_name,
                        analysis_type = "full_model",
                        adjustment_family = "full_model_terms",
                        conf.low = estimate - 1.96 * std.error,
                        conf.high = estimate + 1.96 * std.error,
                        odds_ratio = exp(estimate),
                        conf.low.or = exp(conf.low),
                        conf.high.or = exp(conf.high),
                        qvalue = p.adjust(p.value, method = "BH"),
                        selected_model = reported_model_name,
                        posthoc_family = case_when(
                          str_detect(term, "^body_region") ~ "body_region_terms",
                          str_detect(term, "^chronicity_group") ~ "chronicity_terms",
                          term == "culture_positivepositive" ~ "culture_positive_term",
                          term == "years_since_first_sample" ~ "elapsed_time_term",
                          TRUE ~ "other_terms"
                        ),
                        term_label = case_when(
                          term == "years_since_first_sample" ~ "Per year since first patient sample",
                          term == "culture_positivepositive" ~ "Culture positive: yes",
                          term == "body_regionhead_neck" ~ "Body site: head / neck",
                          term == "body_regionupper_extremity" ~ "Body site: upper extremity",
                          term == "body_regiontrunk_perineum" ~ "Body site: trunk / perineum",
                          term == "body_regionunknown" ~ "Body site: unknown",
                          term == "chronicity_groupacute_like" ~ "Chronicity: acute-like",
                          term == "chronicity_groupchronic_like" ~ "Chronicity: chronic-like",
                          term == "chronicity_groupmixed" ~ "Chronicity: mixed",
                          TRUE ~ term
                        )
                      ) |>
                      group_by(posthoc_family) |>
                      mutate(posthoc_qvalue = p.adjust(p.value, method = "BH")) |>
                      ungroup() |>
                      arrange(qvalue, p.value)

                    host_effects <- bind_rows(
                      host_effects,
                      contrast_effects
                    ) |>
                      arrange(adjustment_family, qvalue, p.value)

                    write_tsv(host_effects, table_file(27, "host_beta_binomial_effects"))
                    write_tsv(host_status, table_file(28, "host_beta_binomial_status"))

                    print(host_effects)
                    print(host_status)
                    """
                ),
                md(
                    """
                    ## Summarize Positive And Negative Results
                    """
                ),
                code(
                    """
                    plot_df <- host_effects |>
                      filter(!is.na(odds_ratio)) |>
                      mutate(
                        panel = if_else(analysis_type == "full_model", "Full patient-plus-batch model", "Planned 1-df contrast models"),
                        term_label = factor(term_label, levels = rev(unique(term_label))),
                        display_qvalue = if_else(is.na(posthoc_qvalue), qvalue, posthoc_qvalue),
                        significant = display_qvalue <= 0.1
                      )

                    figure_11 <- ggplot(plot_df, aes(x = odds_ratio, y = term_label, color = significant)) +
                      geom_vline(xintercept = 1, linewidth = 0.5, linetype = "dashed", color = "grey50") +
                      geom_errorbarh(aes(xmin = conf.low.or, xmax = conf.high.or), height = 0.18, linewidth = 0.7) +
                      geom_point(size = 2.6) +
                      scale_x_log10() +
                      scale_color_manual(values = c("TRUE" = "#b22222", "FALSE" = "#3b6a8f")) +
                      facet_grid(panel ~ ., scales = "free_y", space = "free_y") +
                      labs(
                        title = "Host fraction beta-binomial model and planned host contrasts",
                        x = "Odds ratio for host fraction",
                        y = NULL,
                        color = "Targeted q <= 0.1"
                      ) +
                      theme(legend.position = "top")

                    ggsave(
                      figure_file(11, "host_beta_binomial"),
                      figure_11,
                      width = 11,
                      height = 7.5,
                      device = grDevices::svg
                    )
                    print(figure_11)

                    body_region_test <- host_status |>
                      filter(record_type == "fixed_effect_test", tested_effect == "body_region_overall") |>
                      slice(1)
                    chronicity_test <- host_status |>
                      filter(record_type == "fixed_effect_test", tested_effect == "chronicity_group_overall") |>
                      slice(1)
                    planned_hits <- host_effects |>
                      filter(analysis_type == "targeted_contrast") |>
                      arrange(qvalue, p.value)
                    posthoc_hits <- host_effects |>
                      filter(analysis_type == "full_model", posthoc_family %in% c("body_region_terms", "chronicity_terms")) |>
                      arrange(posthoc_qvalue, p.value)

                    host_findings <- tibble(
                      finding = c(
                        sprintf("Positive result: %d host-model terms reached q <= 0.1.", sum(host_effects$qvalue <= 0.1, na.rm = TRUE)),
                        sprintf("Positive result: the reported host model now conditions on body site, chronicity, culture positivity, and patient-relative elapsed time at the same time; the reported random-effect structure was %s.", reported_model_name),
                        sprintf("Positive result: omnibus fixed-effect tests gave p=%.3g for body_region and p=%.3g for chronicity_group in the full patient-plus-batch model.", body_region_test$pvalue_chisq[[1]], chronicity_test$pvalue_chisq[[1]]),
                        sprintf("Positive result: factor-specific post hoc BH gave q=%.3g for %s.", posthoc_hits$posthoc_qvalue[[1]], posthoc_hits$term_label[[1]]),
                        sprintf("Positive result: the strongest planned contrast was %s with OR %.2f and q=%.3g.", planned_hits$term_label[[1]], planned_hits$odds_ratio[[1]], planned_hits$qvalue[[1]]),
                        "Positive result: random-effect contribution is now summarized with explicit likelihood-ratio comparisons against reduced models.",
                        "Negative result: total sequencing depth is not added as a separate covariate here because the beta-binomial denominator already uses trimmed read pairs.",
                        "Negative result: absolute date is not interpreted biologically here; it is absorbed as a technical batch random effect instead.",
                        "Negative result: this still does not include every culture subgroup simultaneously because that would overparameterize the host model for this cohort."
                      )
                    )

                    print(host_findings)
                    """
                ),
            ],
        ),
        (
            "12_adjusted_community_structure_analysis.ipynb",
            [
                md(
                    """
                    # 12. Adjusted Community Structure Analysis (R)

                    This notebook revisits community similarity with multivariable models.
                    It combines a patient-aware PERMANOVA for overall community structure with pairwise mixed models
                    that ask whether shared patient or shared body site remain associated with lower Bray-Curtis distance
                    after adjusting for technical batch, host burden, read depth, and patient-relative elapsed time.
                    """
                ),
                code(
                    R_COMMON_SETUP
                    + """
                    suppressPackageStartupMessages({
                      library(vegan)
                      library(lme4)
                      library(lmerTest)
                      library(broom.mixed)
                      library(emmeans)
                    })
                    """
                ),
                md(
                    """
                    ## Load QC-Passing Community Data

                    The community models use bacterial relative abundance on the model-QC-passing samples.
                    The fixed-effect set is deliberately limited to avoid throwing in several collinear depth-like covariates at once.
                    """
                ),
                code(
                    """
                    qc <- read_tsv(table_file(2, "qc_metrics"), show_col_types = FALSE) |>
                      mutate(
                        model_qc_pass = tolower(as.character(model_qc_pass)) %in% c("true", "t", "1"),
                        culture_date = as.Date(culture_date),
                        batch_id = factor(batch_id),
                        culture_positive = factor(if_else(as.logical(culture_positive), "positive", "negative"),
                                                  levels = c("negative", "positive")),
                        patient_id = factor(sprintf("%02d", as.integer(patient_id))),
                        body_region = factor(body_region, levels = c("lower_extremity", "head_neck", "upper_extremity", "trunk_perineum", "unknown")),
                        chronicity_group = factor(chronicity_group, levels = c("unknown", "acute_like", "chronic_like", "mixed"))
                      ) |>
                      filter(model_qc_pass) |>
                      select(
                        sample_id,
                        patient_id,
                        batch_id,
                        culture_date,
                        years_since_first_sample,
                        location,
                        body_region,
                        chronicity_group,
                        culture_positive,
                        host_removed_fraction,
                        log10_bacterial_reads
                      )

                    counts <- read_csv(file.path(data_dir, "read_count_species_bac.csv"), show_col_types = FALSE)
                    colnames(counts)[1] <- "sample_id"
                    counts <- counts |>
                      filter(sample_id %in% qc$sample_id)
                    counts <- counts[match(qc$sample_id, counts$sample_id), ]

                    count_matrix <- counts |>
                      tibble::column_to_rownames("sample_id") |>
                      as.matrix()

                    rel_abundance <- sweep(count_matrix, 1, rowSums(count_matrix), "/")
                    rel_abundance[is.na(rel_abundance)] <- 0

                    community_summary <- tibble(
                      n_samples = nrow(qc),
                      n_patients = n_distinct(qc$patient_id),
                      n_batches = n_distinct(qc$batch_id),
                      n_species = ncol(rel_abundance),
                      fixed_effects = "batch_id + host_removed_fraction + log10_bacterial_reads + years_since_first_sample + body_region + chronicity_group + culture_positive",
                      patient_handling = "adonis2 strata = patient_id; pairwise lmer with sample-level random intercepts"
                    )

                    print(community_summary)
                    """
                ),
                md(
                    """
                    ## Run A Patient-Restricted PERMANOVA
                    """
                ),
                code(
                    """
                    set.seed(20260303)
                    permanova <- adonis2(
                      rel_abundance ~ batch_id + host_removed_fraction + log10_bacterial_reads + years_since_first_sample + body_region + chronicity_group + culture_positive,
                      data = qc,
                      method = "bray",
                      by = "margin",
                      permutations = 1999,
                      strata = qc$patient_id
                    )

                    permanova_table <- permanova |>
                      as.data.frame() |>
                      tibble::rownames_to_column("term") |>
                      as_tibble() |>
                      rename(
                        df = Df,
                        sum_of_squares = SumOfSqs,
                        r2 = R2,
                        f_statistic = F,
                        pvalue = `Pr(>F)`
                      ) |>
                      mutate(
                        qvalue = NA_real_,
                        term_label = case_when(
                          term == "batch_id" ~ "Batch date",
                          term == "host_removed_fraction" ~ "Host fraction",
                          term == "log10_bacterial_reads" ~ "Bacterial read depth",
                          term == "years_since_first_sample" ~ "Years since first patient sample",
                          term == "body_region" ~ "Body region",
                          term == "chronicity_group" ~ "Chronicity",
                          term == "culture_positive" ~ "Culture positivity",
                          TRUE ~ term
                        )
                      )

                    tested <- permanova_table |> filter(!term %in% c("Residual", "Total"))
                    permanova_table$qvalue[match(tested$term, permanova_table$term)] <- p.adjust(tested$pvalue, method = "BH")

                    write_tsv(permanova_table, table_file(29, "adjusted_community_permanova"))
                    print(permanova_table)
                    """
                ),
                md(
                    """
                    ## Build Pairwise Distances And Fit Cross-Classified Mixed Models

                    Two pairwise models are fit:
                    one using shared body region and one using exact shared cleaned location.
                    This separates broad site similarity from exact-site recurrence.
                    """
                ),
                code(
                    """
                    distance_matrix <- as.matrix(vegdist(rel_abundance, method = "bray"))
                    pair_rows <- vector("list", length = 0)

                    for (i in seq_len(nrow(qc) - 1)) {
                      for (j in seq((i + 1), nrow(qc))) {
                        pair_rows[[length(pair_rows) + 1]] <- tibble(
                          sample_a = qc$sample_id[i],
                          sample_b = qc$sample_id[j],
                          distance = distance_matrix[i, j],
                          same_patient = qc$patient_id[i] == qc$patient_id[j],
                          same_batch = qc$batch_id[i] == qc$batch_id[j],
                          same_body_region = qc$body_region[i] == qc$body_region[j],
                          same_location = qc$location[i] == qc$location[j],
                          same_chronicity = qc$chronicity_group[i] == qc$chronicity_group[j],
                          same_culture_positive = qc$culture_positive[i] == qc$culture_positive[j],
                          delta_years_since_first_sample = abs(qc$years_since_first_sample[i] - qc$years_since_first_sample[j]),
                          mean_host_fraction = mean(c(qc$host_removed_fraction[i], qc$host_removed_fraction[j])),
                          mean_log10_bacterial_reads = mean(c(qc$log10_bacterial_reads[i], qc$log10_bacterial_reads[j]))
                        )
                      }
                    }

                    pair_df <- bind_rows(pair_rows) |>
                      mutate(
                        sample_a = factor(sample_a),
                        sample_b = factor(sample_b),
                        same_patient = as.numeric(same_patient),
                        same_batch = as.numeric(same_batch),
                        same_body_region = as.numeric(same_body_region),
                        same_location = as.numeric(same_location),
                        same_chronicity = as.numeric(same_chronicity),
                        same_culture_positive = as.numeric(same_culture_positive)
                      )

                    pair_model_body_region <- lmer(
                      distance ~ same_patient + same_batch + same_body_region + same_chronicity + same_culture_positive + delta_years_since_first_sample + mean_host_fraction + mean_log10_bacterial_reads + (1 | sample_a) + (1 | sample_b),
                      data = pair_df,
                      REML = FALSE,
                      control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
                    )

                    pair_model_exact_location <- lmer(
                      distance ~ same_patient + same_batch + same_location + same_chronicity + same_culture_positive + delta_years_since_first_sample + mean_host_fraction + mean_log10_bacterial_reads + (1 | sample_a) + (1 | sample_b),
                      data = pair_df,
                      REML = FALSE,
                      control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
                    )

                    extract_pairwise_effects <- function(model, model_name) {
                      tidy(model, effects = "fixed", conf.int = TRUE) |>
                        filter(term != "(Intercept)") |>
                        mutate(
                          model = model_name,
                          term_label = case_when(
                            term == "same_patient" ~ "Same patient",
                            term == "same_batch" ~ "Same batch date",
                            term == "same_body_region" ~ "Same body region",
                            term == "same_location" ~ "Same exact location",
                            term == "same_chronicity" ~ "Same chronicity",
                            term == "same_culture_positive" ~ "Same culture positivity",
                            term == "delta_years_since_first_sample" ~ "Elapsed-time gap (years)",
                            term == "mean_host_fraction" ~ "Mean host fraction",
                            term == "mean_log10_bacterial_reads" ~ "Mean bacterial read depth",
                            TRUE ~ term
                          )
                        )
                    }

                    pairwise_effects_all <- bind_rows(
                      extract_pairwise_effects(pair_model_body_region, "body_region_model"),
                      extract_pairwise_effects(pair_model_exact_location, "exact_location_model")
                    ) |>
                      group_by(model) |>
                      mutate(qvalue = p.adjust(p.value, method = "BH")) |>
                      ungroup()

                    pairwise_status_all <- tibble(
                      model = c("body_region_model", "exact_location_model"),
                      n_pairs = nrow(pair_df),
                      aic = c(AIC(pair_model_body_region), AIC(pair_model_exact_location)),
                      bic = c(BIC(pair_model_body_region), BIC(pair_model_exact_location)),
                      logLik = c(as.numeric(logLik(pair_model_body_region)), as.numeric(logLik(pair_model_exact_location))),
                      singular = c(isSingular(pair_model_body_region), isSingular(pair_model_exact_location))
                    )

                    pairwise_effects <- pairwise_effects_all |> filter(model == "body_region_model")
                    pairwise_effects_exact <- pairwise_effects_all |> filter(model == "exact_location_model")
                    pairwise_status <- pairwise_status_all |> filter(model == "body_region_model")
                    pairwise_status_exact <- pairwise_status_all |> filter(model == "exact_location_model")

                    write_tsv(pairwise_effects, table_file(30, "pairwise_similarity_mixed_effects"))
                    write_tsv(pairwise_status, table_file(31, "pairwise_similarity_model_status"))
                    write_tsv(pairwise_effects_exact, table_file(39, "pairwise_similarity_mixed_effects_exact_location_sensitivity"))
                    write_tsv(pairwise_status_exact, table_file(40, "pairwise_similarity_model_status_exact_location_sensitivity"))

                    print(pairwise_effects)
                    print(pairwise_status)
                    """
                ),
                md(
                    """
                    ## Estimate Covariate-Adjusted Similarity Margins

                    To visualize adjusted similarity directly, we estimate marginal Bray-Curtis distances from the fitted pairwise mixed models.
                    Other pairwise matching indicators are held at 0, and continuous technical covariates are fixed at their sample means.
                    """
                ),
                code(
                    """
                    margin_specs <- tribble(
                      ~model_key, ~focal_term, ~term_label,
                      "body_region_model", "same_patient", "Same patient",
                      "body_region_model", "same_chronicity", "Same chronicity",
                      "body_region_model", "same_body_region", "Same body region",
                      "exact_location_model", "same_location", "Same exact location"
                    )

                    model_lookup <- list(
                      body_region_model = pair_model_body_region,
                      exact_location_model = pair_model_exact_location
                    )

                    make_margin_table <- function(model, focal_term, term_label, model_key) {
                      at_list <- list(
                        same_patient = 0,
                        same_batch = 0,
                        same_body_region = 0,
                        same_location = 0,
                        same_chronicity = 0,
                        same_culture_positive = 0,
                        delta_years_since_first_sample = mean(pair_df$delta_years_since_first_sample),
                        mean_host_fraction = mean(pair_df$mean_host_fraction),
                        mean_log10_bacterial_reads = mean(pair_df$mean_log10_bacterial_reads)
                      )
                      at_list[[focal_term]] <- c(0, 1)
                      em <- emmeans(model, specs = as.formula(paste("~", focal_term)), at = at_list)
                      out <- as_tibble(summary(em, infer = c(TRUE, TRUE)))
                      out$focal_level_value <- out[[focal_term]]
                      out |>
                        transmute(
                          model = model_key,
                          focal_term = focal_term,
                          term_label = term_label,
                          level = if_else(focal_level_value == 1, "Shared", "Not shared"),
                          emmean = emmean,
                          std.error = SE,
                          conf.low = lower.CL,
                          conf.high = upper.CL,
                          held_same_batch = 0,
                          held_same_culture_positive = 0,
                          held_elapsed_time_gap = mean(pair_df$delta_years_since_first_sample),
                          held_mean_host_fraction = mean(pair_df$mean_host_fraction),
                          held_mean_log10_bacterial_reads = mean(pair_df$mean_log10_bacterial_reads)
                        )
                    }

                    adjusted_margins_all <- bind_rows(lapply(seq_len(nrow(margin_specs)), function(i) {
                      make_margin_table(
                        model_lookup[[margin_specs$model_key[[i]]]],
                        margin_specs$focal_term[[i]],
                        margin_specs$term_label[[i]],
                        margin_specs$model_key[[i]]
                      )
                    }))

                    adjusted_margins <- adjusted_margins_all |> filter(model == "body_region_model")
                    adjusted_margins_exact <- adjusted_margins_all |> filter(model == "exact_location_model")
                    write_tsv(adjusted_margins, table_file(38, "pairwise_adjusted_margins"))
                    write_tsv(adjusted_margins_exact, table_file(41, "pairwise_adjusted_margins_exact_location_sensitivity"))
                    print(adjusted_margins)
                    """
                ),
                md(
                    """
                    ## Review Numbered Table Outputs
                    """
                ),
                code(
                    """
                    outputs <- tibble(
                      output = c(
                        table_file(29, "adjusted_community_permanova"),
                        table_file(30, "pairwise_similarity_mixed_effects"),
                        table_file(31, "pairwise_similarity_model_status"),
                        table_file(38, "pairwise_adjusted_margins"),
                        table_file(39, "pairwise_similarity_mixed_effects_exact_location_sensitivity"),
                        table_file(40, "pairwise_similarity_model_status_exact_location_sensitivity"),
                        table_file(41, "pairwise_adjusted_margins_exact_location_sensitivity")
                      )
                    )
                    print(outputs)
                    """
                ),
            ],
        ),
        (
            "12_adjusted_community_structure_plots.ipynb",
            [
                md(
                    """
                    # 12. Adjusted Community Structure Plots (Python)

                    This notebook reads the saved table outputs from the R analysis notebook
                    and renders the adjusted PERMANOVA, pairwise mixed-effects, and adjusted-margin figures.
                    """
                ),
                code(COMMON_SETUP),
                md(
                    """
                    ## Load Inputs
                    """
                ),
                code(
                    """
                    import math
                    from pathlib import Path

                    import matplotlib.pyplot as plt
                    import numpy as np
                    import pandas as pd
                    import seaborn as sns
                    from IPython.display import SVG
                    from matplotlib.lines import Line2D

                    sns.set_theme(style="whitegrid", context="talk")
                    plt.rcParams["svg.fonttype"] = "none"
                    plt.rcParams["pdf.fonttype"] = 42

                    permanova_table = pd.read_csv(
                        wc.table_path(context, 29, "adjusted_community_permanova"), sep="\\t"
                    )
                    pairwise_effects_body = pd.read_csv(
                        wc.table_path(context, 30, "pairwise_similarity_mixed_effects"), sep="\\t"
                    )
                    pairwise_effects_exact = pd.read_csv(
                        wc.table_path(context, 39, "pairwise_similarity_mixed_effects_exact_location_sensitivity"), sep="\\t"
                    )
                    adjusted_margins_body = pd.read_csv(
                        wc.table_path(context, 38, "pairwise_adjusted_margins"), sep="\\t"
                    )
                    adjusted_margins_exact = pd.read_csv(
                        wc.table_path(context, 41, "pairwise_adjusted_margins_exact_location_sensitivity"), sep="\\t"
                    )
                    """
                ),
                md(
                    """
                    ## Define Plot Helpers
                    """
                ),
                code(
                    """
                    MODEL_LABELS = {
                        "body_region_model": "Body-region model",
                        "exact_location_model": "Exact-location model",
                    }


                    def save_svg_and_jpg(fig: plt.Figure, output_path: Path) -> None:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(output_path, bbox_inches="tight")
                        fig.savefig(output_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)


                    def plot_adjusted_permanova(table_df: pd.DataFrame, output_path: Path) -> None:
                        plot_df = table_df.loc[~table_df["term"].isin(["Residual", "Total"])].copy()
                        plot_df = plot_df.sort_values("r2", ascending=True)
                        plot_df["significant"] = plot_df["qvalue"].fillna(1.0) <= 0.1
                        colors = plot_df["significant"].map({True: "#b22222", False: "#3b6a8f"}).tolist()

                        fig, ax = plt.subplots(figsize=(11, 7.5))
                        ax.barh(plot_df["term_label"], plot_df["r2"], color=colors)
                        ax.set_xlabel("Marginal R2")
                        ax.set_ylabel("")
                        ax.set_title("Adjusted PERMANOVA on Bray-Curtis community structure")
                        legend_handles = [
                            Line2D([0], [0], marker="s", color="none", markerfacecolor="#b22222", markeredgecolor="#b22222", markersize=10, label="q <= 0.1"),
                            Line2D([0], [0], marker="s", color="none", markerfacecolor="#3b6a8f", markeredgecolor="#3b6a8f", markersize=10, label="q > 0.1"),
                        ]
                        ax.legend(handles=legend_handles, frameon=False, loc="lower right")
                        fig.tight_layout()
                        save_svg_and_jpg(fig, output_path)
                        plt.close(fig)


                    def plot_pairwise_effects(effects: pd.DataFrame, output_path: Path) -> None:
                        if effects.empty:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.text(0.5, 0.5, "No pairwise-effect rows available.", ha="center", va="center", transform=ax.transAxes)
                            ax.set_axis_off()
                            save_svg_and_jpg(fig, output_path)
                            plt.close(fig)
                            return

                        if "effect" in effects.columns:
                            effects = effects.loc[effects["effect"] == "fixed"].copy()
                        effects["model_label"] = effects["model"].map(MODEL_LABELS).fillna(effects["model"])
                        effects["significant"] = effects["qvalue"].fillna(1.0) <= 0.1

                        preferred_terms = [
                            "Same patient",
                            "Same batch date",
                            "Same body region",
                            "Same exact location",
                            "Same chronicity",
                            "Same culture positivity",
                            "Elapsed-time gap (years)",
                            "Mean host fraction",
                            "Mean bacterial read depth",
                        ]
                        observed_terms = effects["term_label"].dropna().unique().tolist()
                        ordered_terms = [term for term in preferred_terms if term in observed_terms]
                        ordered_terms.extend([term for term in sorted(observed_terms) if term not in ordered_terms])
                        y_order = list(reversed(ordered_terms))

                        model_order = [model for model in ["body_region_model", "exact_location_model"] if model in effects["model"].unique()]
                        if not model_order:
                            model_order = sorted(effects["model"].dropna().unique().tolist())
                        model_labels = [MODEL_LABELS.get(model, model) for model in model_order]

                        fig, axes = plt.subplots(
                            1,
                            len(model_order),
                            figsize=(6.5 * len(model_order), max(5.0, 0.55 * len(y_order))),
                            sharey=True,
                        )
                        if len(model_order) == 1:
                            axes = [axes]

                        color_map = {True: "#b22222", False: "#3b6a8f"}
                        for ax, model, model_label in zip(axes, model_order, model_labels):
                            sub = effects.loc[effects["model"] == model].copy()
                            ax.axvline(0.0, linestyle="--", color="#808080", linewidth=0.8)
                            for _, row in sub.iterrows():
                                if row["term_label"] not in y_order:
                                    continue
                                y_pos = y_order.index(row["term_label"])
                                x = float(row["estimate"])
                                low = float(row["conf.low"])
                                high = float(row["conf.high"])
                                color = color_map[bool(row["significant"])]
                                ax.errorbar(
                                    x=x,
                                    y=y_pos,
                                    xerr=[[x - low], [high - x]],
                                    fmt="o",
                                    color=color,
                                    ecolor=color,
                                    elinewidth=1.0,
                                    capsize=3,
                                    markersize=6,
                                )

                            ax.set_title(model_label)
                            ax.set_xlabel("Coefficient on Bray-Curtis distance")
                            ax.set_yticks(np.arange(len(y_order)))
                            ax.set_yticklabels(y_order)

                        axes[0].set_ylabel("")
                        legend_handles = [
                            Line2D([0], [0], marker="o", color="none", markerfacecolor="#b22222", markeredgecolor="#b22222", label="q <= 0.1"),
                            Line2D([0], [0], marker="o", color="none", markerfacecolor="#3b6a8f", markeredgecolor="#3b6a8f", label="q > 0.1"),
                        ]
                        fig.legend(handles=legend_handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
                        fig.suptitle("Pairwise mixed models for Bray-Curtis distance", y=1.06)
                        fig.tight_layout()
                        save_svg_and_jpg(fig, output_path)
                        plt.close(fig)


                    def plot_adjusted_margins(margins: pd.DataFrame, output_path: Path) -> None:
                        if margins.empty:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.text(0.5, 0.5, "No adjusted-margin rows available.", ha="center", va="center", transform=ax.transAxes)
                            ax.set_axis_off()
                            save_svg_and_jpg(fig, output_path)
                            plt.close(fig)
                            return

                        preferred_terms = ["Same patient", "Same chronicity", "Same body region", "Same exact location"]
                        observed_terms = margins["term_label"].dropna().unique().tolist()
                        term_order = [term for term in preferred_terms if term in observed_terms]
                        term_order.extend([term for term in sorted(observed_terms) if term not in term_order])

                        ncols = 2
                        nrows = max(1, math.ceil(len(term_order) / ncols))
                        fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, max(4.5, 3.4 * nrows)), squeeze=False)
                        axes_flat = [ax for row in axes for ax in row]

                        level_order = ["Not shared", "Shared"]
                        level_colors = {"Not shared": "#6c757d", "Shared": "#b22222"}
                        y_map = {level: idx for idx, level in enumerate(level_order)}

                        for ax, term in zip(axes_flat, term_order):
                            sub = margins.loc[margins["term_label"] == term].copy()
                            if sub.empty:
                                ax.set_axis_off()
                                continue
                            sub["level"] = pd.Categorical(sub["level"], categories=level_order, ordered=True)
                            sub = sub.sort_values("level")

                            xs = []
                            ys = []
                            for _, row in sub.iterrows():
                                level = row["level"]
                                if pd.isna(level):
                                    continue
                                y = y_map[str(level)]
                                x = float(row["emmean"])
                                low = float(row["conf.low"])
                                high = float(row["conf.high"])
                                ax.errorbar(
                                    x=x,
                                    y=y,
                                    xerr=[[x - low], [high - x]],
                                    fmt="o",
                                    color=level_colors[str(level)],
                                    ecolor=level_colors[str(level)],
                                    elinewidth=1.2,
                                    capsize=3,
                                    markersize=7,
                                )
                                xs.append(x)
                                ys.append(y)

                            if len(xs) >= 2:
                                ax.plot(xs, ys, color="#a0a0a0", linewidth=0.9)

                            ax.set_title(term)
                            ax.set_yticks([y_map[level] for level in level_order])
                            ax.set_yticklabels(level_order)
                            ax.set_xlabel("Adjusted predicted Bray-Curtis distance")
                            ax.set_ylabel("")

                        for ax in axes_flat[len(term_order) :]:
                            ax.set_axis_off()

                        legend_handles = [
                            Line2D([0], [0], marker="o", color="none", markerfacecolor=level_colors["Not shared"], markeredgecolor=level_colors["Not shared"], label="Not shared"),
                            Line2D([0], [0], marker="o", color="none", markerfacecolor=level_colors["Shared"], markeredgecolor=level_colors["Shared"], label="Shared"),
                        ]
                        fig.legend(handles=legend_handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
                        fig.suptitle(
                            "Adjusted predicted Bray-Curtis distance from pairwise mixed models\\nOther pairwise indicators held at 0; technical covariates fixed at cohort means",
                            y=1.06,
                        )
                        fig.tight_layout()
                        save_svg_and_jpg(fig, output_path)
                        plt.close(fig)
                    """
                ),
                md(
                    """
                    ## Render Figure 12 Panels
                    """
                ),
                code(
                    """
                    fig_12_01 = wc.figure_path(context, 12, "adjusted_permanova")
                    plot_adjusted_permanova(permanova_table, fig_12_01)

                    body_effects = pairwise_effects_body.copy()
                    body_effects["model"] = "body_region_model"
                    fig_12_02 = wc.figure_path(context, 13, "pairwise_similarity_mixed")
                    plot_pairwise_effects(body_effects, fig_12_02)

                    fig_12_03 = wc.figure_path(context, 22, "pairwise_adjusted_margins")
                    plot_adjusted_margins(adjusted_margins_body, fig_12_03)

                    exact_effects = pairwise_effects_exact.copy()
                    exact_effects["model"] = "exact_location_model"
                    fig_12_04 = wc.figure_path(context, 23, "pairwise_similarity_mixed_exact_location_sensitivity")
                    plot_pairwise_effects(exact_effects, fig_12_04)

                    fig_12_05 = wc.figure_path(context, 24, "pairwise_adjusted_margins_exact_location_sensitivity")
                    plot_adjusted_margins(adjusted_margins_exact, fig_12_05)

                    display(SVG(filename=str(fig_12_01)))
                    display(SVG(filename=str(fig_12_02)))
                    display(SVG(filename=str(fig_12_03)))
                    display(SVG(filename=str(fig_12_04)))
                    display(SVG(filename=str(fig_12_05)))
                    """
                ),
            ],
        ),
        (
            "13_culture_threshold_and_concordance_analysis.ipynb",
            [
                md(
                    """
                    # 13. Culture Threshold And Concordance Analysis (R)

                    This notebook performs the analysis side of the culture-versus-metagenomics comparison:
                    descriptive Mann-Whitney concordance tests, threshold sweeps from 0% to 10% relative abundance,
                    Venn overlap table generation across display cutoffs, and nuisance-adjusted rank-based mixed models.
                    It writes all table outputs used by the plotting notebook.
                    """
                ),
                code(
                    R_COMMON_SETUP
                    + """
                    suppressPackageStartupMessages({
                      library(lmerTest)
                      library(broom.mixed)
                    })
                    """
                ),
                md(
                    """
                    ## Load Species Abundances, Define Culture Groups, And Compute Descriptive Concordance
                    """
                ),
                code(
                    """
                    culture_groups <- tibble(
                      group = c("s_aureus", "p_aeruginosa", "serratia_marcescens", "proteus_mirabilis", "gas", "klebsiella_spp", "e_coli", "acinetobacter_baumannii", "e_faecalis"),
                      label = c("S. aureus", "P. aeruginosa", "Serratia", "Proteus", "GAS", "Klebsiella spp.", "E. coli", "A. baumannii", "E. faecalis"),
                      culture_col = c("culture_s_aureus", "culture_p_aeruginosa", "culture_serratia_marcescens", "culture_proteus_mirabilis", "culture_gas", "culture_klebsiella_spp", "culture_e_coli", "culture_acinetobacter_baumannii", "culture_e_faecalis"),
                      taxa = list(
                        c("Staphylococcus aureus"),
                        c("Pseudomonas aeruginosa"),
                        c("Serratia marcescens"),
                        c("Proteus mirabilis"),
                        c("Streptococcus pyogenes"),
                        c("Klebsiella pneumoniae", "Klebsiella oxytoca"),
                        c("Escherichia coli"),
                        c("Acinetobacter baumannii"),
                        c("Enterococcus faecalis")
                      )
                    )

                    qc <- read_tsv(table_file(2, "qc_metrics"), show_col_types = FALSE) |>
                      mutate(
                        culture_date = as.Date(culture_date),
                        batch_id = factor(batch_id),
                        body_region = factor(body_region, levels = c("lower_extremity", "head_neck", "upper_extremity", "trunk_perineum", "unknown")),
                        chronicity_group = factor(chronicity_group, levels = c("unknown", "acute_like", "chronic_like", "mixed")),
                        patient_id = factor(sprintf("%02d", as.integer(patient_id)))
                      )

                    counts <- read_csv(file.path(data_dir, "read_count_species_bac.csv"), show_col_types = FALSE)
                    colnames(counts)[1] <- "sample_id"
                    counts <- counts |>
                      filter(sample_id %in% qc$sample_id)
                    counts <- counts[match(qc$sample_id, counts$sample_id), ]

                    count_matrix <- counts |>
                      tibble::column_to_rownames("sample_id") |>
                      as.matrix()
                    rel_abundance <- sweep(count_matrix, 1, rowSums(count_matrix), "/")
                    rel_abundance[is.na(rel_abundance)] <- 0

                    group_abundance <- tibble(sample_id = rownames(rel_abundance))
                    for (idx in seq_len(nrow(culture_groups))) {
                      taxa_present <- intersect(culture_groups$taxa[[idx]], colnames(rel_abundance))
                      values <- if (length(taxa_present) == 0) rep(0, nrow(rel_abundance)) else rowSums(rel_abundance[, taxa_present, drop = FALSE])
                      group_abundance[[culture_groups$group[idx]]] <- values
                    }

                    culture_model_data <- qc |>
                      left_join(group_abundance, by = "sample_id")

                    descriptive_rows <- vector("list", length = 0)
                    for (idx in seq_len(nrow(culture_groups))) {
                      group_name <- culture_groups$group[idx]
                      group_label <- culture_groups$label[idx]
                      culture_col <- culture_groups$culture_col[idx]
                      observed <- as.logical(culture_model_data[[culture_col]])
                      abundance <- culture_model_data[[group_name]]
                      positive_values <- abundance[observed]
                      negative_values <- abundance[!observed]
                      n_positive <- sum(observed, na.rm = TRUE)
                      n_negative <- sum(!observed, na.rm = TRUE)
                      if (n_positive > 0 && n_negative > 0) {
                        test <- suppressWarnings(wilcox.test(positive_values, negative_values, alternative = "greater", exact = FALSE))
                        pvalue <- unname(test$p.value)
                        u_stat <- unname(test$statistic)
                      } else {
                        pvalue <- NA_real_
                        u_stat <- NA_real_
                      }
                      descriptive_rows[[length(descriptive_rows) + 1]] <- tibble(
                        group = group_name,
                        label = group_label,
                        n_culture_positive = n_positive,
                        n_culture_negative = n_negative,
                        u_statistic = u_stat,
                        pvalue = pvalue
                      )
                    }
                    descriptive_concordance <- bind_rows(descriptive_rows) |>
                      mutate(qvalue = if_else(!is.na(pvalue), p.adjust(pvalue, method = "BH"), NA_real_))
                    write_tsv(descriptive_concordance, table_file(42, "culture_concordance_descriptive"))

                    abundance_plot_df <- culture_model_data |>
                      select(sample_id, all_of(culture_groups$culture_col), all_of(culture_groups$group)) |>
                      pivot_longer(cols = all_of(culture_groups$group), names_to = "group", values_to = "rel_abundance") |>
                      left_join(culture_groups |> select(group, label, culture_col), by = "group") |>
                      rowwise() |>
                      mutate(culture_status = if_else(as.logical(cur_data()[[culture_col]]), "Culture positive", "Culture negative")) |>
                      ungroup() |>
                      group_by(group, label, culture_status) |>
                      filter(n() > 0) |>
                      ungroup() |>
                      mutate(log10_rel_abundance = log10(rel_abundance + 1e-6)) |>
                      select(sample_id, group, label, culture_status, rel_abundance, log10_rel_abundance)

                    write_tsv(abundance_plot_df, table_file(43, "culture_abundance_plot_data"))

                    print(culture_groups)
                    print(descriptive_concordance)
                    """
                ),
                md(
                    """
                    ## Sweep Detection Thresholds From 0% To 10%
                    """
                ),
                code(
                    """
                    safe_div <- function(num, den) ifelse(den > 0, num / den, NA_real_)

                    calc_kappa <- function(tp, fp, fn, tn) {
                      total <- tp + fp + fn + tn
                      observed <- safe_div(tp + tn, total)
                      expected <- safe_div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), total^2)
                      ifelse(is.na(observed) | is.na(expected) | expected >= 1, NA_real_, (observed - expected) / (1 - expected))
                    }

                    thresholds <- seq(0, 0.10, by = 0.001)
                    threshold_rows <- vector("list", length = 0)

                    for (idx in seq_len(nrow(culture_groups))) {
                      group_name <- culture_groups$group[idx]
                      group_label <- culture_groups$label[idx]
                      culture_col <- culture_groups$culture_col[idx]

                      observed <- as.logical(culture_model_data[[culture_col]])
                      abundance <- culture_model_data[[group_name]]

                      for (threshold in thresholds) {
                        detected <- abundance >= threshold
                        tp <- sum(observed & detected, na.rm = TRUE)
                        fp <- sum(!observed & detected, na.rm = TRUE)
                        fn <- sum(observed & !detected, na.rm = TRUE)
                        tn <- sum(!observed & !detected, na.rm = TRUE)

                        precision <- safe_div(tp, tp + fp)
                        recall <- safe_div(tp, tp + fn)
                        f1 <- ifelse(is.na(precision) | is.na(recall) | precision + recall == 0, NA_real_, 2 * precision * recall / (precision + recall))

                        threshold_rows[[length(threshold_rows) + 1]] <- tibble(
                          group = group_name,
                          label = group_label,
                          threshold = threshold,
                          true_positive = tp,
                          false_positive = fp,
                          false_negative = fn,
                          true_negative = tn,
                          sensitivity = recall,
                          specificity = safe_div(tn, tn + fp),
                          ppv = precision,
                          npv = safe_div(tn, tn + fn),
                          f1 = f1,
                          kappa = calc_kappa(tp, fp, fn, tn)
                        )
                      }
                    }

                    threshold_sweep <- bind_rows(threshold_rows)
                    optimal_thresholds <- threshold_sweep |>
                      group_by(group, label) |>
                      arrange(desc(f1), desc(kappa), threshold) |>
                      slice_head(n = 1) |>
                      ungroup()

                    display_cutoffs <- c(0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)
                    venn_counts <- threshold_sweep |>
                      filter(round(threshold, 6) %in% round(display_cutoffs, 6)) |>
                      transmute(
                        group,
                        label,
                        threshold,
                        culture_only = false_negative,
                        sequencing_only = false_positive,
                        both = true_positive,
                        neither = true_negative
                      )

                    write_tsv(threshold_sweep, table_file(32, "culture_threshold_sweep"))
                    write_tsv(optimal_thresholds, table_file(33, "culture_optimal_thresholds"))
                    write_tsv(venn_counts, table_file(35, "culture_venn_counts"))

                    print(optimal_thresholds)
                    """
                ),
                md(
                    """
                    ## Fit Technical/Nuisance-Adjusted Rank-Based Mixed Models
                    """
                ),
                code(
                    """
                    concordance_rows <- vector("list", length = 0)

                    for (idx in seq_len(nrow(culture_groups))) {
                      group_name <- culture_groups$group[idx]
                      group_label <- culture_groups$label[idx]
                      culture_col <- culture_groups$culture_col[idx]
                      threshold_used <- optimal_thresholds |>
                        filter(group == group_name) |>
                        pull(threshold)

                      model_df <- culture_model_data |>
                        transmute(
                          sample_id,
                          patient_id,
                          batch_id,
                          bacterial_species_reads,
                          host_removed_fraction,
                          log10_bacterial_reads,
                          culture_status = factor(
                            if_else(as.logical(.data[[culture_col]]), "Culture positive", "Culture negative"),
                            levels = c("Culture negative", "Culture positive")
                          ),
                          rel_abundance = .data[[group_name]]
                        ) |>
                        filter(
                          !is.na(batch_id),
                          !is.na(bacterial_species_reads),
                          !is.na(host_removed_fraction),
                          !is.na(log10_bacterial_reads),
                          !is.na(culture_status),
                          !is.na(rel_abundance)
                        ) |>
                        filter(bacterial_species_reads >= 5000) |>
                        mutate(
                          abundance_rank = rank(rel_abundance, ties.method = "average"),
                          normal_score_abundance = qnorm((abundance_rank - 0.5) / n())
                        )

                      n_positive <- sum(model_df$culture_status == "Culture positive")
                      n_negative <- sum(model_df$culture_status == "Culture negative")
                      if (n_positive == 0 || n_negative == 0 || sd(model_df$normal_score_abundance) == 0) {
                        concordance_rows[[length(concordance_rows) + 1]] <- tibble(
                          group = group_name,
                          label = group_label,
                          threshold = threshold_used,
                          n_samples = nrow(model_df),
                          n_positive = n_positive,
                          n_negative = n_negative,
                          predictor = "normal_score_abundance",
                          status = "failed_before_fit",
                          singular_fit = NA,
                          estimate = NA_real_,
                          conf.low = NA_real_,
                          conf.high = NA_real_,
                          pvalue = NA_real_,
                          qvalue = NA_real_,
                          detail = "No class variation or no variation in the rank-normalized metagenomic abundance response."
                        )
                        next
                      }

                      fit_warnings <- character(0)
                      fit <- tryCatch(
                        withCallingHandlers(
                          lmer(
                            normal_score_abundance ~ culture_status + host_removed_fraction + log10_bacterial_reads + (1 | patient_id) + (1 | batch_id),
                            data = model_df,
                            REML = FALSE
                          ),
                          warning = function(w) {
                            fit_warnings <<- c(fit_warnings, conditionMessage(w))
                            invokeRestart("muffleWarning")
                          }
                        ),
                        error = function(e) e
                      )

                      if (inherits(fit, "error")) {
                        concordance_rows[[length(concordance_rows) + 1]] <- tibble(
                          group = group_name,
                          label = group_label,
                          threshold = threshold_used,
                          n_samples = nrow(model_df),
                          n_positive = n_positive,
                          n_negative = n_negative,
                          predictor = "normal_score_abundance",
                          status = "failed",
                          singular_fit = NA,
                          estimate = NA_real_,
                          conf.low = NA_real_,
                          conf.high = NA_real_,
                          pvalue = NA_real_,
                          qvalue = NA_real_,
                          detail = as.character(fit$message)
                        )
                        next
                      }

                      abundance_term <- tidy(fit, effects = "fixed", conf.int = TRUE, conf.method = "Wald") |>
                        filter(term == "culture_statusCulture positive")
                      fit_singular <- isSingular(fit, tol = 1e-5)
                      fit_status <- case_when(
                        nrow(abundance_term) == 0 || is.na(abundance_term$p.value) || is.na(abundance_term$conf.low) || is.na(abundance_term$conf.high) ~ "separated",
                        fit_singular ~ "ok_singular",
                        TRUE ~ "ok"
                      )
                      fit_detail <- c()
                      if (fit_status == "separated") {
                        fit_detail <- c(fit_detail, "Model showed undefined Wald intervals for the culture-status effect in the rank-based mixed model.")
                      }
                      if (fit_singular) {
                        fit_detail <- c(fit_detail, "Singular random-effects fit (boundary variance estimate).")
                      }
                      if (length(fit_warnings) > 0) {
                        fit_detail <- c(fit_detail, unique(fit_warnings))
                      }

                      concordance_rows[[length(concordance_rows) + 1]] <- tibble(
                        group = group_name,
                        label = group_label,
                        threshold = threshold_used,
                        n_samples = nrow(model_df),
                        n_positive = n_positive,
                        n_negative = n_negative,
                        predictor = "normal_score_abundance",
                        status = fit_status,
                        singular_fit = fit_singular,
                        estimate = abundance_term$estimate,
                        conf.low = abundance_term$conf.low,
                        conf.high = abundance_term$conf.high,
                        pvalue = abundance_term$p.value,
                        qvalue = NA_real_,
                        detail = paste(unique(fit_detail), collapse = " | ")
                      )
                    }

                    concordance_table <- bind_rows(concordance_rows)
                    ok_rows <- concordance_table$status %in% c("ok", "ok_singular") & !is.na(concordance_table$pvalue)
                    concordance_table$qvalue[ok_rows] <- p.adjust(concordance_table$pvalue[ok_rows], method = "BH")

                    write_tsv(concordance_table, table_file(34, "culture_mixed_concordance"))

                    print(concordance_table)
                    """
                ),
                md(
                    """
                    ## Review Numbered Table Outputs
                    """
                ),
                code(
                    """
                    outputs <- tibble(
                      output = c(
                        table_file(32, "culture_threshold_sweep"),
                        table_file(33, "culture_optimal_thresholds"),
                        table_file(34, "culture_mixed_concordance"),
                        table_file(35, "culture_venn_counts"),
                        table_file(42, "culture_concordance_descriptive"),
                        table_file(43, "culture_abundance_plot_data")
                      )
                    )
                    print(outputs)
                    """
                ),
            ],
        ),
        (
            "13_culture_threshold_and_concordance_plots.ipynb",
            [
                md(
                    """
                    # 13. Culture Threshold And Concordance Plots (Python)

                    This notebook reads the numbered table outputs from the R analysis notebook
                    and generates Figure 13 panels directly in Python/matplotlib/seaborn.
                    """
                ),
                code(COMMON_SETUP),
                md(
                    """
                    ## Load Table Inputs
                    """
                ),
                code(
                    """
                    import math
                    from pathlib import Path

                    import matplotlib.pyplot as plt
                    import numpy as np
                    import pandas as pd
                    import seaborn as sns
                    from IPython.display import SVG
                    from matplotlib.lines import Line2D
                    from matplotlib.patches import Circle

                    try:
                        from matplotlib_venn import venn2
                    except Exception:
                        venn2 = None

                    sns.set_theme(style="whitegrid", context="talk")
                    plt.rcParams["svg.fonttype"] = "none"
                    plt.rcParams["pdf.fonttype"] = 42

                    threshold_sweep = pd.read_csv(
                        wc.table_path(context, 32, "culture_threshold_sweep"), sep="\\t"
                    )
                    optimal_thresholds = pd.read_csv(
                        wc.table_path(context, 33, "culture_optimal_thresholds"), sep="\\t"
                    )
                    mixed_concordance = pd.read_csv(
                        wc.table_path(context, 34, "culture_mixed_concordance"), sep="\\t"
                    )
                    venn_counts = pd.read_csv(
                        wc.table_path(context, 35, "culture_venn_counts"), sep="\\t"
                    )
                    descriptive_concordance = pd.read_csv(
                        wc.table_path(context, 42, "culture_concordance_descriptive"), sep="\\t"
                    )
                    abundance_plot_df = pd.read_csv(
                        wc.table_path(context, 43, "culture_abundance_plot_data"), sep="\\t"
                    )
                    """
                ),
                md(
                    """
                    ## Define Plotting Helpers
                    """
                ),
                code(
                    """
                    DISPLAY_THRESHOLDS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


                    def save_svg_and_jpg(fig: plt.Figure, output_path: Path) -> None:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(output_path, bbox_inches="tight")
                        fig.savefig(output_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)


                    def axes_grid(
                        n_panels: int,
                        ncols: int = 3,
                        panel_width: float = 4.5,
                        panel_height: float = 3.4,
                        sharex: bool = False,
                        sharey: bool = False,
                    ):
                        ncols = max(1, min(ncols, n_panels if n_panels > 0 else 1))
                        nrows = max(1, math.ceil(n_panels / ncols))
                        fig, axes = plt.subplots(
                            nrows,
                            ncols,
                            figsize=(panel_width * ncols, panel_height * nrows),
                            squeeze=False,
                            sharex=sharex,
                            sharey=sharey,
                        )
                        flat_axes = [ax for row in axes for ax in row]
                        return fig, flat_axes


                    def plot_threshold_sweep(
                        threshold_data: pd.DataFrame,
                        optimal_data: pd.DataFrame,
                        output_path: Path,
                    ) -> None:
                        merged = threshold_data.merge(
                            optimal_data[["group", "threshold"]].rename(columns={"threshold": "threshold_opt"}),
                            on="group",
                            how="left",
                        )
                        label_order = (
                            threshold_data[["group", "label"]]
                            .drop_duplicates()
                            .sort_values("group")["label"]
                            .tolist()
                        )
                        fig, axes = axes_grid(
                            len(label_order),
                            ncols=3,
                            panel_width=4.4,
                            panel_height=3.1,
                        )
                        for idx, label in enumerate(label_order):
                            ax = axes[idx]
                            sub = merged.loc[merged["label"] == label].sort_values("threshold")
                            ax.plot(sub["threshold"] * 100.0, sub["f1"], color="#3b6a8f", linewidth=1.4)
                            if sub["threshold_opt"].notna().any():
                                x_opt = float(sub["threshold_opt"].dropna().iloc[0] * 100.0)
                                ax.axvline(x_opt, color="#b22222", linestyle="--", linewidth=1.0)
                            ax.set_title(label, fontsize=10)
                            ax.set_xlabel("Threshold (%)", fontsize=9)
                            ax.set_ylabel("F1", fontsize=9)
                            ax.set_xlim(0, 10)
                            ax.set_ylim(0, 1.05)
                            ax.tick_params(labelsize=8)

                        for ax in axes[len(label_order) :]:
                            ax.set_axis_off()

                        fig.suptitle("Culture-versus-metagenomics threshold sweep", y=0.995, fontsize=13)
                        fig.tight_layout(rect=(0, 0, 1, 0.98))
                        save_svg_and_jpg(fig, output_path)
                        plt.close(fig)


                    def circle_intersection_area(r1: float, r2: float, d: float) -> float:
                        if d >= r1 + r2:
                            return 0.0
                        if d <= abs(r1 - r2):
                            return math.pi * min(r1, r2) ** 2
                        term1 = r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
                        term2 = r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
                        term3 = 0.5 * math.sqrt(
                            (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
                        )
                        return term1 + term2 - term3


                    def solve_center_distance(r1: float, r2: float, target_intersection: float) -> float:
                        max_overlap = math.pi * min(r1, r2) ** 2
                        target = min(max(target_intersection, 0.0), max_overlap)
                        if target <= 0:
                            return r1 + r2 + max(r1, r2) * 0.35
                        if abs(target - max_overlap) < 1e-12:
                            return abs(r1 - r2)
                        lo = abs(r1 - r2) + 1e-10
                        hi = r1 + r2 - 1e-10
                        for _ in range(80):
                            mid = 0.5 * (lo + hi)
                            area = circle_intersection_area(r1, r2, mid)
                            if area > target:
                                lo = mid
                            else:
                                hi = mid
                        return 0.5 * (lo + hi)


                    def style_venn_axis(ax: plt.Axes) -> None:
                        ax.set_aspect("equal")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        for spine in ax.spines.values():
                            spine.set_visible(False)


                    def draw_with_custom_solver(
                        ax: plt.Axes,
                        culture_only: float,
                        sequencing_only: float,
                        both: float,
                        neither: float,
                    ) -> None:
                        a_size = max(culture_only + both, 0.0)
                        b_size = max(sequencing_only + both, 0.0)
                        overlap = max(both, 0.0)

                        if a_size <= 0 and b_size <= 0:
                            ax.text(
                                0.5,
                                0.57,
                                "No positive calls",
                                ha="center",
                                va="center",
                                fontsize=8,
                                transform=ax.transAxes,
                            )
                            ax.text(
                                0.5,
                                0.35,
                                f"Neither: {int(round(neither))}",
                                ha="center",
                                va="center",
                                fontsize=8,
                                transform=ax.transAxes,
                            )
                            style_venn_axis(ax)
                            return

                        r1 = max(math.sqrt(a_size / math.pi), 1e-8)
                        r2 = max(math.sqrt(b_size / math.pi), 1e-8)
                        distance = solve_center_distance(r1, r2, overlap)

                        c1, c2 = 0.0, distance
                        x_min = min(c1 - r1, c2 - r2)
                        x_max = max(c1 + r1, c2 + r2)
                        y_lim = max(r1, r2)
                        pad = max(0.22 * max(r1, r2), 0.2)
                        center_shift = -0.5 * (x_min + x_max)
                        c1 += center_shift
                        c2 += center_shift

                        ax.add_patch(
                            Circle((c1, 0), r1, facecolor="#3b6a8f", edgecolor="#2f4d63", alpha=0.35, linewidth=1.0)
                        )
                        ax.add_patch(
                            Circle((c2, 0), r2, facecolor="#b22222", edgecolor="#7f1b1b", alpha=0.35, linewidth=1.0)
                        )
                        ax.text(c1 - 0.45 * r1, 0, f"{int(round(culture_only))}", ha="center", va="center", fontsize=8)
                        ax.text((c1 + c2) / 2, 0, f"{int(round(overlap))}", ha="center", va="center", fontsize=8, fontweight="bold")
                        ax.text(c2 + 0.45 * r2, 0, f"{int(round(sequencing_only))}", ha="center", va="center", fontsize=8)
                        ax.text(
                            0.5 * (x_min + x_max) + center_shift,
                            -(y_lim + 0.55 * pad),
                            f"Neither: {int(round(neither))}",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )
                        ax.set_xlim(x_min - pad + center_shift, x_max + pad + center_shift)
                        ax.set_ylim(-(y_lim + 1.0 * pad), y_lim + 0.9 * pad)
                        style_venn_axis(ax)


                    def draw_proportional_venn(
                        ax: plt.Axes,
                        culture_only: float,
                        sequencing_only: float,
                        both: float,
                        neither: float,
                    ) -> None:
                        if venn2 is not None:
                            venn = venn2(
                                subsets=(culture_only, sequencing_only, both),
                                set_labels=("", ""),
                                ax=ax,
                            )
                            if venn is not None:
                                style_map = {"10": "#3b6a8f", "01": "#b22222", "11": "#6c4a86"}
                                for key, color in style_map.items():
                                    patch = venn.get_patch_by_id(key)
                                    if patch is not None:
                                        patch.set_facecolor(color)
                                        patch.set_alpha(0.35)
                                        patch.set_edgecolor("#404040")
                                        patch.set_linewidth(0.9)

                                label_values = {
                                    "10": int(round(culture_only)),
                                    "11": int(round(both)),
                                    "01": int(round(sequencing_only)),
                                }
                                for key, value in label_values.items():
                                    text = venn.get_label_by_id(key)
                                    if text is not None:
                                        text.set_text(str(value))
                                        text.set_fontsize(8)
                                        if key == "11":
                                            text.set_fontweight("bold")

                                ax.text(
                                    0.5,
                                    0.03,
                                    f"Neither: {int(round(neither))}",
                                    ha="center",
                                    va="bottom",
                                    transform=ax.transAxes,
                                    fontsize=7,
                                )
                                style_venn_axis(ax)
                                return

                        draw_with_custom_solver(ax, culture_only, sequencing_only, both, neither)


                    def plot_venn_grid(venn_data: pd.DataFrame, output_path: Path) -> None:
                        venn_df = venn_data.copy()
                        venn_df["threshold"] = venn_df["threshold"].astype(float)
                        available = sorted(venn_df["threshold"].unique().tolist())
                        cutoff_levels = [
                            value
                            for value in DISPLAY_THRESHOLDS
                            if any(math.isclose(value, a, rel_tol=0, abs_tol=1e-9) for a in available)
                        ]
                        if not cutoff_levels:
                            cutoff_levels = available

                        group_df = venn_df[["group", "label"]].drop_duplicates()
                        group_order = group_df["group"].tolist()
                        group_to_label = dict(zip(group_df["group"], group_df["label"]))

                        n_rows = len(group_order)
                        n_cols = len(cutoff_levels)
                        fig, axes = plt.subplots(
                            n_rows,
                            n_cols,
                            figsize=(2.45 * n_cols, 2.05 * n_rows),
                            squeeze=False,
                        )

                        for row_idx, group in enumerate(group_order):
                            for col_idx, cutoff in enumerate(cutoff_levels):
                                ax = axes[row_idx, col_idx]
                                row = venn_df.loc[
                                    (venn_df["group"] == group)
                                    & (
                                        np.isclose(
                                            venn_df["threshold"],
                                            cutoff,
                                            rtol=0.0,
                                            atol=1e-9,
                                        )
                                    )
                                ]
                                if row.empty:
                                    ax.text(
                                        0.5,
                                        0.5,
                                        "No data",
                                        ha="center",
                                        va="center",
                                        fontsize=8,
                                        transform=ax.transAxes,
                                    )
                                    style_venn_axis(ax)
                                else:
                                    rec = row.iloc[0]
                                    draw_proportional_venn(
                                        ax=ax,
                                        culture_only=float(rec["culture_only"]),
                                        sequencing_only=float(rec["sequencing_only"]),
                                        both=float(rec["both"]),
                                        neither=float(rec["neither"]),
                                    )

                                if row_idx == 0:
                                    ax.set_title(f"{cutoff * 100:.1f}%", fontsize=9, pad=4)
                                if col_idx == 0:
                                    ax.text(
                                        -0.26,
                                        0.5,
                                        group_to_label[group],
                                        transform=ax.transAxes,
                                        rotation=90,
                                        ha="center",
                                        va="center",
                                        fontsize=8,
                                    )

                        fig.suptitle(
                            "Culture-only / Both / Sequencing-only overlaps (area-proportional)",
                            y=0.997,
                            fontsize=12,
                        )
                        fig.text(
                            0.5,
                            0.01,
                            "Display cutoff for metagenomic relative abundance",
                            ha="center",
                            fontsize=9,
                        )
                        fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.985))
                        save_svg_and_jpg(fig, output_path)
                        plt.close(fig)


                    def plot_abundance_boxplot(
                        abundance_df: pd.DataFrame,
                        summary_df: pd.DataFrame,
                        output_path: Path,
                    ) -> None:
                        q_lookup = {}
                        if "label" in summary_df.columns and "qvalue" in summary_df.columns:
                            q_lookup = summary_df.set_index("label")["qvalue"].to_dict()

                        label_order = (
                            abundance_df[["group", "label"]]
                            .drop_duplicates()
                            .sort_values("group")["label"]
                            .tolist()
                        )
                        status_order = ["Culture negative", "Culture positive"]
                        palette = {
                            "Culture negative": "#4c78a8",
                            "Culture positive": "#d65f5f",
                        }

                        fig, axes = axes_grid(
                            len(label_order),
                            ncols=3,
                            panel_width=2.0,
                            panel_height=3.0,
                        )
                        for idx, label in enumerate(label_order):
                            ax = axes[idx]
                            sub = abundance_df.loc[abundance_df["label"] == label].copy()
                            sns.boxplot(
                                data=sub,
                                x="culture_status",
                                y="log10_rel_abundance",
                                order=status_order,
                                palette=palette,
                                width=0.62,
                                fliersize=3.0,
                                linewidth=1.0,
                                ax=ax,
                            )
                            qvalue = q_lookup.get(label, np.nan)
                            if pd.notna(qvalue):
                                ax.set_title(f"{label} U-test q={qvalue:.4}", fontsize=10)
                            else:
                                ax.set_title(label, fontsize=10)
                            ax.set_xlabel("")
                            ax.set_ylabel("log10(rel. abundance + 1e-6)", fontsize=9)
                            ax.set_xticklabels(
                                [tick.get_text().split()[-1].capitalize() for tick in ax.get_xticklabels()],
                                rotation=20,
                                ha="right",
                                fontsize=8,
                            )
                            ax.set_yticklabels(
                                [tick.get_text() for tick in ax.get_yticklabels()],
                                fontsize=8,
                            )

                        for ax in axes[len(label_order) :]:
                            ax.set_axis_off()

                        fig.suptitle(
                            "Metagenomic abundance by culture status (boxplots)",
                            y=0.995,
                            fontsize=13,
                        )
                        fig.tight_layout(rect=(0, 0, 1, 0.98))
                        save_svg_and_jpg(fig, output_path)
                        plt.close(fig)


                    def plot_adjusted_concordance(
                        concordance_df: pd.DataFrame,
                        output_path: Path,
                    ) -> None:
                        plot_df = concordance_df.loc[
                            concordance_df["status"].isin(["ok", "ok_singular"])
                        ].copy()
                        if plot_df.empty:
                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.text(
                                0.5,
                                0.5,
                                "No adjusted concordance models converged.",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_axis_off()
                            save_svg_and_jpg(fig, output_path)
                            plt.close(fig)
                            return

                        plot_df["significant"] = plot_df["qvalue"].fillna(1.0) <= 0.1
                        plot_df["fit_quality"] = np.where(
                            plot_df["status"] == "ok_singular",
                            "Singular fit",
                            "Regular fit",
                        )
                        plot_df = plot_df.sort_values("estimate").reset_index(drop=True)

                        fig, ax = plt.subplots(figsize=(7, max(5.0, 0.55 * len(plot_df))))
                        ax.axvline(0, color="#808080", linestyle="--", linewidth=0.9)
                        color_map = {True: "#b22222", False: "#3b6a8f"}
                        marker_map = {"Regular fit": "o", "Singular fit": "^"}

                        y_values = np.arange(len(plot_df))
                        for idx, row in plot_df.iterrows():
                            x = float(row["estimate"])
                            low = float(row["conf.low"])
                            high = float(row["conf.high"])
                            color = color_map[bool(row["significant"])]
                            marker = marker_map[row["fit_quality"]]
                            ax.errorbar(
                                x=x,
                                y=idx,
                                xerr=[[x - low], [high - x]],
                                fmt=marker,
                                color=color,
                                ecolor=color,
                                elinewidth=1.1,
                                capsize=3,
                                markersize=7,
                            )

                        ax.set_yticks(y_values)
                        ax.set_yticklabels(plot_df["label"].tolist())
                        ax.set_xlabel(
                            "Adjusted difference in rank-normalized abundance\\n(Culture positive - Culture negative)"
                        )
                        ax.set_ylabel("")
                        ax.set_title(
                            "Technical/nuisance-adjusted rank-based culture concordance models"
                        )

                        legend_handles = [
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="none",
                                markerfacecolor="#b22222",
                                markeredgecolor="#b22222",
                                label="q <= 0.1",
                            ),
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="none",
                                markerfacecolor="#3b6a8f",
                                markeredgecolor="#3b6a8f",
                                label="q > 0.1",
                            ),
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="black",
                                linestyle="None",
                                label="Regular fit",
                            ),
                            Line2D(
                                [0],
                                [0],
                                marker="^",
                                color="black",
                                linestyle="None",
                                label="Singular fit",
                            ),
                        ]
                        ax.legend(handles=legend_handles, frameon=False, loc="best")
                        fig.tight_layout()
                        save_svg_and_jpg(fig, output_path)
                        plt.close(fig)
                    """
                ),
                md(
                    """
                    ## Render Figure 13 Panels From Numbered Tables
                    """
                ),
                code(
                    """
                    fig_threshold = wc.figure_path(context, 14, "culture_threshold_sweep")
                    plot_threshold_sweep(threshold_sweep, optimal_thresholds, fig_threshold)

                    fig_venn = wc.figure_path(context, 15, "culture_venn_diagrams")
                    plot_venn_grid(venn_counts, fig_venn)

                    fig_abundance = wc.figure_path(context, 16, "culture_abundance_density")
                    plot_abundance_boxplot(abundance_plot_df, descriptive_concordance, fig_abundance)

                    fig_adjusted = wc.figure_path(context, 17, "culture_adjusted_concordance")
                    plot_adjusted_concordance(mixed_concordance, fig_adjusted)

                    display(SVG(filename=str(fig_threshold)))
                    display(SVG(filename=str(fig_venn)))
                    display(SVG(filename=str(fig_abundance)))
                    display(SVG(filename=str(fig_adjusted)))
                    """
                ),
            ],
        ),
        (
            "14_host_fraction_gaussian_story_figures.ipynb",
            [
                md(
                    """
                    # 14. Host Fraction Gaussian Model Figures

                    This notebook assembles the main figure set for the Gaussian mixed-model host-fraction story:
                    overview plots, mixed-model summary plots, focused chronicity/location follow-up, and patient/batch random-effect structure.
                    """
                ),
                code(COMMON_SETUP),
                md(
                    """
                    ## Load Host Inputs And Refit The Gaussian Mixed Model
                    """
                ),
                code(
                    """
                    import re
                    import warnings

                    import matplotlib.pyplot as plt
                    import numpy as np
                    import pandas as pd
                    import seaborn as sns
                    import statsmodels.formula.api as smf
                    from matplotlib.ticker import PercentFormatter
                    from scipy.stats import chi2
                    from statsmodels.stats.multitest import multipletests

                    sns.set_theme(style="whitegrid", context="talk")

                    host_effects = pd.read_csv(wc.table_path(context, 9, "host_mixed_effects"), sep="\\t")
                    host_status = pd.read_csv(wc.table_path(context, 10, "host_mixed_status"), sep="\\t")

                    host_model_effects = (
                        host_effects.loc[host_effects["model_name"] == advanced.HOST_MODEL_NAME]
                        .copy()
                    )
                    host_model_effects = host_model_effects.loc[host_model_effects["term"] != "Intercept"].copy()

                    def factor_family(term: str) -> str:
                        if "C(body_region" in term:
                            return "body_region"
                        if "C(chronicity_group" in term:
                            return "chronicity_group"
                        if "culture_positive" in term:
                            return "culture_positive"
                        if "years_since_first_sample" in term:
                            return "elapsed_time"
                        return "other"

                    host_model_effects["factor_family"] = host_model_effects["term"].map(factor_family)
                    host_model_effects["term_label"] = host_model_effects["term"].map(base.prettify_model_term)
                    host_model_effects["posthoc_qvalue"] = np.nan
                    for family, idx in host_model_effects.groupby("factor_family").groups.items():
                        pvals = host_model_effects.loc[list(idx), "pvalue"].to_numpy()
                        host_model_effects.loc[list(idx), "posthoc_qvalue"] = multipletests(pvals, method="fdr_bh")[1]

                    qc = base_data["qc"].copy()
                    host_plot_df = qc.dropna(
                        subset=[
                            "host_removed_fraction",
                            "host_logit",
                            "body_region",
                            "chronicity_group",
                            "patient_id",
                            "batch_id",
                            "culture_positive_label",
                            "years_since_first_sample",
                        ]
                    ).copy()
                    host_plot_df["acute_like_binary"] = np.where(
                        host_plot_df["chronicity_group"].astype(str) == "acute_like",
                        "Acute-like",
                        "Other",
                    )
                    host_plot_df["upper_extremity_binary"] = np.where(
                        host_plot_df["body_region"].astype(str) == "upper_extremity",
                        "Upper extremity",
                        "Other",
                    )

                    METHODS = ["lbfgs", "powell", "bfgs"]
                    HOST_MODEL_FORMULA = advanced.HOST_FORMULAS[advanced.HOST_MODEL_NAME]
                    VC_FORMULA = {"patient": "0 + C(patient_id)", "batch": "0 + C(batch_id)"}

                    def fit_best_gaussian_mixed_model(frame, formula):
                        work = frame.copy()
                        work["all_group"] = "all"
                        candidates = []
                        last_error = None
                        for method in METHODS:
                            try:
                                with warnings.catch_warnings(record=True) as caught:
                                    warnings.simplefilter("always")
                                    fit = smf.mixedlm(
                                        formula=formula,
                                        data=work,
                                        groups=work["all_group"],
                                        vc_formula=VC_FORMULA,
                                    ).fit(reml=False, method=method, maxiter=500, disp=False)
                                candidates.append((fit, [str(item.message) for item in caught], method))
                            except Exception as exc:
                                last_error = exc
                        if not candidates:
                            raise RuntimeError(last_error)
                        converged = [item for item in candidates if bool(getattr(item[0], "converged", False))]
                        pool = converged if converged else candidates
                        fit, warning_messages, optimizer = min(pool, key=lambda item: item[0].aic)
                        return fit, {
                            "optimizer": optimizer,
                            "warning_count": len(warning_messages),
                            "warnings": " | ".join(sorted(set(warning_messages)))[:2000],
                        }

                    fit, fit_meta = fit_best_gaussian_mixed_model(host_plot_df, HOST_MODEL_FORMULA)

                    no_body_formula = (
                        "host_logit ~ C(chronicity_group, Treatment('unknown')) "
                        "+ C(culture_positive_label, Treatment('negative')) + years_since_first_sample"
                    )
                    no_chronicity_formula = (
                        "host_logit ~ C(body_region, Treatment('lower_extremity')) "
                        "+ C(culture_positive_label, Treatment('negative')) + years_since_first_sample"
                    )
                    no_body_fit, _ = fit_best_gaussian_mixed_model(host_plot_df, no_body_formula)
                    no_chronicity_fit, _ = fit_best_gaussian_mixed_model(host_plot_df, no_chronicity_formula)

                    def likelihood_ratio(full_fit, reduced_fit):
                        statistic = max(0.0, 2.0 * (full_fit.llf - reduced_fit.llf))
                        df_diff = int(len(full_fit.params) - len(reduced_fit.params))
                        pvalue = chi2.sf(statistic, df_diff)
                        return statistic, df_diff, pvalue

                    body_stat, body_df, body_p = likelihood_ratio(fit, no_body_fit)
                    chronicity_stat, chronicity_df, chronicity_p = likelihood_ratio(fit, no_chronicity_fit)

                    gaussian_followup = host_model_effects.loc[
                        :,
                        [
                            "model_name",
                            "term",
                            "term_label",
                            "estimate",
                            "conf_low",
                            "conf_high",
                            "pvalue",
                            "qvalue",
                            "factor_family",
                            "posthoc_qvalue",
                        ],
                    ].copy()
                    gaussian_followup["record_type"] = "full_model_term"
                    gaussian_followup["lrt_statistic"] = np.nan
                    gaussian_followup["df_diff"] = np.nan
                    omnibus_rows = pd.DataFrame(
                        [
                            {
                                "model_name": advanced.HOST_MODEL_NAME,
                                "term": "body_region_overall",
                                "term_label": "Body region overall",
                                "estimate": np.nan,
                                "conf_low": np.nan,
                                "conf_high": np.nan,
                                "pvalue": body_p,
                                "qvalue": np.nan,
                                "factor_family": "body_region",
                                "posthoc_qvalue": np.nan,
                                "record_type": "omnibus_test",
                                "lrt_statistic": body_stat,
                                "df_diff": body_df,
                            },
                            {
                                "model_name": advanced.HOST_MODEL_NAME,
                                "term": "chronicity_group_overall",
                                "term_label": "Chronicity overall",
                                "estimate": np.nan,
                                "conf_low": np.nan,
                                "conf_high": np.nan,
                                "pvalue": chronicity_p,
                                "qvalue": np.nan,
                                "factor_family": "chronicity_group",
                                "posthoc_qvalue": np.nan,
                                "record_type": "omnibus_test",
                                "lrt_statistic": chronicity_stat,
                                "df_diff": chronicity_df,
                            },
                        ]
                    )
                    gaussian_followup = pd.concat([gaussian_followup, omnibus_rows], ignore_index=True)
                    wc.save_table(gaussian_followup, wc.table_path(context, 36, "host_gaussian_followup"))

                    re_series = fit.random_effects[next(iter(fit.random_effects))].rename("random_intercept").reset_index()
                    re_series = re_series.rename(columns={"index": "raw_label"})
                    pattern = re.compile(r"^(batch|patient)\\[C\\((?:batch_id|patient_id)\\)\\[(.+)\\]\\]$")
                    random_rows = []
                    for _, row in re_series.iterrows():
                        match = pattern.match(row["raw_label"])
                        if match is None:
                            continue
                        random_rows.append(
                            {
                                "group": match.group(1),
                                "level": match.group(2),
                                "random_intercept": row["random_intercept"],
                            }
                        )
                    random_effects_df = pd.DataFrame(random_rows).sort_values(["group", "random_intercept"])
                    wc.save_table(random_effects_df, wc.table_path(context, 37, "host_gaussian_random_effects"))
                    """
                ),
                md(
                    """
                    ## Figure 18. Host Fraction Overview
                    """
                ),
                code(
                    """
                    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8), gridspec_kw={"width_ratios": [2.0, 1.0, 1.0]})

                    patient_order = (
                        host_plot_df["patient_id"]
                        .astype(str)
                        .dropna()
                        .drop_duplicates()
                        .sort_values(key=lambda s: pd.to_numeric(s, errors="coerce").fillna(9999))
                        .tolist()
                    )
                    body_region_order = [region for region in ["lower_extremity", "head_neck", "upper_extremity", "trunk_perineum"] if region in set(host_plot_df["body_region"].astype(str))]
                    if not body_region_order:
                        body_region_order = sorted(host_plot_df["body_region"].dropna().astype(str).unique().tolist())
                    body_region_labels = {
                        "lower_extremity": "Lower extremity",
                        "head_neck": "Head/neck",
                        "upper_extremity": "Upper extremity",
                        "trunk_perineum": "Trunk/perineum",
                    }
                    body_palette = dict(zip(body_region_order, sns.color_palette("Set2", n_colors=len(body_region_order))))
                    patient_counts = host_plot_df["patient_id"].astype(str).value_counts()
                    box_patients = set(patient_counts[patient_counts > 3].index.tolist())
                    box_df = host_plot_df.loc[host_plot_df["patient_id"].astype(str).isin(box_patients)].copy()

                    if not box_df.empty:
                        sns.boxplot(
                            data=box_df,
                            x="patient_id",
                            y="host_removed_fraction",
                            order=patient_order,
                            color="white",
                            fliersize=0,
                            width=0.55,
                            linewidth=1.0,
                            ax=axes[0],
                        )
                    sns.stripplot(
                        data=host_plot_df,
                        x="patient_id",
                        y="host_removed_fraction",
                        hue="body_region",
                        order=patient_order,
                        hue_order=body_region_order,
                        palette=body_palette,
                        dodge=False,
                        jitter=0.22,
                        size=4.5,
                        alpha=0.9,
                        ax=axes[0],
                    )
                    handles, labels = axes[0].get_legend_handles_labels()
                    if handles:
                        display_labels = [body_region_labels.get(label, label) for label in labels]
                        axes[0].legend(
                            handles,
                            display_labels,
                            title="Body region",
                            loc="upper left",
                            bbox_to_anchor=(1.01, 1.0),
                            frameon=False,
                        )
                    axes[0].set_title("Host fraction by patient")
                    axes[0].set_xlabel("Patient")
                    axes[0].set_ylabel("Host genomic DNA fraction")
                    axes[0].tick_params(axis="x", rotation=45)
                    axes[0].yaxis.set_major_formatter(PercentFormatter(1))

                    chronicity_order = ["unknown", "acute_like", "chronic_like", "mixed"]
                    sns.boxplot(
                        data=host_plot_df,
                        x="chronicity_group",
                        y="host_removed_fraction",
                        order=chronicity_order,
                        color="white",
                        ax=axes[1],
                    )
                    sns.stripplot(
                        data=host_plot_df,
                        x="chronicity_group",
                        y="host_removed_fraction",
                        order=chronicity_order,
                        color="#b56576",
                        size=5,
                        alpha=0.75,
                        ax=axes[1],
                    )
                    axes[1].set_title("Host fraction by chronicity")
                    axes[1].set_xlabel("")
                    axes[1].set_ylabel("Host genomic DNA fraction")
                    axes[1].tick_params(axis="x", rotation=25)
                    axes[1].yaxis.set_major_formatter(PercentFormatter(1))

                    body_order = ["lower_extremity", "head_neck", "upper_extremity", "trunk_perineum"]
                    sns.boxplot(
                        data=host_plot_df,
                        x="body_region",
                        y="host_removed_fraction",
                        order=body_order,
                        color="white",
                        ax=axes[2],
                    )
                    sns.stripplot(
                        data=host_plot_df,
                        x="body_region",
                        y="host_removed_fraction",
                        order=body_order,
                        color="#6d597a",
                        size=5,
                        alpha=0.75,
                        ax=axes[2],
                    )
                    axes[2].set_title("Host fraction by body region")
                    axes[2].set_xlabel("")
                    axes[2].set_ylabel("")
                    axes[2].set_xticklabels(["Lower ext.", "Head/neck", "Upper ext.", "Trunk/perineum"], rotation=25)
                    axes[2].yaxis.set_major_formatter(PercentFormatter(1))

                    fig.tight_layout()
                    fig_14_01_path = wc.figure_path(context, 18, "host_fraction_overview")
                    fig.savefig(fig_14_01_path, bbox_inches="tight")
                    fig.savefig(fig_14_01_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)
                    display(SVG(filename=str(wc.figure_path(context, 18, "host_fraction_overview"))))
                    """
                ),
                md(
                    """
                    ## Figure 19. Gaussian Mixed-Model Summary
                    """
                ),
                code(
                    """
                    fig = plt.figure(figsize=(18, 7))
                    grid = fig.add_gridspec(1, 3, width_ratios=[1.8, 0.8, 1.2])
                    ax_forest = fig.add_subplot(grid[0, 0])
                    ax_var = fig.add_subplot(grid[0, 1])
                    ax_lrt = fig.add_subplot(grid[0, 2])

                    plot_df = host_model_effects.copy().sort_values("estimate")
                    y = np.arange(plot_df.shape[0])
                    colors = np.where(plot_df["factor_family"] == "chronicity_group", "#bc4749", "#457b9d")
                    ax_forest.axvline(0, color="grey", linestyle="--", linewidth=1)
                    ax_forest.hlines(y, plot_df["conf_low"], plot_df["conf_high"], color=colors, linewidth=2)
                    ax_forest.scatter(plot_df["estimate"], y, color=colors, s=55, zorder=3)
                    ax_forest.set_yticks(y)
                    ax_forest.set_yticklabels(plot_df["term_label"])
                    ax_forest.set_xlabel("Coefficient on logit host-fraction scale")
                    ax_forest.set_title("Fixed effects: host model")

                    host_model_status = host_status.loc[
                        (host_status["record_type"] == "model_fit")
                        & (host_status["model_name"] == advanced.HOST_MODEL_NAME)
                    ].iloc[0]
                    var_df = pd.DataFrame(
                        {
                            "component": ["Patient", "Batch"],
                            "variance": [host_model_status["patient_var"], host_model_status["batch_var"]],
                        }
                    )
                    ax_var.hlines(var_df["component"], 0, var_df["variance"], color="#8d99ae", linewidth=2)
                    ax_var.scatter(var_df["variance"], var_df["component"], color="#1d3557", s=70, zorder=3)
                    for _, row in var_df.iterrows():
                        ax_var.text(row["variance"] + 0.03, row["component"], f"{row['variance']:.2f}", va="center", fontsize=11)
                    ax_var.set_title("Random-effect variance")
                    ax_var.set_xlabel("Variance component")

                    lrt_df = host_status.loc[
                        (host_status["record_type"] == "random_effect_test")
                        & (host_status["model_name"] == advanced.HOST_MODEL_NAME)
                    ].copy()
                    label_map = {
                        "patient_only_vs_fixed": "Patient vs fixed-only",
                        "batch_only_vs_fixed": "Batch vs fixed-only",
                        "batch_added_to_patient": "Add batch to patient",
                        "patient_added_to_batch": "Add patient to batch",
                    }
                    lrt_df["test_label"] = lrt_df["tested_effect"].map(label_map)
                    lrt_df["neglog10_p"] = -np.log10(lrt_df["pvalue_boundary"].clip(lower=1e-12))
                    lrt_df = lrt_df.sort_values("neglog10_p")
                    ax_lrt.axvline(-np.log10(0.05), color="grey", linestyle="--", linewidth=1)
                    ax_lrt.hlines(lrt_df["test_label"], 0, lrt_df["neglog10_p"], color="#adb5bd", linewidth=2)
                    ax_lrt.scatter(lrt_df["neglog10_p"], lrt_df["test_label"], color="#d62828", s=65, zorder=3)
                    for _, row in lrt_df.iterrows():
                        ax_lrt.text(row["neglog10_p"] + 0.04, row["test_label"], f"p={row['pvalue_boundary']:.3g}", va="center", fontsize=11)
                    ax_lrt.set_title("Random-effect LRTs")
                    ax_lrt.set_xlabel("-log10(boundary-corrected p)")

                    fig.tight_layout()
                    fig_14_02_path = wc.figure_path(context, 19, "host_gaussian_mixed_summary")
                    fig.savefig(fig_14_02_path, bbox_inches="tight")
                    fig.savefig(fig_14_02_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)
                    display(SVG(filename=str(wc.figure_path(context, 19, "host_gaussian_mixed_summary"))))
                    """
                ),
                md(
                    """
                    ## Figure 20. Focused Chronicity And Location Follow-Up
                    """
                ),
                code(
                    """
                    acute_q = float(
                        host_model_effects.loc[
                            host_model_effects["term"] == "C(chronicity_group, Treatment('unknown'))[T.acute_like]",
                            "posthoc_qvalue",
                        ].iloc[0]
                    )
                    upper_q = float(
                        host_model_effects.loc[
                            host_model_effects["term"] == "C(body_region, Treatment('lower_extremity'))[T.upper_extremity]",
                            "posthoc_qvalue",
                        ].iloc[0]
                    )

                    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

                    sns.boxplot(
                        data=host_plot_df,
                        x="acute_like_binary",
                        y="host_removed_fraction",
                        order=["Other", "Acute-like"],
                        palette=["#d9d9d9", "#bc4749"],
                        ax=axes[0],
                    )
                    sns.stripplot(
                        data=host_plot_df,
                        x="acute_like_binary",
                        y="host_removed_fraction",
                        order=["Other", "Acute-like"],
                        color="black",
                        alpha=0.65,
                        size=5,
                        ax=axes[0],
                    )
                    axes[0].set_title(f"Acute-like chronicity\\nfull-model factor-specific q = {acute_q:.3f}")
                    axes[0].set_xlabel("")
                    axes[0].set_ylabel("Host genomic DNA fraction")
                    axes[0].yaxis.set_major_formatter(PercentFormatter(1))

                    sns.boxplot(
                        data=host_plot_df,
                        x="upper_extremity_binary",
                        y="host_removed_fraction",
                        order=["Other", "Upper extremity"],
                        palette=["#d9d9d9", "#457b9d"],
                        ax=axes[1],
                    )
                    sns.stripplot(
                        data=host_plot_df,
                        x="upper_extremity_binary",
                        y="host_removed_fraction",
                        order=["Other", "Upper extremity"],
                        color="black",
                        alpha=0.65,
                        size=5,
                        ax=axes[1],
                    )
                    axes[1].set_title(f"Upper-extremity location\\nfull-model factor-specific q = {upper_q:.3f}")
                    axes[1].set_xlabel("")
                    axes[1].set_ylabel("")
                    axes[1].yaxis.set_major_formatter(PercentFormatter(1))

                    fig.tight_layout()
                    fig_14_03_path = wc.figure_path(context, 20, "host_gaussian_followup")
                    fig.savefig(fig_14_03_path, bbox_inches="tight")
                    fig.savefig(fig_14_03_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)
                    display(SVG(filename=str(wc.figure_path(context, 20, "host_gaussian_followup"))))
                    """
                ),
                md(
                    """
                    ## Figure 21. Patient And Batch Random Intercepts
                    """
                ),
                code(
                    """
                    patient_df = random_effects_df.loc[random_effects_df["group"] == "patient"].sort_values("random_intercept")
                    batch_df = random_effects_df.loc[random_effects_df["group"] == "batch"].sort_values("random_intercept")

                    fig, axes = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [0.9, 1.6]})

                    axes[0].axvline(0, color="grey", linestyle="--", linewidth=1)
                    axes[0].hlines(patient_df["level"], 0, patient_df["random_intercept"], color="#adb5bd", linewidth=1.8)
                    axes[0].scatter(patient_df["random_intercept"], patient_df["level"], color="#2a9d8f", s=45, zorder=3)
                    axes[0].set_title("Patient random intercepts")
                    axes[0].set_xlabel("Conditional random intercept")
                    axes[0].set_ylabel("Patient")

                    axes[1].axvline(0, color="grey", linestyle="--", linewidth=1)
                    axes[1].hlines(batch_df["level"], 0, batch_df["random_intercept"], color="#adb5bd", linewidth=1.5)
                    axes[1].scatter(batch_df["random_intercept"], batch_df["level"], color="#f4a261", s=30, zorder=3)
                    axes[1].set_title("Culture-date batch random intercepts")
                    axes[1].set_xlabel("Conditional random intercept")
                    axes[1].set_ylabel("Batch")

                    fig.tight_layout()
                    fig_14_04_path = wc.figure_path(context, 21, "host_gaussian_random_intercepts")
                    fig.savefig(fig_14_04_path, bbox_inches="tight")
                    fig.savefig(fig_14_04_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)
                    display(SVG(filename=str(wc.figure_path(context, 21, "host_gaussian_random_intercepts"))))
                    """
                ),
                md(
                    """
                    ## Review The Numbered Outputs
                    """
                ),
                code(
                    """
                    display(pd.read_csv(wc.table_path(context, 36, "host_gaussian_followup"), sep="\\t"))
                    display(pd.read_csv(wc.table_path(context, 37, "host_gaussian_random_effects"), sep="\\t").head(20))
                    """
                ),
            ],
        ),
    ]

    notebook_paths: list[Path] = []
    r_notebooks = {
        "09_maaslin2_multivariable.ipynb",
        "10_lefse_targeted_contrast.ipynb",
        "11_host_fraction_beta_binomial.ipynb",
        "12_adjusted_community_structure_analysis.ipynb",
        "13_culture_threshold_and_concordance_analysis.ipynb",
    }

    for name, cells in notebooks:
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        if name in r_notebooks:
            nb["metadata"]["kernelspec"] = {
                "display_name": "R (eb)",
                "language": "R",
                "name": "ir-eb",
            }
            nb["metadata"]["language_info"] = {
                "name": "R",
                "version": "4.3",
            }
        else:
            nb["metadata"]["kernelspec"] = {
                "display_name": "Python (eb)",
                "language": "python",
                "name": "eb",
            }
            nb["metadata"]["language_info"] = {
                "name": "python",
                "version": "3.12",
            }
        path = output_dir / name
        nbf.write(nb, path)
        notebook_paths.append(path)
    return notebook_paths


def execute_notebooks(paths: list[Path]) -> None:
    for path in paths:
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", path.name],
            cwd=path.parent,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the analysis_update notebook workflow.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the notebooks in-place after generating them.",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent
    notebooks = build_notebooks(output_dir)
    if args.execute:
        execute_notebooks(notebooks)


if __name__ == "__main__":
    main()
