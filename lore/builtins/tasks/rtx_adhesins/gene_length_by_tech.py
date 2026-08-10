"""
Plot the length of genes by sequencing technology.
Used in the RTX Adhesins project to verify the need for long-read sequencing.
"""
import lore
import json
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns


class GeneLengthByTechInputs:
    """Inputs for Gene Length by Sequencing Technology Task"""
    genome_reports = lore.ArtifactInput(
        accepted_data="tabular",
        load_as="adapted",
        select="single",
        label="Assembly metadata",
    )


class GeneLengthByTechOutputs:
    """Outputs for Gene Length by Sequencing Technology Task"""
    plot = lore.TaskOutput(
        data_type="plot",
        label="Gene length by sequencing technology",
        is_primary=True,
    )
    statistics = lore.TaskOutput(
        data_type="json",
        label="Gene length statistics by sequencing technology",
    )


def categorize_tech(raw_tech):
    """Assigns discrete categories to sequencing technologies based on the raw tech string"""
    if pd.isna(raw_tech):
        return "Unknown"
    tech = str(raw_tech).lower()

    has_illumina = any(x in tech for x in ["illumina", "miseq", "hiseq", "nextseq", "novaseq", "solexa"])
    has_pacbio = any(x in tech for x in ["pacbio", "smrt", "sequel", "rsii", "rs ii"])
    has_nanopore = any(x in tech for x in ["ont", "nanopore", "minion", "promethion", "flongle", "gridion"])

    has_other = any(x in tech for x in ["454", "sanger", "ion", "torrent", "solid", "bgi", "mgi"])

    if has_illumina and has_pacbio:
        return "PacBio + Illumina"
    elif has_illumina and has_nanopore:
        return "Nanopore + Illumina"
    elif has_pacbio and has_nanopore:
        return "PacBio + Nanopore"
    elif has_illumina:
        return "Illumina"
    elif has_pacbio:
        return "PacBio"
    elif has_nanopore:
        return "Nanopore"
    elif has_other:
        return "Other"
    else:
        return "Unknown"


def _statistics_by_tech(df: pd.DataFrame) -> dict:
    """Compute statistics and significance tests on gene lengths by tech"""
    clean_df = df.dropna(subset=["subject_length", "tech"]).copy()

    stats_out = {"summary": {}, "anova": {}, "tukey_hsd": None}

    # 1. Summary statistics
    for tech, group in clean_df.groupby("tech"):
        stats_out["summary"][tech] = {
            "count": int(len(group)),
            "mean_length": float(group["subject_length"].mean()),
            "median_length": float(group["subject_length"].median()),
            "min_length": int(group["subject_length"].min()),
            "max_length": int(group["subject_length"].max()),
        }

    # 2. ANOVA test
    groups = [group["subject_length"].values for name, group in clean_df.groupby("tech")]

    if len(groups) > 1:
        # A. One-way ANOVA
        f_stat, p_val = stats.f_oneway(*groups)
        stats_out["anova"] = {
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
        }

        # B. Tukey's Post-Hoc (honest significant difference/HSD)
        tukey = pairwise_tukeyhsd(
            endog=clean_df["subject_length"],
            groups=clean_df["tech"],
            alpha=0.05,
        )

        # C. Parse statsmodels summary table into a list of dicts
        reject = tukey.reject
        meandiffs = tukey.meandiffs
        pvalues = tukey.pvalues

        if reject is None or meandiffs is None or pvalues is None:
            raise RuntimeError("Tukey HSD returned incomplete results")

        pair_i, pair_j = tukey._multicomp.pairindices
        group_names = tukey.groupsunique

        tukey_results = [
            {
                "group1": str(group_names[i]),
                "group2": str(group_names[j]),
                "meandiff": float(mean_diff),
                "p_adj": float(p_adj),
                "reject_null": bool(reject_null),
            }
            for i, j, mean_diff, p_adj, reject_null in zip(
                pair_i,
                pair_j,
                meandiffs,
                pvalues,
                reject,
                strict=True,
            )
        ]
        stats_out["tukey_hsd"] = tukey_results

    return stats_out


@lore.task(
    "rtx_adhesins.gene_length_by_tech",
    inputs=GeneLengthByTechInputs,
    outputs=GeneLengthByTechOutputs,
    name="Plot Gene Length by Sequencing Technology",
    category="Adhesins manuscript",
    icon="🗠",
    preview_mode="full",
)
def gene_length_by_tech(
    ctx: lore.ExecutionContext,
    genome_reports: list[dict],
):
    """
    Generate a violin plot of gene lengths by sequencing technology.
    """
    # 1. Prepare dataframe
    df = pd.DataFrame(genome_reports)

    if "subject_length" not in df.columns:
        raise ValueError("Missing 'subject_length' column!")

    df["subject_length"] = pd.to_numeric(df["subject_length"], errors="coerce")
    df["tech"] = df["sequencing_tech"].apply(categorize_tech)

    tech_order = [
        "Illumina",
        "PacBio",
        "PacBio + Illumina",
        # "PacBio + Nanopore",  # not typically present
        "Nanopore",
        "Nanopore + Illumina",
        "Other",
    ]

    # 2. Compute statistics
    stats_out = _statistics_by_tech(df)
    ctx.materialize_content(
        content=json.dumps(stats_out),
        output_key="statistics",
        extension="json",
    )

    # 3. Build the plot
    fig, ax = plt.subplots(
        nrows=2,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 3],
            "hspace": 0.08,
        },
    )

    # Top panel
    sns.countplot(
        data=df,
        x="tech",
        order=tech_order,
        color=sns.color_palette("muted")[0],
        ax=ax[0],
    )

    ax[0].bar_label(ax[0].containers[0], padding=3, fmt="n=%d")

    # Bottom panel
    sns.violinplot(
        data=df,
        x="tech",
        y="subject_length",
        order=tech_order,
        cut=0,
        density_norm="width",
        common_norm=True,
        # inner box-and-whisker plot
        inner="box",
        inner_kws={
            "box_width": 4.0,
            "whis_width": 1.0,
            "marker": "o",
        },
        # Appearance
        linewidth=0,
        hue="tech",
        palette="muted",
        ax=ax[1],
    )
    ax[1].set_xlabel("Sequencing technology")
    ax[1].set_ylabel("Protein length (aa)")

    plt.xticks(rotation=45, ha="right")
    sns.despine()

    # 4. Save the plot
    png_path = ctx.get_temp_path("length_by_tech_violinplot.png")

    plt.rcParams["svg.fonttype"] = "none"  # Editable text rather than vectorized paths
    svg_path = ctx.get_temp_path("length_by_tech_violinplot.svg")

    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)

    plt.close(fig)

    ctx.materialize_file(
        source={
            "main": svg_path,
            "png": png_path,
        },
        metadata={
            "tech_categories": df["tech"].value_counts().to_dict(),
            "gene_count": len(df),
        }
    )

