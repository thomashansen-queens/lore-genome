"""
Visualize a simple genomic feature track for a specific window.
"""
import lore
from lore import viz as v
import pandas as pd


class RegionFeaturesInputs:
    """Input data for the Region Features task"""
    annotations = lore.ArtifactInput(
        accepted_data=["ncbi_annotation_packages", "genome_annotations"],
        select="single",
        load_as="adapted",
        description="Genome annotations containing gene/CDS features",
    )
    replicon = lore.ValueInput(
        str | None,
        label="Target Replicon/Chromosome",
        examples=["NZ_TH123456.1"],
        default=None,
        description="The exact accession/ID of the contig. If blank, visualize all annotations.",
    )
    start_bp = lore.ValueInput(
        int | None,
        label="Start Position (bp)",
        default=None,
        description="The start position of the genomic window to visualize. If blank, starts at start of annotations.",
    )
    end_bp = lore.ValueInput(
        int | None,
        label="End Position (bp)",
        default=None,
        description="The end position of the genomic window to visualize. If blank, ends at end of annotations.",
    )
    # anchor_locus = lore.ValueInput(str, label="Anchor Locus Tag (Overrides Start/End)")
    include_metadata = lore.ValueInput(
        bool,
        description="Whether to include metadata in the SVG hover tooltip. Disable to reduce SVG file size.",
        default=True,
        label="Tooltip data",
    )


class RegionFeaturesOutputs:
    svg = lore.TaskOutput(
        label="Region Features SVG",
        data_type="svg",
        is_primary=True,
    )


@lore.task(
    "annotation.region_features",
    name="Features by region (bp)",
    inputs=RegionFeaturesInputs,
    outputs=RegionFeaturesOutputs,
    description="Generate a feature track for a specific genomic window from genome annotations.",
    icon="☷",
    preview_mode="live_full",
)
def region_features(
    ctx: lore.ExecutionContext,
    annotations: list[dict],
    replicon: str | None = None,
    start_bp: int | None = None,
    end_bp: int | None = None,
    include_metadata: bool = True,
):
    """
    Generate a feature track for a specific genomic window from genome annotations.
    """
    df = pd.DataFrame(annotations)
    if df.empty:
        ctx.logger.warning(f"No annotations found in the input artifact.")
        raise ValueError(f"No annotations found in the input artifact.")

    # 1. Isolate replicon
    if replicon is not None:
        replicon = replicon.strip()
        df = df[df["contig"] == replicon]
        if df.empty:
            ctx.logger.warning(f"No annotations found for replicon '{replicon}'")
            raise ValueError(f"No annotations found for replicon '{replicon}'")

    # 2. Trim to window
    df[["begin", "end"]] = df[["begin", "end"]].apply(pd.to_numeric, errors="coerce")

    if start_bp is None:
        start_bp = int(df["begin"].min()) or 0
    if end_bp is None:
        end_bp = int(df["end"].max()) or 1

    if start_bp >= end_bp:
        ctx.logger.warning("Start position >= end position. Reversing order.")
        start_bp, end_bp = end_bp, start_bp

    df = df[(df["end"] >= start_bp) & (df["begin"] <= end_bp)].copy()
    if df.empty:
        ctx.logger.warning(f"No annotations found in the window {start_bp}-{end_bp} for replicon '{replicon}'")
        raise ValueError(f"No annotations found in the window {start_bp}-{end_bp} for replicon '{replicon}'")

    # 3. Create features
    shape_map = {"plus": v.FeatureShape.ARROW_RIGHT, "minus": v.FeatureShape.ARROW_LEFT}
    df["_shape"] = df["orientation"].map(shape_map).fillna(v.FeatureShape.BOX)

    df["_label"] = df.get("symbol", pd.Series(dtype=str)) \
                    .combine_first(df.get("name", pd.Series(dtype=str))) \
                    .combine_first(df.get("locus_tag", pd.Series(dtype=str))) \
                    .fillna("")

    # 4. Build tracks
    bounds = v.TrackBounds(start=start_bp, end=end_bp)
    stack = v.TrackStack(width=1200)

    for rep_name, group_df in df.groupby("contig"):
        features = []
        for row in group_df.to_dict("records"):
            meta = {}
            if include_metadata:
                meta = {
                    str(k): v for k, v in row.items()
                    if pd.notna(v) and not str(k).startswith("_")
                }

            features.append(v.Feature(
                start=int(row["begin"]),
                end=int(row["end"]),
                shape=row["_shape"],
                label=row["_label"],
                metadata=meta,
            ))

        track_meta = {}
        if include_metadata:
            track_meta = {
                "Replicon": str(rep_name),
                "Start": start_bp,
                "End": end_bp,
                "Length": end_bp - start_bp,
                "Feature count": len(features),
            }

        stack.add_track(v.FeatureTrack(
            name=f"Annotations ({rep_name})",
            features=features,
            label_pos=v.LabelPosition.LEFT,
            extent=(start_bp, end_bp),
            metadata=track_meta,
        ))

    svg_string = stack.render(bounds)
    ctx.materialize_content(
        svg_string,
        data_type="svg",
        name=f"RegionFeatures_{replicon or 'All'}_{start_bp}_{end_bp}",
        extension="svg",
    )
