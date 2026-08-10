"""
Visualize InterProScan results.
"""
import json

import lore
import lore.viz as v


class IpsTracksInput:
    """Input data for visualizing InterProScan results."""
    interproscan_results = lore.ArtifactInput(
        label="InterProScan results",
        accepted_data=["interproscan_json"],
        select="single",
        load_as="raw",
    )
    protein_accessions = lore.ArtifactInput(
        label="Protein accessions",
        description="The protein accession(s) for the gene(s) of interest",
        accepted_data=["protein_accession"],
        select="optional_multiple",
        default=None,
        load_as="adapted",
    )
    max_proteins = lore.ValueInput(
        int,
        label="Max proteins to draw",
        description="If no specific accessions are provided, draw this many proteins from the top of the file.",
        default=50,
        min=1, max=500,
        widget="slider",
    )
    e_threshold = lore.ValueInput(
        float | None,
        label="E-value threshold",
        description="Only show annotations with an E-value below this threshold.",
        default=1e-5,
    )


class IpsTracksOutput:
    annotations_svg = lore.TaskOutput(
        data_type="svg",
        label="Annotated protein domains",
    )


@lore.task(
    key="interproscan.visualize_scan",
    name="Visualize InterProScan results",
    description="Visualize protein domain annotations from InterProScan results",
    inputs=IpsTracksInput,
    outputs=IpsTracksOutput,
    preview_mode="live",
)
def visualize_interproscan_results(
    ctx: lore.ExecutionContext,
    interproscan_results: list[dict],
    max_proteins: int,
    protein_accessions: list[str] | None,
    e_threshold: float | None,
):
    """
    Visualize InterProScan results for the given protein accessions.
    """
    # 1. Process inputs
    records = interproscan_results[0].get("results", [])
    if not records:
        ctx.logger.error("No records found in InterProScan results.")
        raise ValueError("No records found in InterProScan results.")

    # 2. Build a quick lookup dictionary by xref ID
    record_map = {}
    for r in records:
        for xref in r.get("xref", []):
            record_map[xref.get("id")] = r

    # 3. Draw tracks for each requested protein accession
    accs_to_draw = protein_accessions or list(record_map.keys())[:max_proteins]

    stack = v.TrackStack(width=1200)
    max_length_seen = 0

    for accession in accs_to_draw:
        # Strict match first, then allow for missing version suffix
        record = record_map.get(accession)
        if record is None:
            item = next(
                (
                    (k, v)
                    for k, v in record_map.items()
                    if k.startswith(accession + ".")
                ),
                None
            )
            if item:
                accession, record = item

        if not record:
            ctx.logger.warning(f"Accession '{accession}' not found in InterProScan results.")
            continue

        # Deal with InterProScan naming conventions
        name = record.get("xref", [{}])[0].get("name", "")
        name = name.removeprefix(accession).strip()
        # disp = name[:47] + "..." if len(name) > 50 else name
        parts = name.split(" [", 1)
        if len(parts) > 1:
            parts[1] = "[" + parts[1]  # Add back the opening bracket for display
        disp = "\n".join(p[:49] + "…" if len(p) > 50 else p for p in parts)

        seq = record.get("sequence", "")
        seq_len = len(seq)
        max_length_seen = max(max_length_seen, seq_len)

        # Draw the sequence baseline
        seq_meta = {
            "Accession": accession,
            "Name": name,
            "Length": seq_len,
        }
        stack.add_track(v.SequenceTrack(
            sequence=seq,
            name=disp,
            metadata=seq_meta,
        ))

        features = []
        for match in record.get("matches", []):
            sig = match.get("signature", {})
            db_name = sig.get("signatureLibraryRelease", {}).get("library", "UNKNOWN")
            domain_name = sig.get("name") or sig.get("accession") or "Unknown Domain"

            for loc in match.get("locations", []):
                e_val = loc.get("evalue")

                # Filter by E-value (Note: some tools like SignalP/Coils don't return e-values)
                if e_threshold is not None and e_val is not None:
                    if e_val > e_threshold:
                        continue

                # Build Tooltip Data
                annot_meta = {
                    "Domain": domain_name,
                    "Database": db_name,
                    "Accession": sig.get("accession", "N/A"),
                    "Start": loc.get('start'),
                    "End": loc.get('end'),
                }
                if e_val is not None:
                    annot_meta["E-value"] = f"{e_val:.2e}"

                # Add feature
                features.append(v.Feature(
                    start=loc.get("start", 0),
                    end=loc.get("end", 0),
                    label=domain_name,
                    fill=_color_by_db(db_name),
                    stroke="none",
                    metadata=annot_meta,
                ))

        # Add Pileup track
        track_meta = {
            "Accession": accession,
            "Total annotations": len(features),
        }
        stack.add_track(v.PileupTrack(
            name="Annotations",
            features=features,
            packing_gap=2.0, # Slight gap so distinct domains don't visually merge
            font_size=8,  # pyright: ignore[reportCallIssue]
            lane_height=10.0,
            metadata=track_meta,
        ))

    if max_length_seen == 0:
        raise ValueError("No matching sequences found to visualize.")

    # 4. Render the SVG
    bounds = v.TrackBounds(start=0, end=max_length_seen)
    svg_string = stack.render(bounds)

    # 5. Materialize Output
    ctx.materialize_content(
        svg_string,
        data_type="svg",
        name=f"InterProScan_Domains_{len(accs_to_draw)}_proteins",
        label="Protein Domain SVG",
        extension="svg",
    )


def _color_by_db(db_name: str) -> str:
    """Assign consistent, distinct categorical colors to all InterProScan databases."""
    palette = {
        "CDD": "#f59e0b",             # Amber
        "COILS": "#cbd5e1",           # Slate Light (Structural)
        "GENE3D": "#8b5cf6",          # Purple
        "HAMAP": "#14b8a6",           # Teal
        "MOBIDBLITE": "#64748b",      # Slate Dark (Disordered)
        "PANTHER": "#10b981",         # Emerald
        "PFAM": "#3b82f6",            # Blue
        "PIRSF": "#6366f1",           # Indigo
        "PRINTS": "#84cc16",          # Lime
        "PROSITEPATTERNS": "#f43f5e", # Rose
        "PROSITEPROFILES": "#ef4444", # Red
        "SFLD": "#eab308",            # Yellow
        "SMART": "#06b6d4",           # Cyan
        "SUPERFAMILY": "#ec4899",     # Pink
        "NCBIFAM": "#0ea5e9",         # Sky
    }

    # Strip spaces and uppercase to ensure matching (e.g., "MobiDBLite" -> "MOBIDBLITE")
    db_clean = db_name.upper().replace(" ", "")

    # Fallback to a safe, neutral gray for any unrecognized custom databases
    return palette.get(db_clean, "#9ca3af")
