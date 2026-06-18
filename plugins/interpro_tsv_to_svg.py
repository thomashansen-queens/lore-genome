"""
Task for visualizing InterproScan tsv results as an svg
"""
from pathlib import Path

from lore import viz as v
import lore.core.dsl as lore

import csv

class InterproVizInputs:
    """Inputs for InterproScan TSV to SVG visualization Task"""
    source_tsvs = lore.ArtifactInput(
        label="InterproScan TSV",
        accepted_data=["interpro_tsv", "tsv"],
        select="multiple",
        load_as="path",
    )
    
    source_fastas = lore.ArtifactInput(
        label="Protein FASTA",
        accepted_data=["protein_fasta", "fasta"],
        select="optional_multiple",
        load_as="path",
        description="The FASTA you inputted into InterProScan. Needed to be able to interactively access the protein sequences from the visualized domains."
    )
    
    undefined_domain_size = lore.ValueInput(
        int,
        label="Undefined Domain Size",
        description="Length of unidentified regions before they are considered as undefined domains.",
        default=20,
    )
    
    residue_interval = lore.ValueInput(
        int,
        label="Residue Interval",
        default=500
    )
    
    horizontal_scale = lore.ValueInput(
        float,
        label="Horizontal Squash/Stretch",
        default=1,
    )
    
    hide_context_in_tooltip = lore.ValueInput(
        bool,
        label="Hide Context in Tooltips",
        description="Hides the identifying labels in the tooltips such as 'Interpro Domain', 'Source Database', etc. leaving behind only the bare values.",
        default=False,
    )


class InterproVizOutputs:
    """Outputs for InterproScan TSV to SVG visualization Task"""
    protein_domain_svg = lore.TaskOutput(
        data_type="svg",
        label="SVG Visualization",
        description="SVG file containing the visualization of InterproScan results.",
        is_primary=True,
    )
    
TSV_HEADER_INDEX = {
    "protein_accession" : 0,
    "protein_length" : 2,
    "source_database" : 3,
    "signature_accession" : 4,
    "signature_domain" : 5,
    "start" : 6,
    "end" : 7,
    "e_value" : 8,
    "interpro_accession" : 11,
    "interpro_domain" : 12,
}
    
# Helpers
def parse_interpro_tsvs(tsv_paths: list[str], undefined_domain_size=20):
    """
    Parses a list of InterproScan TSV files into a list of dictionaries containing domain annotations.
    
    Return format:
    [
        {   
            "protein_accession": str,
            "protein_length": int,
            "domains": [
                {
                    "signature_accession": str, 
                    "signature_domain": str, 
                    "source_database": str,
                    "interpro_accession": str, 
                    "interpro_domain": str, 
                    "start": int, 
                    "end": int, 
                    "e_value": float
                }, 
                ...
            ]
        }, 
        ...
    ]
    """
    
    parsed_result = []
    prev_protein = None
    for path in tsv_paths:
        tsv_path = Path(path)
        with open(tsv_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for row in reader:
                if row[0].startswith("#"):  # Skip comment lines
                    continue
                
                entry = dict()
                
                protein_accession = row[TSV_HEADER_INDEX["protein_accession"]]
                
                # Tooltip text is displayed in the order of these entries
                entry["interpro_domain"] = row[TSV_HEADER_INDEX["interpro_domain"]]
                entry["interpro_accession"] = row[TSV_HEADER_INDEX["interpro_accession"]]
                entry["signature_domain"] = row[TSV_HEADER_INDEX["signature_domain"]]
                entry["signature_accession"] = row[TSV_HEADER_INDEX["signature_accession"]]
                entry["source_database"] = row[TSV_HEADER_INDEX["source_database"]]
                entry["start"] = int(row[TSV_HEADER_INDEX["start"]])
                entry["end"] = int(row[TSV_HEADER_INDEX["end"]])
                entry["total_length"] = entry["end"] - entry["start"] + 1
                entry["e_value"] = row[TSV_HEADER_INDEX["e_value"]]
                
                if prev_protein != protein_accession:
                    protein_length = int(row[TSV_HEADER_INDEX["protein_length"]])
                    if prev_protein is not None:
                        # Identify undefined regions by gaps in between labelled regions
                        sorted_domains = sorted(curr_entry["domains"], key=lambda x: x["start"])
                        
                        curr_entry["domains"] = sorted_domains
                        left = {"end": 0}
                        domain_qty = len(sorted_domains) + 1
                        for i in range(domain_qty):
                            if i == domain_qty - 1:
                                right = {"start": protein_length + 1, "end": protein_length + 1}
                            else:
                                right = sorted_domains[i] 
                            diff = right["start"] - left["end"] + 1
                            if diff >= undefined_domain_size:
                                sorted_domains.append({
                                    "domain": "UNDEFINED REGION", 
                                    "start": left["end"] + 1,
                                    "end": right["start"] - 1,
                                    "total_length": right["start"] - left["end"] - 1
                                })
                            if left["end"] < right["end"]:
                                left = right
                            
                            
                        parsed_result.append(curr_entry)
                    
                    prev_protein = protein_accession
                    curr_entry = {
                        "protein_accession": protein_accession, 
                        "protein_length": protein_length, 
                        "domains": []
                    }
                
                curr_entry["domains"].append(entry)
    
    return parsed_result

def parse_fastas(fasta_paths: list[str]):
    '''Parses a list of fasta files into a dictionary containing {accession: sequence} key-value pairs.'''
    proteins = dict()
    for path in fasta_paths:
        fasta_path = Path(path)
        with open(fasta_path, "r") as f:
            line = f.readline()
            accession = None
            sequence_fragments = []
            while line:
                if line[0] == "#":
                    continue
                elif line[0] == ">":
                    if accession:
                        proteins[accession] = "".join(sequence_fragments)
                    sequence_fragments = []
                    accession = line[1:].split()[0]
                else:
                    sequence_fragments.append(line)
                line = f.readline() 
        if accession:
            proteins[accession] = "".join(sequence_fragments)
    return proteins
    
# --- Main Task ---

@lore.task(
    'interpro.tsv_to_svg',
    inputs=InterproVizInputs,
    outputs=InterproVizOutputs,
    name="InterproScan TSV to SVG",
    category="Visualization",
    icon="🐈︎",
    preview_mode="full"
)
def interpro_viz_handler(
    ctx: lore.ExecutionContext,
    source_tsvs: list[str],
    undefined_domain_size: int,
    residue_interval: int,
    horizontal_scale: float,
    hide_context_in_tooltip: bool,
    source_fastas: list[str],
):
    """Visualizes the TSV output of InterProScan, allowing for multiple proteins and their domains to be visually compared and specific domain sequences to be extracted for further analysis."""
    parsed_data = parse_interpro_tsvs(source_tsvs, undefined_domain_size)
    proteins = parse_fastas(source_fastas)
    
    config = SVG_CONFIG.copy()
    row_height = config["row_height"]
    canvas_height = (len(parsed_data) * row_height) + config["vert_margin"] * 2
            
    # Determine overall span
    lengths = [x["protein_length"] for x in parsed_data]
    longest_length = max(lengths)
    global_max = ((longest_length // residue_interval) * residue_interval + (residue_interval * int(longest_length % residue_interval != 0))) * horizontal_scale
    
    canvas = v.SvgCanvas(width=config["canvas_padding_w"] + global_max, height=canvas_height)
    
    top_y = config["vert_margin"]
    track_min_x = config["label_margin"]
    
    # Prepare a track row for each protein
    idx = 0
    for protein in parsed_data:
        protein_accession, protein_length, domains = protein["protein_accession"], protein["protein_length"], protein["domains"]
        track_group = v.SvgGroup(classes=["protein-container"])
        
        # Draw the backbone line for the protein
        track_max_x = protein_length * horizontal_scale + track_min_x
        
        y_center = top_y + row_height
        top_y = y_center
        
        tracks = [{"domain_ranges": [], "y_center":y_center}]
        track_count = 1
        
        track_group.add(v.SvgLine(
            x1=track_min_x, y1=y_center, x2=track_max_x, y2=y_center,
            style=v.SvgStyle(stroke=config["color_backbone"], stroke_width=2.0),
        ))
        
        protein_label = v.SvgGroup(classes=["protein-label", "parents-tooltip"])
        
        # Draw the protein accession to the left of the track
        protein_label.add(v.SvgText(
            x=config["label_margin"] - 15,
            y=y_center + 4,  # eyeball centering
            text=protein_accession,
            style=v.SvgStyle(
                text_anchor="end",
                font_size=config["font_size"],
                font_family=config["font_family"],
            ),
            classes=["has-tooltip", "protein-accession"],
            data={
                "title": f"Total Length: {protein_length}" + ("\nProtein sequence not found in provided FASTA" if not protein_accession in proteins.keys() else ""),
                "start": 1,
                "end": protein_length,
            }
        ))
        
        if protein_accession in proteins.keys():
            protein_label.add(v.SvgGroup(classes=["copy-text"], data={
                "title": proteins[protein_accession]
            }))
        
        track_group.add(protein_label)
        
        # Draw polygons for each protein domain
        for domain in domains:
            i = -1
            while i < len(tracks):
                i += 1
                track = tracks[i]
                start: int = domain["start"] * horizontal_scale + track_min_x
                end: int = domain["end"] * horizontal_scale + track_min_x
                
                # Collision checking
                collision_detected = False
                for domain_range in track["domain_ranges"]:
                    if (domain["end"] > domain_range[0] and domain["end"] < domain_range[1]) or (domain["start"] > domain_range[0] and domain["start"] < domain_range[1]):
                        if i == len(tracks) - 1:
                            y_center = top_y + config["same_protein_margin"]
                            tracks.append({"domain_ranges": [], "y_center":y_center})
                            top_y = y_center
                            track_count += 1
                        collision_detected = True
                        break
                if collision_detected:
                    continue
                    
                track["domain_ranges"].append((domain["start"], domain["end"]))
                    
                y_center = track["y_center"]
                
                domain_group = v.SvgGroup(classes=["domain-container", "parents-tooltip"])
                
                arrow_h = row_height * 0.5  # thickness
                head_w = min(10.0, abs(end - start))  # arrowhead can't exceed gene length

                y0 = y_center - (arrow_h / 2)
                y1 = y_center + (arrow_h / 2)
                
                pts = [
                    (start, y0),
                    (end, y0),
                    (end, y1),
                    (start, y1),
                ]
                
                if "domain" in domain.keys():
                    fill_color = config["color_anchor_text"]
                else:
                    fill_color = config["color_context_fill"]
                
                stroke_color = config["color_context_stroke"]
                
                # Format the tooltip text which is displayed when clicking on a domain
                if hide_context_in_tooltip:
                    tooltip_text = "\n".join([f"{value}" for value in domain.values()])
                else:
                    tooltip_text = "\n".join([f"{key.replace("_", " ").title()}: {value}" for key, value in domain.items()])
                
                arrow = v.SvgPolygon(
                    points=pts,
                    classes=["protein-domain", "has-tooltip"],
                    data={
                        "title": tooltip_text,
                        "start": domain["start"],
                        "end": domain["end"],
                    },
                    style=v.SvgStyle(fill=fill_color, stroke=stroke_color, stroke_width=1.0),
                )
                domain_group.add(arrow)
                
                # Domain text label (if space allows)
                def _trim_label(text: str, width_px: int) -> str:
                    """Trims labels to fit if possible"""
                    char_width = config["font_size"] * 0.6
                    max_chars = int(width_px / char_width)
                    if max_chars < 3:
                        return ""
                    elif len(text) > max_chars:
                        return text[:max_chars - 1] + "…"
                    else:
                        return text
                
                if "domain" in domain.keys():
                    domain_name = domain["domain"]
                else:
                    domain_name = domain["interpro_domain"] if domain["interpro_domain"] != "-" else domain["signature_domain"]
                text_label = _trim_label(domain_name, abs(int(end - start)))
                if text_label:
                    # Use white text on the dark anchor background for readability
                    text_color = config["color_context_text"]
                    label = v.SvgText(
                        x=(start + end) / 2,
                        y=y_center + (config["font_size"] * 0.35), # True vertical centering for text
                        text=text_label,
                        style=v.SvgStyle(
                            text_anchor="middle",
                            fill=text_color,
                            font_size=config["font_size"] * 0.8,
                            font_family=config["font_family"],
                        ),
                    )
                    domain_group.add(label)
                track_group.add(domain_group)
                break
            
        canvas.add(track_group)

        idx += track_count
        
    canvas.height = top_y + config["vert_margin"] * 2
    
    # Draw marking line
    marker = v.SvgGroup(classes=["marker"])
    
    # Marker background
    marker.add(v.SvgRect(x=0, y=-50, width=global_max + config["label_margin"], height=config["marker_margin"] + 50, style=v.SvgStyle(fill="#FFFFFF", opacity=0.9)))
    
    # Marker main line
    marker.add(v.SvgRect(x=track_min_x, y=config["marker_margin"], width=global_max, height=config["marker_thickness"], style=v.SvgStyle(fill=config["color_backbone"])))
    
    # Marker segments
    for x in range(0, longest_length + residue_interval, residue_interval):
        line_x = x * horizontal_scale + track_min_x
        marker.add(v.SvgLine(x1=line_x, x2=line_x, y1=config["marker_margin"]+5, y2=config["marker_margin"]+10, style=v.SvgStyle(stroke=config["color_backbone"], stroke_width=2.0),))
        marker.add(v.SvgText(
            x=line_x,
            y=config["marker_margin"] - 2,  # eyeball centering
            text=str(x),
            style=v.SvgStyle(
                text_anchor="middle",
                font_size=config["font_size"],
                font_family=config["font_family"],
            ),
        ))
    
    
    canvas.add(marker)
    
    svg_str = canvas.render()
    
    ctx.materialize_content(
        svg_str,
        name="protein_domain_view",
        output_key="protein_domain_svg",
        data_type="svg",
        extension="svg",
    )

SVG_CONFIG = {
    "canvas_width": 1200,
    "canvas_padding_w": 200,
    "row_height": 40,
    "label_margin": 150,
    "right_margin": 50,
    "marker_margin": 25,
    "marker_thickness": 3,
    "vert_margin": 25,
    "same_protein_margin": 25,  # Space between tracks for the same protein
    "arrow_thickness_ratio": 0.5,
    "max_arrowhead_width": 10.0,
    "color_backbone": "#64748B",
    "color_anchor_fill": "#318686",
    "color_anchor_stroke": "#2A6B6B",
    "color_context_fill": "#ADD8E6",
    "color_context_stroke": "#8BB4C2",
    "color_anchor_text": "#FAFFFF",
    "color_context_text": "#333333",
    "font_family": "monospace",
    "font_size": 12,
}