import lore.core.dsl as lore
import re

from time import sleep

DEFAULT_ACCESSION_HEADER_NAMES = ["mmseqs_cluster_id", "protein_accession", "accession"]
DEFAULT_PROTEIN_HEADER_NAMES = ["protein_sequence", "sequence"]

class Inputs:
    files = lore.ArtifactInput(
        label="HHPred Raw Files",
        accepted_data=["hhr"],
        select="multiple",
        load_as="path",
    )
    
    top_matches = lore.ValueInput(
        int,
        label="Max top matches",
        default=3
    )
    
    keep_probability = lore.ValueInput(
        float,
        lable="Override Probability",
        default="0.95",
        description="Probability threshold to ignore 'Max top matches' and keep the domain anyway."
    )

class Outputs:
    tsv = lore.TaskOutput(
        data_type="hhpred_tsv",
        label="HHPred TSV",
        description="A trimmed-down and merged tabular format of one or more HHPred raw outputs.",
        is_primary=True,
    )

TARGET_HEADERS = (
    "Prob",
    "E-value",
    "Score",
    "Query-HMM",
    "Template-HMM"
)

OUT_HEADERS = (
    "Protein",
    "Homology Accession",
    "Homology Name",
    "Start",
    "End",
    "Homo Start",
    "Homo End",
    "Prob",
    "E-value",
    "Score",
)

@lore.task(
    "aleyssu.hhpred_to_tsv",
    name="HHPred to TSV",
    inputs=Inputs,
    outputs=Outputs,
    icon="🗏",
    preview_mode="live",
)
def hhpred_to_tsv(
    ctx: lore.ExecutionContext,
    files: list[str],
    top_matches: int,
    keep_probability: float
):
    """Produces a trimmed-down and merged TSV file of one or more HHPred raw outputs which can be passed into the protein domain visualizer."""
    out_path = ctx.get_temp_path("hhpred_tsv.tsv")
    
    with open(out_path, "w") as out_f:
        print("\t".join(OUT_HEADERS), file=out_f)
        for file_path in files:
            results = []
            with open(file_path, "r") as f:
                line = f.readline().split(maxsplit=1)
                if not line[0] == "Query":
                    raise ValueError(f"Expected \"Query\" at the beginning of the file {file_path}. Are you sure it's formatted correctly?")
                query_name = line[1].split(maxsplit=1)[0].strip()
                
                # Seek the table section
                line = f.readline()
                while True:
                    line = f.readline()
                    if not line:
                        raise ValueError(f"Could not find the table section of the file {file_path} (expected to find a header starting with \"No\")")
                    else:  
                        temp = line.split(maxsplit=1)
                        if len(temp) == 0:
                            continue
                        elif line.split()[0] == "No":
                            break
                
                # Get the header names - assumes that the first two headers are "No" and "Hit" and the rest are data entries
                line = line.replace("Query HMM", "Query-HMM")
                line = line.replace("Template HMM", "Template-HMM")
                col_names = line.split()[2:]
                
                # Get the bounding indices of the entries under each header (.hhr files are formatted horribly for parsing)
                header_indices = dict()
                for header in TARGET_HEADERS:
                    start_idx = line.index(header)
                    end_idx = start_idx + len(header)
                    
                    if line[end_idx] == "\n":
                        end_idx += 69420
                    
                    header_indices[header] = (start_idx, end_idx)  
                
                # Read the table entries
                line = f.readline()
                while not line[0] == "\n":
                    result = {"Protein": query_name}                
                    for header in TARGET_HEADERS:
                        idxs = header_indices[header]
                        val = line[idxs[0]: idxs[1]].strip()
                        if header == "Query-HMM":
                            r = re.sub(r"\(.*\)", "", val).strip().split("-")
                            start = r[0]
                            end = r[1]
                            result["Start"] = start
                            result["End"] = end
                        elif header == "Template-HMM":
                            r = re.sub(r"\(.*\)", "", val).strip().split("-")
                            start = r[0]
                            end = r[1]
                            result["Homo Start"] = start
                            result["Homo End"] = end
                        else:
                            result[header] = val
                    line = f.readline()
                    results.append(result)
            
                line = f.readline()
                while line:
                    if line[0:2] == "No":
                        idx = int(line.split()[1]) - 1
                        results[idx]["Homology Accession"], results[idx]["Homology Name"] = f.readline()[1:].strip().split(maxsplit=1)
                    line = f.readline()
                    
            results.sort(key=lambda entry: float(entry["Score"]), reverse=True)
            tracks = []
            for result in results:
                start, end = result["Start"], result["End"]
                free_space_found = False
                for track in tracks:
                    collision_detected = False
                    for _start, _end in track:
                        if (end >= _start and start <= _end):
                            collision_detected = True
                            break
                    if not collision_detected:
                        free_space_found = True
                        track.append((start, end))
                        break
                if not free_space_found:
                    if len(tracks) >= top_matches:
                        if float(result["Prob"]) >= keep_probability * 100:
                            skip_entry = False
                        else:
                            skip_entry = True
                    else: 
                        tracks.append([(start, end)])
                        skip_entry = False
                else:
                    skip_entry = False
                
                if not skip_entry:
                    print("\t".join([result[header] for header in OUT_HEADERS]), file=out_f)

    ctx.materialize_file(
        output_key="tsv",
        source_path=out_path,
    )