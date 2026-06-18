import lore.core.dsl as lore
from typing import Iterator

class Inputs:
    proteins = lore.ArtifactInput(
        label="Protein FASTA",
        accepted_data=["fasta", "protein_fasta"],
        select="multiple",
        load_as="adapted_stream",
    )
    
    indices = lore.ValueInput(
        list[str],
        label="Index",
        description="Defines how the proteins are subsequenced. Some examples: [:500] indexes the first 500aa; [-500:] the last 500aa; [:-500] all but the last 500aa; [100:200] residues 101-200",
        examples=["[:500], [-500:]"],
    )

class Outputs:
    fasta = lore.TaskOutput(
        data_type="protein_fasta",
        label="Protein FASTA",
        is_primary=True,
    )

def parse_index(s):
    """
    Parses an index string into a list form which can be used to index from a list/string as such:
    idx = parse_slice(s)
    my_list(idx)
    """
    s = s.strip("[]")
    parts = [int(p) if p else None for p in s.split(":")]
    return slice(*parts)

@lore.task(
    "aleyssu.fasta_subsequence",
    name="Protein FASTA Subsequence",
    inputs=Inputs,
    outputs=Outputs,
    icon="🗏",
    preview_mode="live",
)
def fasta_subsequence(
    ctx: lore.ExecutionContext,
    proteins: list[dict],
    indices: list[str],
):
    """Picks out a defined subsequence from the proteins in a given FASTA and produces a new FASTA with just the resulting subsequences."""
    if len(indices) == 0:
        indices = ["[:]"]
    parsed_indices = []
    for idx in indices:
        parsed_indices.append(parse_index(idx))
        
    out_path = ctx.get_temp_path("subsequenced_fasta.faa")
    index_description = f"(Subset of residues indexed by {", ".join(indices)})"
        
    with open(out_path, "w") as f:
        for protein in proteins:
            seq = protein["protein_sequence"]
            seq_fragments = []
            for idx in parsed_indices:
                seq_fragments.append(seq[idx])
            f.write(f">{protein["protein_accession"]} {protein["protein_description"]} {index_description}\n{"".join(seq_fragments)}\n")
        
    ctx.materialize_file(
        output_key="fasta",
        source_path=out_path,
    )