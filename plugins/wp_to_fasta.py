import pandas as pd

import lore.core.dsl as lore
import re

from time import sleep

DEFAULT_ACCESSION_HEADER_NAMES = ["mmseqs_cluster_id", "protein_accession", "accession"]
DEFAULT_PROTEIN_HEADER_NAMES = ["protein_sequence", "sequence"]

class Inputs:
    accessions = lore.ValueInput(
        str,
        label="WP Accessions",
        examples=["WP_322621732.1,WP_001264253.1"],
        default="",
    )
    
    table = lore.ArtifactInput(
        label="Protein Accession and Sequence Table",
        accepted_data=["clustered_summary", "csv", "tsv"],
        select="optional_multiple",
        load_as="adapted",
        default="",
        description="A table containing at least two columns: one with the WP accessions and one with the corresponding protein sequences. The column names or indices can be specified in the next two fields.\nThis can be left empty if 'Index from NCBI' is set to True.",
    )
    
    acc_column = lore.ValueInput(
        str,
        label="Accession Column",
        description="The column name or index where the protein accessions are in the table.",
        default="",
        examples=['0'],
    )
    
    seq_column = lore.ValueInput(
        str,
        label="Sequence Column",
        description="The column name or index where the protein sequences are in the table. Defaults to 'protein_sequence' or 'sequence' if left empty.",
        default="",
        examples=['0'],
    )
    
    index_from_ncbi = lore.ValueInput(
        bool,
        label="Index from NCBI (optional)",
        description="If True, the task will attempt to acquire any missing protein sequences directly from NCBI. This is much slower than indexing from a pre-made table but can be used if you don't have a local table of sequences to index from. Requires internet connection.",
        default=False,
    )

class Outputs:
    fasta = lore.TaskOutput(
        data_type="protein_fasta",
        label="Protein FASTA",
        description="A simplified FASTA file of just WP protein accessions and their associated sequences.",
        is_primary=True,
    )

@lore.task(
    "aleyssu.wp_to_fasta",
    name="WP Accession to FASTA",
    inputs=Inputs,
    outputs=Outputs,
    icon="🗏",
    preview_mode="live",
)
def wp_to_fasta(
    ctx: lore.ExecutionContext,
    accessions: str = "",
    table: list[dict] = [],
    seq_column: str = "",
    acc_column: str = "",
    index_from_ncbi: bool = False,
):
    """Produces a simplified FASTA file given a list of WP accessions and a table to index protein sequences from. Can optionally acquire protein sequences directly from NCBI (slower than indexing from a pre-made table and requires internet)."""
    if index_from_ncbi: 
        from lore.builtins.tasks.ncbi.datasets_client import datasets_client
        import json, zipfile, io
        ncbi_config = ctx.get_config("ncbi")        
        api_key = ncbi_config.api_key if ncbi_config else None
        if not api_key:
            has_api_key = False 
            ctx.logger.warning("No NCBI API key set in Settings! Authentication may be rate-limited.")
        else:
            has_api_key = True
        
    acc_list = re.split(r"[,\s]+", accessions)
    out_path = ctx.get_temp_path("wp_protein.faa")
    acc_seq_map = dict()
    
    if table:
        # Check for sequences under headers named either "protein_sequence" or "sequence"
        if seq_column == "":
            found = False
            for header_name in DEFAULT_PROTEIN_HEADER_NAMES:
                if header_name in table[0].keys():
                    seq_header = header_name
                    found = True
                    break
            if not found:    
                raise ValueError("Could not find the protein sequence column in the provided table - please manually indicate the column under 'Sequence Column (optional)'")
        # If a header name was specified for the protein sequence, try for that
        elif seq_column in table[0].keys():
            seq_header = seq_column
        else:
            try:
                seq_header_idx = int(seq_column)
            except:
                raise ValueError(f"Could not find '{seq_column}' in table header for the Sequence Column.\nAvailable columns are: {", ".join(df.columns)}")
            
            if seq_header_idx >= len(table[0]) or seq_header_idx < 0:
                raise ValueError(f"Column index {seq_header_idx} is out of bounds.")
            else:
                seq_header = list(table[0].keys())[seq_header_idx]
                
        if acc_column == "":
            raise ValueError("Please specify the Accession Column under 'Accession Column' - this column is required to index the protein sequences. Available columns are: " + ", ".join(df.columns))

        # Get accession column
        if acc_column in table[0].keys():
            acc_header = acc_column
        else:
            try:
                acc_header_idx = int(acc_column)
            except:
                raise ValueError(f"Could not find '{acc_column}' in table header for the Accession Column.\nAvailable columns are: {", ".join(table[0].keys())}")
            
            if acc_header_idx >= len(table[0]) or acc_header_idx < 0:
                raise ValueError(f"Column index {acc_header_idx} is out of bounds.")
            else:
                acc_header = list(table[0].keys())[acc_header_idx]
        
        for entry in table:
            acc_seq_map[entry[acc_header]] = f">{entry[acc_header]}\n{entry[seq_header]}"
        
        # Check for matches in the table and keep track of any accessions that were missed so we can optionally pull them from NCBI
        all_matches = {acc for acc in acc_list if acc in acc_seq_map}
        
        missed_matches = set(acc_list) - all_matches
    else:
        missed_matches = set(acc_list)
        
    if len(missed_matches) > 0:
        if not index_from_ncbi:
            raise ValueError(f"Could not find protein sequences for {missed_matches}. To acquire missing sequences, set 'Index from NCBI' to True (note this is slower than indexing from a pre-made table and requires internet connection).")
        else:
            ctx.logger.info(f"Attempting to acquire missing protein sequences for {missed_matches} directly from NCBI...")
            # Pull missing protein sequences from NCBI
            with datasets_client(api_key) as api:
                # Make requests to NCBI in batches of 10
                for i in range(0, len(missed_matches), 10):
                    batch = list(missed_matches)[i:i+10]
                    try:
                        zip_bytes = api.get(f"/protein/accession/{','.join(batch)}/download", params={"include_annotation_type": "FASTA_PROTEIN"}).read()
                        if not zip_bytes:
                            raise Exception("No data returned for the batch of accessions:", batch)

                        # Extract the FASTA file from the zip
                        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                            targets = [f for f in z.namelist() if f.endswith(".faa")]
                            if not targets:
                                raise Exception("No protein FASTA file found in zip for accessions %s", batch)

                            acc = None
                            with z.open(targets[0]) as zip_f:
                                # Add FASTA contents to the acc_seq_map dictionary
                                seq_fragments = []
                                while True:
                                    line = zip_f.readline()
                                    if not line:
                                        break
                                    line = line.decode("utf-8").rstrip()
                                    if not line.startswith(">"):
                                        line = line.replace("\n", "")
                                        seq_fragments.append(line)
                                    else:
                                        if acc:
                                            acc_seq_map[acc] = f"{header}\n{"".join(seq_fragments)}"
                                        header = line
                                        acc = line[1:].split(maxsplit=1)[0]
                                        
                                acc_seq_map[acc] = f"{header}\n{"".join(seq_fragments)}"
                    except:
                        raise Exception("Error fetching the following batch of accessions from NCBI - did you enter an invalid accession?: %s", batch)
                    if has_api_key:
                        sleep(0.1)
                    else:
                        sleep(0.34)
        
    with open(out_path, "w") as f:
        for acc in acc_list:
            # Write the wp accessions and protein sequences into a FASTA file in the form:
            # >Acc1
            # Seq1
            # >Acc2
            # Seq2
            # ...
            print(acc_seq_map[acc], file=f)

    ctx.materialize_file(
        output_key="fasta",
        source_path=out_path,
    )