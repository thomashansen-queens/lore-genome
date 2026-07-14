import lore.core.dsl as lore
from typing import Iterator
from time import sleep
from contextlib import contextmanager
from importlib.metadata import version
import httpx

URL = "https://www.ncbi.nlm.nih.gov/Structure/bwrpsb/bwrpsb.cgi"

class Inputs:
    proteins = lore.ArtifactInput(
        label="Protein FASTA",
        accepted_data=["fasta", "protein_fasta"],
        select="multiple",
        load_as="adapted_stream",
    )
    
    database = lore.ValueInput(
        str,
        label="Search Database",
        default="cdd",
        options=["cdd", "pfam", "smart", "tigrfam", "cog", "kog"],
    )
    
    e_value = lore.ValueInput(
        float,
        label="Threshold E-Value",
        default=0.01,
    )
    
    data_mode = lore.ValueInput(
        str,
        label="Data Mode",
        description='Determines the entries returned - "rep" for only the highest-scoring hit for each domain, "std" for the highest-scoring hits from each database for each domain, and "full" for all the hits.',
        default="std",
        options=["rep", "std", "full"],
    )
    
    include_domain_definition = lore.ValueInput(
        bool,
        label="Include Domain Description",
        description="Determines if the domain descriptions will be included in the final table.",
        default=True,
    )

class Outputs:
    ncbi_cd_search_tsv = lore.TaskOutput(
        data_type="ncbi_cd_search_tsv",
        label="NCBI CD-Search TSV",
        is_primary=True,
    )

@contextmanager
def cd_search_client(api_key: str | None = None, timeout: float = 60.0):
    """
    Create a configured httpx client for the NCBI Datasets API.
    event_hooks allows us to raise exceptions on HTTP errors, rather than checking
    {"success": false, "error": {...}} in the JSON response.
    """
    headers = {
        "User-Agent": f"lore-genome/{version('lore-genome')}",
    }
    params = {}
    if api_key:
        params["api-key"] = api_key

    def raise_on_4xx_5xx(response: httpx.Response):
        response.raise_for_status()

    with httpx.Client(
        headers=headers,
        params=params,
        timeout=httpx.Timeout(connect=5.0, read=timeout, write=timeout, pool=timeout),
        event_hooks={"response": [raise_on_4xx_5xx]},
        verify=False,
    ) as client:
        yield client
        
@lore.memoize(prefix="ncbi_cd_search")
def cd_search(
    ctx: lore.ExecutionContext,
    fasta_str: str,
    database: str,
    data_mode: str,
    e_value: float,
    include_domain_definition: bool,
    protein_lengths: dict,
    ) -> list[str]:
    """Returns a list of tsv format strings from NCBI's CD search for a given FASTA format string. Includes the header."""
    with cd_search_client() as client:
        payload = {
            "queries": fasta_str,
            "useid1": "true",
            "tdata": "hits",
            "cddefl": "true",
            "smode": "live",
            "db": database,
            "dmode": data_mode,
            "evalue": e_value,
            "qdelf": "true" if include_domain_definition else "false",
        }
        response = client.post(URL, data=payload)
        
        cdsid = response.text.split()[5]
        result = []
        
        while True:
            with client.stream(
                "GET",
                URL,
                params={"cdsid": cdsid}
            ) as response:
                response.raise_for_status()
                
                # tsv_comment_fragments = []
                iter = response.iter_lines()
                for i in range(4):
                    row = next(iter)
                    # tsv_comment_fragments.append(row)
                if row.split(maxsplit=2)[1] == '3':
                    sleep(5)
                    continue
                elif row.split(maxsplit=2)[1] == '0':
                    # Skip to where the header column is
                    while not row.startswith("Query"):
                        row = next(iter)
                    result.append(f"{row}\tProtein Length") # Header
                    for row in iter:
                        if row.startswith("Q#"):
                            row = row[row.index('>')+1:] # Clean up the accession column
                            acc = row.split(maxsplit=1)[0]
                        result.append(f"{row}\t{protein_lengths[acc]}")
                    break
                else:
                    raise RuntimeError(f"Failed to search through NCBI CD-Search: {response}")
        return result

@lore.task(
    "aleyssu.ncbi_cd_search",
    name="NCBI CD-Search",
    inputs=Inputs,
    outputs=Outputs,
    icon="🗏",
    preview_mode="full",
)
def ncbi_cd_search(
    ctx: lore.ExecutionContext,
    proteins: Iterator,
    database: str,
    e_value: float,
    data_mode: str,
    include_domain_definition: bool,
):
    """Runs the NCBI conserved domain search through the NCBI servers and saves the result in TSV format. Requires internet connection."""
    out_path = ctx.get_temp_path("ncbi_cd_search.tsv")
    
    proteins = proteins.__iter__()
    
    iterating = True
    file_write_mode = "w"
    protein_lengths = {}  # Keep track of protein lengths to append to the final table (NCBI doesn't keep track of this)
    while iterating:
        # Query NCBI in batch sizes of 10
        search_fragments = []
        for _ in range(10):
            protein = next(proteins, None)
            if protein:
                acc = protein["protein_accession"]
                seq = protein["protein_sequence"]
                search_fragments.append(f">{acc}\n{seq}")
                protein_lengths[acc] = len(seq)
            else:
                iterating = False
                break
        result = cd_search(ctx, "\n".join(search_fragments), database, data_mode, e_value, include_domain_definition, protein_lengths)
        with open(out_path, file_write_mode) as f:
            if file_write_mode == 'w':
                file_write_mode = 'a'  # Append the contents to the output in future iterations so we don't overwrite what's already there
                for line in result: print(line, file=f) 
            else:
                for line in result[1:]: print(line, file=f) # Skip the header in future iterations
        ctx.logger.info("Completed one search batch of 10...")
        sleep(0.34)
            
    ctx.materialize_file(
        output_key="ncbi_cd_search_tsv",
        source_path=out_path,
    )