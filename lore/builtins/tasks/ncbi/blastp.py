"""
A task to run BLASTP using NCBI's web API.

https://blast.ncbi.nlm.nih.gov/doc/blast-help/developerinfo.html
https://blast.ncbi.nlm.nih.gov/doc/blast-help/urlapi.html
"""
from enum import StrEnum
from time import sleep
import httpx
import lore
import re
from .config import retry


BLAST_URL = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"


class BlastDb(StrEnum):
    """
    Enum for supported BLAST databases.
    """
    NR = "nr"
    SWISSPROT = "swissprot"
    REFSEQ_PROTEIN = "refseq_protein"


class BlastProgram(StrEnum):
    """
    Enum for supported BLAST programs.
    """
    BLASTN = "blastn"
    BLASTP = "blastp"
    BLASTX = "blastx"


class BlastpInputs:
    """
    Input parameters for the BLASTP task.
    """
    query = lore.ArtifactInput(
        description="Accession, GI, or FASTA sequence to search against the database.",
        accepted_data=["fasta, protein_fasta, nucleotide_fasta", "protein_accession", "gene_accession"],
        load_as="adapted",
        label="Query",
    )
    database = lore.ValueInput(
        BlastDb,
        label="Database",
        description="The protein database to search against (e.g., 'nr', 'swissprot').",
    )
    program = lore.ValueInput(
        BlastProgram,
        label="Program",
        description="The BLAST program to use.",
    )
    organism = lore.ValueInput(
        str,
        default="",
        label="Organism",
        description="Optional organism filter for the search.",
    )
    expect = lore.ValueInput(
        float,
        default=0.01,
        min=0,
        label="E-value threshold",
        description="The E-value threshold for reporting matches.",
    )
    hitlist_size = lore.ValueInput(
        int,
        default=5000,
        min=1,
        max=5000,
        label="Hitlist Size",
        widget="slider",
        description="The maximum number of hits to return.",
    )


class BlastpOutputs:
    """
    Output parameters for the BLASTP task.
    """
    results = lore.TaskOutput(
        description="BLASTP results in XML format.",
        data_type="blast_results",
        label="Results",
    )


@lore.task(
    "ncbi.blast",
    inputs=BlastpInputs,
    outputs=BlastpOutputs,
    description="Run BLAST using NCBI's web API.",
    preview_mode="dry_run",
)
def blast_handler(
    ctx: lore.ExecutionContext,
    program: BlastProgram,
    database: BlastDb,
    query: list[str],
    organism: str = "",
    expect: float = 0.01,
    hitlist_size: int = 5000,
):
    """
    Handler function for the BLASTP task.
    """
    # 1. Build config
    ncbi_config = ctx.get_config("ncbi")
    email = ncbi_config.email if ncbi_config else None
    if not email:
        ctx.logger.warning("No email set in Settings! Authentication may be rate-limited.")

    entrez_query = f"{organism}[Organism]" if organism else ""
    query_newline = "\n".join(query)

    # 2. === Submit Put request ===
    put_params = {
        "CMD": "Put",
        "PROGRAM": program.value,
        "DATABASE": database.value,
        "QUERY": query_newline,
        "EXPECT": expect,
        "HITLIST_SIZE": hitlist_size,
        "ENTREZ_QUERY": entrez_query,
        "EMAIL": email or "",
        "TOOL": "lore-genome",
    }
    # Do not contact the server more than once every 10 seconds (NCBI rules)
    @retry(tries=3, delay=2, default_logger=ctx.logger)
    def submit_job():
        return httpx.post(BLAST_URL, data=put_params, timeout=60.0)

    ctx.logger.info("Submitting query to NCBI...")
    put_response = submit_job()

    rid_match = re.search(r"^\s*RID\s*=\s*(.+)$", put_response.text, re.MULTILINE)
    rtoe_match = re.search(r"^\s*RTOE\s*=\s*(\d+)$", put_response.text, re.MULTILINE)

    if not rid_match:
        ctx.logger.debug(f"NCBI response text:\n{put_response.text[:1000]}")
        raise ValueError("Failed to parse Request ID (RID) from NCBI response.")

    rid = rid_match.group(1).strip()

    if not rtoe_match:
        ctx.logger.debug(f"NCBI response text:\n{put_response.text[:1000]}")
        raise ValueError("Failed to parse Remaining Time (RTOE) from NCBI response.")
    else:
        rtoe = 60
        ctx.logger.warning("Failed to parse RTOE from NCBI response. Defaulting to 60 seconds.")

    for line in put_response.text.splitlines():
        if line.startswith("RID="):
            rid = line.split("=")[1].strip()
        if line.startswith("RTOE="):
            rtoe = int(line.split("=")[1].strip())
            ctx.logger.info(f"Estimated time to completion: {rtoe} seconds.")

    ctx.logger.info(f"Submitted BLAST job with RID: {rid}. Polling for results...")
    sleep(rtoe)

    # 3. === Poll for results ===
    check_params = {
        "CMD": "Get",
        "FORMAT_OBJECT": "SearchInfo",
        "RID": rid,
    }

    @retry(tries=3, delay=2, default_logger=ctx.logger)
    def check_status():
        return httpx.get(BLAST_URL, params=check_params, timeout=60.0)

    while True:
        ctx.logger.info("Checking job status...")
        status_response = check_status()

        if "Status=WAITING" in status_response.text:
            ctx.logger.info("Job still running. Sleeping for 60 seconds (NCBI policy)...")
            sleep(60)
            continue
        if "Status=FAILED" in status_response.text:
            raise RuntimeError(f"BLAST job {rid} failed on NCBI servers.")
        if "Status=UNKNOWN" in status_response.text:
            raise RuntimeError(f"BLAST job {rid} expired or does not exist.")
        if "Status=READY" in status_response.text:
            if "ThereAreHits=yes" in status_response.text:
                break
            else:
                ctx.logger.warning("Job completed, but no hits were found.")
                return ctx.materialize_content(
                    content='{"BlastOutput2": []}',
                    output_key="results",
                    extension="json",
                )

    # 4. === Retrieve results ===
    ctx.logger.info("Job ready! Downloading tabular results...")

    get_params = {
        "CMD": "Get",
        "FORMAT_TYPE": "JSON2_S",
        "RID": rid,
        "EMAIL": email or "",
        "TOOL": "lore-genome",
    }

    @retry(tries=4, delay=2, default_logger=ctx.logger)
    def fetch_results():
        return httpx.get(BLAST_URL, params=get_params, timeout=60.0)

    final_results = fetch_results()
    json_results = final_results.text

    ctx.logger.info("BLAST results successfully retrieved.")
    return ctx.materialize_content(
        content=json_results,
        output_key="results",
        extension="json",
    )
