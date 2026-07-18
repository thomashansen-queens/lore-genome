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
from typing import Any
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
        description="Sequence to search against the database. Accessions and GI numbers are not supported by this plugin.",
        accepted_data=["fasta", "protein_fasta", "nucleotide_fasta"],
        load_as="adapted",
        select="multiple",
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


def _extract_fasta_data(record: dict | str) -> tuple[str, str]:
    """
    Extracts the header and sequence from an adapted FASTA dict or raw string.
    Returns a tuple of (header, sequence).
    """
    # Fallback for raw text blocks
    if isinstance(record, str):
        record_str = record.strip()
        if not record_str.startswith(">"):
            return "", ""
        lines = record_str.splitlines()
        header = lines[0]
        seq = "".join(l.strip() for l in lines[1:])
        return header, seq

    # Duck-typing for Adapted Dictionaries
    if isinstance(record, dict):
        # 1. Sniff the Accession/ID
        acc = next((str(v) for k, v in record.items() if k.lower().endswith(("accession", "id", "acc", "name"))), "Unknown")

        # 2. Sniff the Description (optional)
        desc = next((str(v) for k, v in record.items() if k.lower().endswith(("description", "desc"))), "")

        # 3. Sniff the Sequence (catches "seq", "sequence", "protein_sequence", etc.)
        seq = next((str(v) for k, v in record.items() if k.lower().endswith(("seq", "sequence"))), "")

        if not seq:
            raise ValueError(f"Could not locate sequence data in record keys: {list(record.keys())}")

        header = acc if acc.startswith(">") else f">{acc}"
        header += f" {desc}" if desc else ""

        return header, seq

    return "", ""


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
    query: list[Any],
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
        ctx.logger.debug("No NCBI email set in Settings!")

    entrez_query = f"{organism}[Organism]" if organism else ""

    # 2. Parse query input
    parsed_queries = []

    for i, val in enumerate(query):
        header, seq = _extract_fasta_data(val)
        # A. FASTA records
        if seq:
            parsed_queries.append(f"{header}\n{seq}")
        # B. Manual sequence strings
        elif isinstance(val, str) and not val.startswith(">"):
            parsed_queries.append(f">input_{i}\n{val.strip()}")

    if not parsed_queries:
        raise ValueError("No valid query sequences found.")

    # 2. === Submit Put request ===
    ctx.logger.info(f"Preparing to submit {len(parsed_queries)} sequence(s) to NCBI BLAST...")
    ctx.logger.info(f"First query header: {parsed_queries[0].splitlines()[0]}")
    ctx.logger.info(f"First query seq   : {parsed_queries[0].splitlines()[1][:80]}")
    ctx.logger.info(f"First query len   : {len(parsed_queries[0].splitlines()[1])}")

    put_params = {
        "CMD": "Put",
        "PROGRAM": program.value,
        "DATABASE": database.value,
        "QUERY": "\n".join(parsed_queries),
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
    response_text = put_response.text

    # Get the request ID (RID) and request time-of-execution (RTOE) from response HTML
    rid_match = re.search(r"^\s*RID\s*=\s*([\w-]+)$", response_text, re.MULTILINE)
    rtoe_match = re.search(r"^\s*RTOE\s*=\s*(\d+)$", response_text, re.MULTILINE)

    if not rid_match:
        ctx.logger.debug(f"NCBI response text:\n{response_text[:1000]}")
        raise ValueError("Failed to parse Request ID (RID) from NCBI response.")

    rid = rid_match.group(1).strip()
    ctx.logger.info(f"Successfully secured BLAST RID: {rid}")

    if rtoe_match:
        rtoe = int(rtoe_match.group(1).strip())
        ctx.logger.info(f"NCBI estimates {rtoe} seconds until results will be ready.")
    else:
        rtoe = 60
        ctx.logger.warning(f"Failed to parse RTOE from NCBI response. Defaulting to {rtoe} seconds.")

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
            snooze = 60
            ctx.logger.info(f"Job still running. Sleeping for {snooze} seconds (NCBI policy)...")
            sleep(snooze)
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
    }

    @retry(tries=4, delay=2, default_logger=ctx.logger)
    def fetch_results():
        response = httpx.get(BLAST_URL, params=get_params, timeout=60.0)
        response.raise_for_status()
        return response

    final_results = fetch_results()
    json_results = final_results.text

    ctx.logger.info("BLAST results successfully retrieved.")
    return ctx.materialize_content(
        content=json_results,
        output_key="results",
        extension="json",
    )
