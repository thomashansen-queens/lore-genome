"""
EFetch task for querying NCBI's Gene database
https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EFetch

Defaults to Identical Protein Groups (IPG) database records, which contain
various linked metadata for a given protein accession (assembly, nucleotide,
taxonomy, etc.).)
"""
from enum import StrEnum
import lore
from time import sleep
from .entrez_client import Retmode, entrez_client
from .config import retry


# --- Parsers for returned data ---

def _parse_standard(raw_text: str, is_first_chunk: bool) -> list[str]:
    """
    Simple records as a block
    """
    cleaned = raw_text.strip()
    return [cleaned] if cleaned else []

# --- Task definition ---

class EfetchGeneInputs:
    """Inputs for EFetch Gene task"""
    uid = lore.ArtifactInput(
        accepted_data=["accession", "nucleotide_accession", "Nucleotide Accession"],
        select="multiple",
        load_as="adapted",
        label="Gene accession(s)",
        description="Gene accessions to query the NCBI Gene database.",
    )

class EfetchGeneOutputs:
    """Outputs for EFetch Gene task"""
    nucleotide_fasta = lore.TaskOutput(
        data_type="nucleotide_fasta",
        label="Nucleotide FASTA",
        description="FASTA sequences retrieved from NCBI for the given nucleotide accession(s).",
    )


@lore.task(
    "ncbi.entrez.efetch_gene",
    name="NCBI Entrez EFetch Gene",
    inputs=EfetchGeneInputs,
    outputs=EfetchGeneOutputs,
    description="Query NCBI's Gene database for given gene accessions.",
    preview_mode="full",
)
def efetch_gene(
    ctx: lore.ExecutionContext,
    uid: list[str],
):
    """
    Query NCBI's Gene database for given gene accessions.
    """
    config = ctx.get_config("ncbi").model_dump() if ctx.get_config("ncbi") else {}
    api_key = config.get("api_key")
    email = config.get("email")

    clean_uids = [u.strip() for u in uid if u.strip()]
    if not clean_uids:
        raise ValueError("No valid accessions provided.")

    # 1. Chunking logic (NCBI limits EFetch POSTs to avoid timeouts)
    chunk_size = 200
    uid_chunks = [clean_uids[i:i + chunk_size] for i in range(0, len(clean_uids), chunk_size)]

    accumulated_data = []

    @retry(tries=3, delay=2, default_logger=ctx.logger)
    def _execute_efetch(chunk):
        with entrez_client(api_key=api_key, email=email, ret="text") as client:
            data_payload = {
                "db": "nuccore",
                "id": ",".join(chunk),
                "retmode": "text",  # xml by default, but this returns TSV
                "rettype": "fasta",
            }

            response = client.post(
                "efetch.fcgi",
                data=data_payload,
                timeout=60.0,
            )
            response.raise_for_status()
            return response.text

    for i, chunk in enumerate(uid_chunks):
        ctx.logger.info(f"Fetching chunk {i+1}/{len(uid_chunks)} ({len(chunk)} accessions) as FASTA...")
        raw_text = _execute_efetch(chunk)

        is_first = (i == 0)

        parsed = _parse_standard(raw_text, is_first)
        accumulated_data.extend(parsed)
        join_char = "\n\n"  # double newline separates e.g. FASTA/XML records

        # Be a good citizen of the NCBI API
        sleep(0.2)  # 200 ms delay between requests, 10 requests per second limit

    if not accumulated_data:
        ctx.logger.warning(f"No FASTA results found for the provided accessions.")
        return ctx.materialize_content("", output_key="nucleotide_fasta", extension="fasta")

    final_content = join_char.join(accumulated_data) + "\n"  # Ensure final newline

    ctx.logger.info(f"Successfully fetched {len(accumulated_data) - 1} rows of FASTA.")

    return ctx.materialize_content(
        content=final_content,
        output_key="nucleotide_fasta",
        name="nucleotide_fasta",
        extension="fasta",
        data_type="nucleotide_fasta",
    )
