"""
EFetch task for querying NCBI's Protein database
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


class ProteinFormat(StrEnum):
    """Supported formats for the Protein database"""
    IPG = "IPG Table"
    FASTA = "FASTA Sequence"
    FEATURE_TABLE = "Feature Table"
    NATIVE_XML = "Native XML"
    GBSEQ_XML = "GBSeq XML"

# Map input arg to (rettype, retmode, file extension, data type)
_FORMAT_MAP = {
    ProteinFormat.IPG: ("ipg", "text", "tsv", "ipg_records"),
    ProteinFormat.FASTA: ("fasta", "text", "fasta", "protein_fasta"),
    ProteinFormat.FEATURE_TABLE: ("ft", "text", "txt", "feature_table"),
    ProteinFormat.NATIVE_XML: ("native", "xml", "xml", "native_xml"),
    ProteinFormat.GBSEQ_XML: ("gbseq", "xml", "xml", "gbseq_xml"),
}

# --- Parsers for returned data ---

def _parse_ipg_tsv(raw_text: str, is_first_chunk: bool) -> list[str]:
    """
    Parse the TSV text returned by NCBI EFetch IPG into a list of lines.
    The first line is the header, and subsequent lines are data rows.
    """
    lines = [line for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return []
    return lines if is_first_chunk else lines[1:]  # Skip header


def _parse_standard(raw_text: str, is_first_chunk: bool) -> list[str]:
    """
    Simple records as a block
    """
    cleaned = raw_text.strip()
    return [cleaned] if cleaned else []

# --- Task definition ---

class EfetchProteinInputs:
    """Inputs for EFetch Protein task"""
    uid = lore.ArtifactInput(
        accepted_data=["accession", "protein_accession"],
        select="multiple",
        load_as="adapted",
        label="Protein accession(s)",
        description="WP_* or ABC1234567 to query the NCBI Protein database.",
    )
    fetch_format = lore.ValueInput(
        ProteinFormat,
        default=ProteinFormat.IPG,
        label="Fetch format",
        description="What data you want to retrieve for the given protein accession(s).",
    )


class EfetchProteinOutputs:
    """Outputs for EFetch Protein task"""
    protein_records = lore.TaskOutput(
        data_type="overridden_in_task",
        label="Protein records",
        description="Data retrieved from NCBI per the selected fetch format.",
    )


@lore.task(
    "ncbi.entrez.efetch_protein",
    name="NCBI Entrez EFetch Protein",
    inputs=EfetchProteinInputs,
    outputs=EfetchProteinOutputs,
    description="Query NCBI's Protein database for given protein accessions.",
    preview_mode="full",
)
def efetch_protein(
    ctx: lore.ExecutionContext,
    uid: list[str],
    fetch_format: ProteinFormat,
):
    """
    Query NCBI's Protein database for given protein accessions.
    """
    config = ctx.get_config("ncbi").model_dump() if ctx.get_config("ncbi") else {}
    api_key = config.get("api_key")
    email = config.get("email")

    clean_uids = [u.strip() for u in uid if u.strip()]
    if not clean_uids:
        raise ValueError("No valid UIDs provided.")

    rettype, retmode, extension, out_type = _FORMAT_MAP[fetch_format]

    # 1. Chunking logic (NCBI limits EFetch POSTs to avoid timeouts)
    chunk_size = 200
    uid_chunks = [clean_uids[i:i + chunk_size] for i in range(0, len(clean_uids), chunk_size)]

    accumulated_data = []

    @retry(tries=3, delay=2, default_logger=ctx.logger)
    def _execute_efetch(chunk):
        with entrez_client(api_key=api_key, email=email, ret=Retmode(retmode)) as client:
            data_payload = {
                "db": "protein",
                "id": ",".join(chunk),
                "retmode": "text",  # xml by default, but this returns TSV
            }
            if rettype:
                data_payload["rettype"] = rettype

            response = client.post(
                "efetch.fcgi",
                data=data_payload,
                timeout=60.0,
            )
            response.raise_for_status()
            return response.text

    for i, chunk in enumerate(uid_chunks):
        ctx.logger.info(f"Fetching chunk {i+1}/{len(uid_chunks)} ({len(chunk)} accessions) as {fetch_format.value}...")
        raw_text = _execute_efetch(chunk)

        is_first = (i == 0)

        if fetch_format == ProteinFormat.IPG:
            parsed = _parse_ipg_tsv(raw_text, is_first)
            accumulated_data.extend(parsed)
            join_char = "\n"
        else:
            parsed = _parse_standard(raw_text, is_first)
            accumulated_data.extend(parsed)
            join_char = "\n\n"  # double newline separates e.g. FASTA/XML records

        # Be a good citizen of the NCBI API
        sleep(0.2)  # 200 ms delay between requests, 10 requests per second limit

    if not accumulated_data:
        ctx.logger.warning(f"No {fetch_format.value} results found for the provided accessions.")
        return ctx.materialize_content("", output_key="protein_records", extension=extension)

    final_content = join_char.join(accumulated_data) + "\n"  # Ensure final newline
    file_prefix = "ipg_mapping" if fetch_format == ProteinFormat.IPG else "protein_records"
    
    ctx.logger.info(f"Successfully fetched {len(accumulated_data) - 1} rows of {fetch_format.value}.")
    return ctx.materialize_content(
        content=final_content,
        output_key="protein_records",
        name=file_prefix,
        extension=extension,
        data_type=out_type,
    )
