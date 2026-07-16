"""
EFetch task for querying NCBI's Identical Protein Groups (IPG) database.
"""
import lore
from time import sleep
from .entrez_client import entrez_client
from .config import retry


class EfetchIpgInputs:
    """Inputs for EFetch IPG task"""
    uid = lore.ArtifactInput(
        accepted_data=["accession", "protein_accession"],
        select="multiple",
        load_as="adapted",
        label="Protein accession(s)",
        description="WP_* or ABC1234567 to query the NCBI IPG database.",
    )


class EfetchIpgOutputs:
    """Outputs for EFetch IPG task"""
    ipg_records = lore.TaskOutput(
        data_type="ipg_records",
        label="IPG Records",
        description="NCBI Identical Protein Groups (IPG) records for the given protein accessions.",
    )


@lore.task(
    "ncbi.entrez.efetch_ipg",
    inputs=EfetchIpgInputs,
    outputs=EfetchIpgOutputs,
    description="Query NCBI's Identical Protein Groups (IPG) database for given protein accessions.",
    preview_mode="full",
)
def efetch_ipg(
    ctx: lore.ExecutionContext,
    uid: list[str],
):
    """
    Query NCBI's Identical Protein Groups (IPG) database for given protein accessions.
    """
    config = ctx.get_config("ncbi").model_dump() if ctx.get_config("ncbi") else {}
    api_key = config.get("api_key")
    email = config.get("email")

    clean_uids = [u.strip() for u in uid if u.strip()]
    if not clean_uids:
        raise ValueError("No valid UIDs provided.")

    all_tsv_lines = []
    header_captured = False

    # 1. Chunking logic (NCBI limits EFetch POSTs to avoid timeouts)
    chunk_size = 200
    uid_chunks = [clean_uids[i:i + chunk_size] for i in range(0, len(clean_uids), chunk_size)]

    @retry(tries=3, delay=2, default_logger=ctx.logger)
    def _execute_efetch(chunk):
        with entrez_client(api_key=api_key, email=email, ret="text") as client:
            response = client.post(
                "efetch.fcgi",
                data={
                    "db": "protein",
                    "id": ",".join(chunk),
                    "rettype": "ipg",
                    "retmode": "text",  # xml by default, but this returns TSV
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.text

    for i, chunk in enumerate(uid_chunks):
        ctx.logger.info(f"Fetching IPG chunk {i+1}/{len(uid_chunks)} ({len(chunk)} accessions)...")
        raw_text = _execute_efetch(chunk)

        # Parse and accumulate the TSV rows
        lines = [line for line in raw_text.splitlines() if line.strip()]
        if not lines:
            continue

        if not header_captured:
            # Grab the header from the very first chunk
            all_tsv_lines.append(lines[0])
            header_captured = True

        # Append the data rows (skipping the header)
        all_tsv_lines.extend(lines[1:])

        # Be a good citizen of the NCBI API
        sleep(0.2)  # 200 ms delay between requests, 10 requests per second limit

    if len(all_tsv_lines) <= 1:
        ctx.logger.warning("No IPG results found for the provided accessions.")
        return ctx.materialize_content("", output_key="ipg_table", extension="tsv")

    final_tsv = "\n".join(all_tsv_lines) + "\n"

    ctx.logger.info(f"Successfully mapped {len(all_tsv_lines) - 1} IPG rows.")
    return ctx.materialize_content(
        content=final_tsv,
        name="ipg_mapping",
        extension="tsv",
    )
