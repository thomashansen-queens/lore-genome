"""
EFetch task for querying NCBI's Protein database
https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EFetch

Defaults to Identical Protein Groups (IPG) database records, which contain
various linked metadata for a given protein accession (assembly, nucleotide,
taxonomy, etc.).)
"""
import csv
from collections import defaultdict
from enum import StrEnum
from io import StringIO
from time import sleep

import lore

from .config import retry
from .entrez_client import Retmode, entrez_client


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


def _parse_ipg_records(raw_text: str) -> list[tuple[str, str, str]]:
    """Return protein, nucleotide, and assembly accessions in IPG table order."""
    rows = list(csv.reader(raw_text.splitlines(), delimiter="\t"))
    if not rows:
        return []

    protein_headers = {"protein", "protein accession", "protein_accession"}
    nucleotide_headers = {"nucleotide accession", "nucleotide_accession"}
    assembly_headers = {"assembly", "assembly accession", "assembly_accession"}
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if any(cell.strip().lstrip("#").strip().casefold() in protein_headers for cell in row)
        ),
        None,
    )
    if header_index is None:
        raise ValueError("NCBI IPG response did not contain a Protein column header.")

    header = [cell.strip().lstrip("#").strip().casefold() for cell in rows[header_index]]
    protein_index = next(i for i, cell in enumerate(header) if cell in protein_headers)
    nucleotide_index = next(
        (i for i, cell in enumerate(header) if cell in nucleotide_headers),
        None,
    )
    assembly_index = next(
        (i for i, cell in enumerate(header) if cell in assembly_headers),
        None,
    )
    records = []
    for row in rows[header_index + 1:]:
        if protein_index < len(row):
            protein = row[protein_index].strip()
            if protein:
                nucleotide = (
                    row[nucleotide_index].strip()
                    if nucleotide_index is not None and nucleotide_index < len(row)
                    else ""
                )
                assembly = (
                    row[assembly_index].strip()
                    if assembly_index is not None and assembly_index < len(row)
                    else ""
                )
                records.append((protein, nucleotide, assembly))
    return records


# Selects the representative accession for a cluster of identical proteins, preferring WP_* accessions.
def _select_ipg_cluster_representative(cluster: dict[str, int]) -> str:
    return max(
        cluster,
        key=lambda accession: (cluster[accession], accession.startswith("WP")),
    )


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
    protein_fasta = lore.TaskOutput(
        data_type="protein_fasta",
        label="Protein FASTA",
        description="FASTA sequences retrieved from NCBI for the given protein accession(s).",
    )
    ipg_records = lore.TaskOutput(
        data_type="ipg_records",
        label="Protein records",
        description=(
            "Identical Protein Groups (IPG) records, containing linked metadata for each protein."
        ),
    )
    ipg_clusters = lore.TaskOutput(
        data_type="protein_nucleotide_assembly_mapping",
        label="IPG Protein Clusters",
        description="Unique protein accessions grouped under each queried protein accession.",
        yields="optional",
    )
    feature_table = lore.TaskOutput(
        data_type="feature_table",
        label="Feature Table",
        description="Feature table retrieved from NCBI for the given protein accession(s).",
    )
    native_xml = lore.TaskOutput(
        data_type="protein_xml",
        label="Native XML",
        description="Native XML data retrieved from NCBI for the given protein accession(s).",
    )
    gbseq_xml = lore.TaskOutput(
        data_type="protein_xml",
        label="GBSeq XML",
        description="GBSeq XML data retrieved from NCBI for the given protein accession(s).",
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

    clean_uids = []
    for u in uid:
        clean_uids.extend(u.split())
    if not clean_uids:
        raise ValueError("No valid UIDs provided.")

    rettype, retmode, extension, out_type = _FORMAT_MAP[fetch_format]

    # IPG results need to retain the relationship to each queried accession.
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

    acc_to_cluster_map = defaultdict(lambda: defaultdict(int))
    acc_to_metadata = defaultdict(dict)

    for i, chunk in enumerate(uid_chunks):
        ctx.logger.info(
            f"Fetching chunk {i+1}/{len(uid_chunks)} "
            f"({len(chunk)} accessions) as {fetch_format.value}..."
        )
        raw_text = _execute_efetch(chunk)

        is_first = not accumulated_data

        if fetch_format == ProteinFormat.IPG:
            parsed = _parse_ipg_tsv(raw_text, is_first)
            accumulated_data.extend(parsed)
            query_accessions = iter(chunk)
            parent_acc = next(query_accessions, None)
            next_acc = next(query_accessions, None)

            ipg_records = _parse_ipg_records(raw_text)
            for acc, nucleotide, assembly in ipg_records:
                acc_to_metadata[acc].setdefault((nucleotide, assembly), None)
                if acc in acc_to_cluster_map:
                    if not acc == parent_acc and id(acc_to_cluster_map[acc]) != id(acc_to_cluster_map[parent_acc]):
                        # Merge the two dictionaries
                        for key in acc_to_cluster_map[acc].keys():
                            acc_to_cluster_map[parent_acc][key] += acc_to_cluster_map[acc][key]
                            acc_to_cluster_map[key] = acc_to_cluster_map[parent_acc]
                    acc_to_cluster_map[acc][acc] += 1
                else:
                    if acc == next_acc:
                        parent_acc = next_acc
                        next_acc = next(query_accessions, None)
                    acc_to_cluster_map[parent_acc][acc] += 1

            join_char = "\n"
        else:
            parsed = _parse_standard(raw_text, is_first)
            accumulated_data.extend(parsed)
            join_char = "\n\n"  # double newline separates e.g. FASTA/XML records

        # Be a good citizen of the NCBI API
        sleep(0.2)  # 200 ms delay between requests, 10 requests per second limit

    if not accumulated_data:
        ctx.logger.warning(f"No {fetch_format.value} results found for the provided accessions.")
        final_content = ""
    else:
        final_content = join_char.join(accumulated_data) + "\n"  # Ensure final newline

    file_prefix = "ipg_mapping" if fetch_format == ProteinFormat.IPG else "protein_records"

    if accumulated_data:
        ctx.logger.info(
            f"Successfully fetched {len(accumulated_data) - 1} rows of {fetch_format.value}."
        )

    primary_output = ctx.materialize_content(
        content=final_content,
        output_key=out_type,
        name=file_prefix,
        extension=extension,
        data_type=out_type,
    )

    if fetch_format == ProteinFormat.IPG:
        cluster_csv = StringIO()
        writer = csv.writer(cluster_csv, lineterminator="\n")
        writer.writerow([
            "representative_protein_accession",
            "protein_accession",
            "nucleotide_accession",
            "assembly_accession",
        ])
        clusters = []
        seen = set()
        for cluster in acc_to_cluster_map.values():
            if id(cluster) not in seen:
                seen.add(id(cluster))
                clusters.append((_select_ipg_cluster_representative(cluster), cluster.keys()))
        for representative, accessions in clusters:
            for accession in accessions:
                metadata = acc_to_metadata.get(accession) or {("", ""): None}
                writer.writerows(
                    (representative, accession, nucleotide, assembly)
                    for nucleotide, assembly in metadata
                )

        ctx.materialize_content(
            content=cluster_csv.getvalue(),
            output_key="ipg_clusters",
            name="ipg_clusters",
            extension="csv",
            data_type="protein_nucleotide_assembly_mapping",
        )

    return primary_output
