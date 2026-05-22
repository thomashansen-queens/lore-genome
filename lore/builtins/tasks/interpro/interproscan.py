"""
Task to submit and acquire the results of applying InterProScan on 
"""

import enum
import io
import logging
import zipfile

import httpx

import lore.dsl as lore
from lore.builtins.tasks.interpro.client import interpro_client

# InterPro's fair use policy requests that jobs are to be submitted in batches of 
# no more than 30 at a time, and for each batch to be fully completed before another job is submitted
MAX_BATCH = 25  

# Format to return completed InterProScan jobs
class ReturnFormat(str, enum.Enum):
    XML = "xml"
    TSV = 'tsv'
    GFF = 'gff3'
    JSON = "json"


NCBI_TYPE_MAP = {
    "PROT_FASTA": ("protein.faa", "protein_fastas"),
    "GENOME_FASTA": ("genomic.fna", "genome_fastas"),
    "GENOME_GFF": ("genomic.gff", "gff_annotations"),
    "GENOME_GBFF": ("genomic.gbff", "gff_annotations"),
    "GENOME_GTF": ("genomic.gtf", "gff_annotations"),
    "CDS_FASTA": ("cds_from_genomic.fna", "genome_fastas"),
    "SEQUENCE_REPORT": ("sequence_report.jsonl", "sequence_reports"),
    "CATALOG": ("dataset_catalog.json", "dataset_catalogs"),
    "ASSEMBLY_REPORT": ("assembly_data_report.jsonl", "assembly_reports"),
}


class InterProScanInputs:
    """Inputs for InterProScan 6"""
    protein_seqs = lore.ArtifactInput(
        description="List of protein sequences to run through InterProScan (max. 1000 sequences)",
        select=lore.MULTIPLE,
        load_as=lore.ADAPTED,
        accepted_data="protein_sequence",
        examples=["MANNKSAKKRAIQAEKRRQHNAS, FRKNMSNPSWVFWSGFKYLTLALASLA"],
    )
    # result_format = lore.ValueInput(
    #     ReturnFormat,
    #     default="xml",
    #     description="The file format for InterProScan jobs to return in.",
    #     label="Result format",
    # )


class InterProScanOutputs:
    """Outputs from InterProScan"""
    interproscan_results = lore.TaskOutput(
        data_type="interproscan_xml",
        label="InterProScan Results",
        description="XML files returned by InterProScan",
        yields=lore.MULTIPLE,
    )


@retry(default_logger=logging.getLogger("lore.ncbi"))
def _fetch_assembly_package(api: httpx.Client, accessions: list[str], **kwargs) -> bytes:
    """
    Fetch assembly package from NCBI Datasets API for a list of genome accessions
    """
    # Override the default json accept header to request the zip file
    headers = {"Accept": "application/zip"}

    result = api.get(
        f"/genome/accession/{','.join(accessions)}/download",
        params=kwargs,
        headers=headers,
    )
    return result.read()


@lore.memoize(prefix="ncbi_assembly_package", ignore="api")
def _fetch_single_assembly_package(
    ctx: lore.ExecutionContext,
    api: httpx.Client,
    genome_accession: str,
    extension: str = "faa",
    **kwargs,
) -> bytes | None:
    """
    Fetch one assembly package from NCBI Datasets API and extract its contents
    """
    clean_kwargs = {}
    for k, v in kwargs.items():
        if v in (None, "", [], {}):
            continue
        if isinstance(v, list):
            clean_kwargs[k] = [item.value if hasattr(item, "value") else str(item) for item in v]
        elif hasattr(v, "value"):
            clean_kwargs[k] = v.value
        else:
            clean_kwargs[k] = v

    response = _fetch_assembly_package(api, [genome_accession], **clean_kwargs)
    if not response:
        return None
    return response


@lore.task(
    "interpro.interproscan",
    inputs=InterProScanInputs,
    outputs=InterProScanOutputs,
    name="InterProScan 6",
    category="InterPro",
    icon="＞",
)
def interproscan(
    ctx: lore.ExecutionContext,
    protein_seqs: list[str],
    **kwargs,
):
    """
    Run InterProScan 6 on Protein sequences for domain prediction.
    """
    interpro_config = ctx.get_config("interpro")
    if not interpro_config or not interpro_config.email:
        ctx.logger.error("No email address was set in LoRe's settings for InterPro. InterPro's API requires you to provide your email to run.")
    
    
    
    with interpro_client() as client:
        payload = {
        "sequence": sequence,
        "stype": "p",  # 'p' for protein, 'n' for nucleotide
        }
        
        response = client.post("https://www.ebi.ac.uk/Tools/services/rest/iprscan6/run", data=payload)
        job_id = response.text.strip()
        
        while True:
            status_response = client.get(f"https://www.ebi.ac.uk/Tools/services/rest/iprscan6/status/{job_id}")
            status = status_response.text.strip()
            print(f"Current status: {status}")

            if status == "FINISHED":
                break
            elif status in ["ERROR", "FAILURE"]:
                print("Job failed on the server side.")
                exit(1)

            # Wait 15 seconds before checking again to prevent server flooding
            sleep(15)

        response = client.get(f"https://www.ebi.ac.uk/Tools/services/rest/iprscan6/result/{job_id}/xml")
        
        print(response.text)
        df = pd.read_csv(io.StringIO(response.text), sep='\t')
        df.to_csv("debug_output/interpro.csv", index=False)

    with ncbi_client(api_key) as api:
        for genome_acc in genome_accessions:
            try:
                zip_bytes = _fetch_single_assembly_package(ctx, api, genome_acc, **kwargs)
                if not zip_bytes:
                    failed_accessions.append(genome_acc)
                    continue

                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                    namelist = z.namelist()

                    for req_type in requested_types:
                        if req_type not in NCBI_TYPE_MAP:
                            ctx.logger.warning("Requested annotation type %s is not yet implemented.", req_type)
                            continue

                        filename, output_key = NCBI_TYPE_MAP[req_type]
                        target_files = [f for f in namelist if filename in f]

                        for target_file in target_files:
                            with z.open(target_file) as f:
                                content = f.read()
                                safe_name = f"{genome_acc}_{filename}"

                                ctx.materialize_content(
                                    content=content,
                                    output_key=output_key,
                                    name=safe_name,
                                    extension=filename.split(".")[-1],
                                )

            except Exception as e:
                ctx.logger.error("Failed to process %s: %s", genome_acc, e, exc_info=True)
                failed_accessions.append(genome_acc)
                continue

    # 4. Optionally save the failed accessions for later re-fetch
    if failed_accessions:
        ctx.logger.warning("Failed accessions: %s", ", ".join(failed_accessions))

        ctx.materialize_content(
            content="\n".join(failed_accessions),
            output_key="failed_accessions",
            name="failed_accessions",
            extension="txt",
            data_type="genome_accession",
        )
