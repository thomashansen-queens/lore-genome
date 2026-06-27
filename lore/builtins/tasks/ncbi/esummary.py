"""
ESummary Task for querying the NCBI Entrez database.
"""
import json

import lore
from .entrez_client import entrez_client, EntrezDb
from .config import retry


class ESummaryInputs:
    """Inputs for ESummary Task"""
    uid = lore.ArtifactInput(
        accepted_data=["uid", "target_uid"],
        select="multiple",
        label="UID",
        description=(
            "The unique identifier (UID) of the record to summarize. "
            "This is typically obtained from an ESearch query."
        ),
    )
    database = lore.ValueInput(
        EntrezDb,
        label="Database",
        description="The NCBI Entrez database to query.",
    )


class ESummaryOutputs:
    """Outputs for ESummary Task"""
    summary = lore.TaskOutput(
        data_type="entrez_summary",
        label="Summary",
        is_primary=True,
    )


@lore.task(
    "ncbi.entrez.esummary",
    inputs=ESummaryInputs,
    outputs=ESummaryOutputs,
    name="NCBI Entrez ESummary",
    category="NCBI Entrez",
    preview_mode="full",
)
def esummary(
    ctx: lore.ExecutionContext,
    uid: list[str],
    database: EntrezDb,
):
    """
    Retrieves summaries from a specified NCBI Entrez database for a list of UIDs.
    """
    config = ctx.get_config("ncbi").model_dump() if ctx.get_config("ncbi") else {}
    api_key = config.get("api_key")
    email = config.get("email")

    clean_uids = [u for u in uid if u]  # Filter out empty UIDs

    # 1. API call closure (for retry decorator)
    @retry(tries=3, delay=2, default_logger=ctx.logger)
    def _execute_summary():
        with entrez_client(api_key=api_key, email=email) as client:
            response = client.post(
                "esummary.fcgi",
                data={
                    "db": database.value,
                    "id": ",".join(clean_uids),
                },
            )
            response.raise_for_status()
            return response.json()

    data = _execute_summary()
    result_dict = data.get("result", {})

    ctx.logger.debug(f"ESummary raw result: {data}")

    if not result_dict:
        ctx.logger.warning("No summary results found in the response.")
        return {"summary": {}}

    if "error" in result_dict:
        raise RuntimeError(f"ESummary API error: {result_dict['error']}")

    table_data = []

    # Track top-level columns dynamically for metadata
    all_seen_columns = set(["uid"])

    for result_uid, summary in result_dict.items():
        if result_uid == "uids":
            # Skip the 'uids' field, which echoes the input UIDs
            continue

        row = {"uid": result_uid}
        for k, v in summary.items():
            # Skip complex nested structures to keep table clean
            if isinstance(v, (str, int, float, bool)):
                row[k] = v
                all_seen_columns.add(k)

        # 2. SRA nested XML parsing
        # TODO: This is a lot of code in the main handler just for SRA XML...
        #       Could be abstracted. OR make an SRA XML Adapter for easy schema mapping
        if database.value == "sra":
            import xml.etree.ElementTree as ET

            runs_xml = summary.get("runs", "")
            exp_xml = summary.get("expxml", "")

            # A. Extract Runs (accessions like SRR12345678)
            if runs_xml.strip():
                try:
                    root = ET.fromstring(f"<root>{runs_xml}</root>")

                    accs, spots, bases, statics = [], [], [], []
                    for run in root.findall(".//Run"):
                        accs.append(run.attrib.get("acc"))
                        spots.append(run.attrib.get("total_spots", ""))
                        bases.append(run.attrib.get("total_bases", ""))
                        statics.append(run.attrib.get("static_data_available", ""))

                    row["srr_accession"] = ", ".join(accs)
                    row["total_spots"] = ", ".join(spots)
                    row["total_bases"] = ", ".join(bases)
                    row["static_data_available"] = ", ".join(statics)
                except ET.ParseError as e:
                    ctx.logger.warning(f"Failed to parse runs XML for UID {result_uid}: {e}")
                    row["srr_accession"] = []
                    row["total_spots"] = []
                    row["total_bases"] = []
                    row["static_data_available"] = []

            # B. Extract experiment metadata (e.g. sequencing platform)
            if exp_xml.strip():
                try:
                    exp_root = ET.fromstring(f"<root>{exp_xml}</root>")

                    platform_tag = exp_root.find(".//Platform")
                    if platform_tag is not None:
                        row["platform"] = platform_tag.text
                        row["instrument_model"] = platform_tag.attrib.get("instrument_model", "")

                    org_tag = exp_root.find(".//Organism")
                    if org_tag is not None:
                        row["organism"] = org_tag.attrib.get("ScientificName", "")

                    lib_strat = exp_root.find(".//LIBRARY_STRATEGY")
                    if lib_strat is not None:
                        row["library_strategy"] = lib_strat.text

                    bioproject = exp_root.find(".//Bioproject")
                    if bioproject is not None:
                        row["bioproject_accession"] = bioproject.text

                    biosample = exp_root.find(".//Biosample")
                    if biosample is not None:
                        row["biosample_accession"] = biosample.text

                except ET.ParseError as e:
                    ctx.logger.debug(f"No experiment metadata XML found for UID {result_uid}: {e}")
                    pass

        table_data.append(row)

    if not table_data:
        ctx.logger.warning(f"No summaries found in {database.value} for UIDs: {clean_uids}")

    # 3. Ensure completeness of columns across all rows
    for row in table_data:
        for col in all_seen_columns:
            if col not in row:
                row[col] = ""

    ctx.materialize_content(
        content=json.dumps(table_data, indent=2),
        output_key="summary",
        name=f"ESummary {database.value}",
        extension="json",
        metadata={"columns": list(all_seen_columns)},
    )
