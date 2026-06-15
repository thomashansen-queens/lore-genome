"""
ELink Task for querying the NCBI Entrez database.
"""
import json

import lore
from .entrez_client import entrez_client, EntrezDb
from .config import retry


class ELinkInputs:
    """Inputs for ELink Task. Accepting target_uid allows chaining ELink"""
    uid = lore.ArtifactInput(
        accepted_data=["uid", "target_uid"],
        select="multiple",
        label="UID",
        description=(
            "The unique identifier (UID) of the record to link from. "
            "This is typically obtained from an ESearch query."
        ),
    )
    dbfrom = lore.ValueInput(
        EntrezDb,
        label="Source Database",
        description="The NCBI Entrez database to link from.",
    )
    db = lore.ValueInput(
        EntrezDb,
        label="Target Database",
        description="The NCBI Entrez database to link to.",
    )


class ELinkOutputs:
    """Outputs for ELink Task"""
    link_results = lore.TaskOutput(
        data_type="entrez_link",
        label="Link Results",
        is_primary=True,
    )


@lore.task(
    "ncbi.entrez.elink",
    inputs=ELinkInputs,
    outputs=ELinkOutputs,
    name="NCBI Entrez ELink",
    category="NCBI Entrez",
    preview_mode="full",
)
def elink(
    ctx: lore.ExecutionContext,
    uid: list[str],
    dbfrom: EntrezDb,
    db: EntrezDb,
):
    """
    Finds linked UIDs in a target NCBI database based that originate
    from UIDs in a source database.
    """
    config = ctx.get_config("ncbi").model_dump() if ctx.get_config("ncbi") else {}
    api_key = config.get("api_key")
    email = config.get("email")

    clean_uids = [u for u in uid if u]  # Filter out empty UIDs

    @retry(tries=3, delay=2, default_logger=ctx.logger)
    def _execute_link():
        with entrez_client(api_key=api_key, email=email) as client:
            response = client.get(
                "elink.fcgi",
                params={
                    "dbfrom": dbfrom.value,
                    "db": db.value,
                    "id": ",".join(clean_uids),
                },
            )
            return response.json()

    data = _execute_link()
    ctx.logger.debug(f"ELink response data: {data}")

    # ELink JSONL format
    linksets = data.get("linksets", [])
    if not linksets:
        ctx.logger.warning("No linksets found in ELink response.")
        linksets = []

    if "error" in linksets[0]:
        raise RuntimeError(f"ELink API error: {linksets[0]['error']}")

    table_data = []
    for linkset in linksets:
        source_uid = linkset.get("ids", [None])[0]  # Get the first UID from the source

        for linksetdb in linkset.get("linksetdbs", []):
            link_name = linksetdb.get("linkname", "unknown_link")

            for target_uid in linksetdb.get("links", []):
                table_data.append({
                    "source_db": dbfrom.value,
                    "source_uid": source_uid,
                    "target_db": db.value,
                    "target_uid": target_uid,
                    "link_name": link_name,
                })

    if not table_data:
        ctx.logger.warning("No links found in ELink response.")

    ctx.materialize_content(
        content=json.dumps(table_data, indent=2),
        output_key="link_results",
        name=f"ELink: {dbfrom.value} to {db.value}",
        extension="json",
        metadata={"columns": ["source_db", "source_uid", "target_db", "target_uid", "link_name"]},
    )
