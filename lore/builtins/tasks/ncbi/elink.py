"""
ELink Task for querying the NCBI Entrez database.
"""
import json

from time import sleep
import lore
from .entrez_client import entrez_client, EntrezDb
from .config import retry


class ELinkInputs:
    """Inputs for ELink Task. Accepting target_uid allows chaining ELink"""
    uid = lore.ArtifactInput(
        accepted_data=["uid", "target_uid"],
        select="multiple",
        load_as="adapted",
        label="Source UIDs",
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

    failed_uids = lore.TaskOutput(
        data_type="uid",
        label="Failed uids",
        description="The list of source UIDs that ELink or the task could not process.",
        yields="optional",
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
    Finds linked UIDs in a target NCBI database from UIDs in a source database.
    """
    # 1. Build NCBI configuration
    config = ctx.get_config("ncbi").model_dump() if ctx.get_config("ncbi") else {}
    
    api_key = config.get("api_key")
    if not api_key:
        ctx.logger.warning("No NCBI API key set in Settings! Authentication may be rate-limited.")
        api_delay = 0.4
    else:
        api_delay = 0.1
    email = config.get("email")

    # Filter out empty UIDs and ensure proper space splitting
    clean_uids = []
    for u in uid:
        if u:
            clean_uids += u.split()

    @retry(tries=1, delay=api_delay, default_logger=ctx.logger)
    def _execute_link_batch(uids: list[str]):
        with entrez_client(api_key=api_key, email=email) as client: # type: ignore
            response = client.post(
                "elink.fcgi",
                data={
                    "dbfrom": dbfrom.value,
                    "db": db.value,
                    "id": uids,  # using a list formats to 'id=UID1&id=UID2', which separates linksets by UID
                    # "byid": "y",  # this only works for retmode='xml'... leaving for now
                },
            )
            response.raise_for_status()
            try:
                return response.json()
            except json.decoder.JSONDecodeError:
                raise ValueError("Invalid accession in input.")

    def _capacity_error(data) -> bool:
        text = json.dumps(data).lower() if isinstance(data, dict) else str(data).lower()
        return any(marker in text for marker in (
            "too many links", "too many records", "result is too large",
            "request uri too long", "414 request-uri", "payload too large",
        ))

    def _execute_adaptive(uids: list[str]):
        # ELink has unusually large responses for records with many neighbors. Keep
        # normal batches bounded, and retry smaller batches if NCBI rejects an otherwise
        # valid request or reports that its result is too large.
        batch_size = 50
        failed_uids = []
        combined_linksets = []

        for start in range(0, len(uids), batch_size):
            batch = uids[start:start + batch_size]

            def execute(batch_uids: list[str]):
                if not batch_uids:
                    return []
                try:
                    data = _execute_link_batch(batch_uids)
                except Exception as exc:
                    response = getattr(exc, "response", None)
                    status = getattr(response, "status_code", None)
                    if len(batch_uids) > 1:
                        ctx.logger.warning(
                            "ELink batch of %d UIDs failed (status %s); splitting it.",
                            len(batch_uids), status,
                        )
                        midpoint = len(batch_uids) // 2
                        sleep(api_delay) # avoid hammering NCBI with immediate retries
                        return execute(batch_uids[:midpoint]) + execute(batch_uids[midpoint:])
                    else:
                        failed_uids.append(batch_uids[0])
                        ctx.logger.error("ELink could not process UID %s: %s", batch_uids[0], exc)
                        return []  

                if _capacity_error(data):
                    if len(batch_uids) == 1:
                        failed_uids.append(batch_uids[0])
                        ctx.logger.error("ELink could not process UID %s: %s", batch_uids[0], data)
                        return []
                    ctx.logger.warning(
                        "ELink result for %d UIDs was too large; splitting it.", len(batch_uids)
                    )
                    midpoint = len(batch_uids) // 2
                    return execute(batch_uids[:midpoint]) + execute(batch_uids[midpoint:])

                linksets = data.get("linksets", [])
                for linkset, query_uid in zip(linksets, batch_uids):
                    linkset["_query_uid"] = query_uid
                return linksets

            combined_linksets.extend(execute(batch))

        return combined_linksets, failed_uids

    # 2. Execute the ELink request, splitting only batches NCBI cannot handle
    linksets, failed_uids = _execute_adaptive(clean_uids)
    # ctx.logger.debug(f"ELink response linksets: {linksets}")

    data = {"linksets": linksets}
    if not linksets:
        ctx.logger.warning("No linksets found in ELink response.")
    elif "error" in linksets[0]:
        raise RuntimeError(f"ELink API error: {linksets[0]['error']}")

    # 3. Process the retrieved linksets and prepare output table
    table_data = []
    for linkset in linksets:
        source_uid = linkset.get("ids", [None])[0]  # Get the first UID from the source
        query_uid = str(linkset.get("_query_uid") or source_uid) if (
            linkset.get("_query_uid") is not None or source_uid is not None
        ) else None

        for linksetdb in linkset.get("linksetdbs", []):
            link_name = linksetdb.get("linkname", "unknown_link")

            for target_uid in linksetdb.get("links", []):
                table_data.append({
                    "source_db": dbfrom.value,
                    "query_uid": query_uid,
                    "source_uid": source_uid,
                    "target_db": db.value,
                    "target_uid": target_uid,
                    "link_name": link_name,
                })

    if not table_data:
        ctx.logger.warning("No links found in ELink response.")

    # 4. Materialize the output content as a JSON artifact
    ctx.materialize_content(
        content=json.dumps(table_data, indent=2),
        output_key="link_results",
        name=f"ELink: {dbfrom.value} to {db.value}",
        extension="json",
        metadata={"columns": ["source_db", "source_uid", "target_db", "target_uid", "link_name"]},
    )

    if failed_uids:
        ctx.logger.warning("Failed UIDs: %s", ", ".join(failed_uids))
        ctx.materialize_content(
            content="\n".join(failed_uids),
            output_key="failed_uids",
            name="failed_uids",
            extension="txt",
            data_type="uid",
        )
