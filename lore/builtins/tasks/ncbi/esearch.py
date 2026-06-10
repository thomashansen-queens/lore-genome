"""
ESearch Task for querying the NCBI Entrez database.
"""
import json

import lore
from .entrez_client import entrez_client, EntrezDb
from .config import retry


class ESearchInputs:
    """Inputs for ESearch Task"""
    term = lore.ValueInput(
        str,
        label="Search Term",
        description=(
            "See the NCBI documentation for more details: \n"
            "https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch"
        ),
    )
    database = lore.ValueInput(
        EntrezDb,
        label="Database",
        description="The NCBI Entrez database to search.",
    )
    

class ESearchOutputs:
    """Outputs for ESearch Task"""
    search_results = lore.TaskOutput(
        data_type="entrez_uid",
        label="Search Results",
        is_primary=True,
    )


@lore.task(
    "ncbi.entrez.esearch",
    inputs=ESearchInputs,
    outputs=ESearchOutputs,
    name="NCBI Entrez ESearch",
    category="NCBI Entrez",
)
def esearch(
    ctx: lore.ExecutionContext,
    database: EntrezDb,
    term: str,
):
    """
    Performs an ESearch query on the NCBI Entrez database.
    Provides unique IDs (UID) for records that match a given search term
    within a specified database.
    """
    # 1. Global config for NCBI
    config = ctx.get_config("ncbi").model_dump() if ctx.get_config("ncbi") else {}

    api_key = config.get("api_key")
    email = config.get("email")

    # 2. API call
    @retry(tries=3, delay=2, default_logger=ctx.logger)
    def _execute_search():
        with entrez_client(api_key=api_key, email=email) as client:
            response = client.get(
                "esearch.fcgi",
                params={"db": database.value, "term": term, "usehistory": "y"},
            )
            return response.json()

    # 3. Execute
    data = _execute_search()

    # 4. Extract the data from the response JSON
    ctx.logger.debug(f"ESearch response data: {data}")
    search_result = data.get("esearchresult", {})

    if "error" in search_result:
        raise RuntimeError(f"ESearch API error: {search_result['error']}")

    table_data = []
    for uid in search_result.get("idlist", []):
        table_data.append({
            "uid": uid,
            "database": database.value,
            "query_key": search_result.get("querykey"),
            "webenv": search_result.get("webenv"),
        })

    # 5. Materialize the output
    output_dict = json.dumps(table_data, indent=2)
    ctx.materialize_content(
        content=output_dict,
        output_key="search_results",
        name="ESearch Results",
        extension="json",
        metadata={"columns": ["uid", "database", "query_key", "webenv"]},
    )
