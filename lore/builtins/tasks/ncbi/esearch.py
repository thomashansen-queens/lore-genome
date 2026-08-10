"""
ESearch Task for querying the NCBI Entrez database.
"""
import json
import re

import lore
from .entrez_client import entrez_client, EntrezDb
from .config import retry


class ESearchInputs:
    """Inputs for ESearch Task"""
    database = lore.ValueInput(
        EntrezDb,
        label="Database",
        description="The NCBI Entrez database to search.",
    )
    terms = lore.ArtifactInput(
        select="single",
        load_as="raw",
        accepted_data="text",
        label="Search Term File",
        description="A file containing a list of search term(s)."
    )


class ESearchOutputs:
    """Outputs for ESearch Task"""
    search_results = lore.TaskOutput(
        data_type="uid",
        label="Search Results",
        is_primary=True,
    )


@lore.task(
    "ncbi.entrez.esearch",
    inputs=ESearchInputs,
    outputs=ESearchOutputs,
    name="NCBI Entrez ESearch",
    category="NCBI Entrez",
    preview_mode="full",
)
def esearch(
    ctx: lore.ExecutionContext,
    database: EntrezDb,
    terms: str | list[str],
):
    """
    Performs an ESearch query on the NCBI Entrez database.
    Provides unique IDs (UID) for records that match a given search term
    within a specified database.

    See the NCBI documentation for more details:
    https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch
    """
    # 1. Format the RAW query for Entrez
    # Manual input and plain-text files arrive as a string; tabular files
    # arrive as a list of lines. Make one string, then split on
    # newlines, commas, and tabs so term lists work however they were provided
    text = terms if isinstance(terms, str) else "\n".join(terms)
    clean_terms = [t.strip() for t in re.split(r"[\n,\t]+", text) if t.strip()]
    query_string = " OR ".join(clean_terms)

    # 2. Global config for NCBI
    config = ctx.get_config("ncbi").model_dump() if ctx.get_config("ncbi") else {}
    api_key = config.get("api_key")
    email = config.get("email")

    # 3. API call
    @retry(tries=3, delay=2, default_logger=ctx.logger)
    def _execute_search():
        with entrez_client(api_key=api_key, email=email) as client:
            response = client.post(
                "esearch.fcgi",
                data={
                    "db": database.value,
                    "term": query_string,
                    "usehistory": "y",
                    "retmax": 10000,  # Max number of UIDs ESearch can return
                },
            )
            response.raise_for_status()
            return response.json()

    # 4. Execute
    data = _execute_search()

    # 5. Extract the data from the response JSON
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

    # 6. Materialize the output
    output_dict = json.dumps(table_data, indent=2)
    ctx.materialize_content(
        content=output_dict,
        output_key="search_results",
        name="ESearch Results",
        extension="json",
        metadata={"columns": ["uid", "database", "query_key", "webenv"]},
    )
