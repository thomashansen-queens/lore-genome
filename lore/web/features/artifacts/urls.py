from urllib.parse import urlencode


def artifact_page_url(
    session_id: str,
    artifact_id: str,
    page: str = "",
    *,
    origin_task: str | None = None,
    **query: str | None,
) -> str:
    params = {key: value for key, value in query.items() if value is not None}
    if origin_task:
        params["origin_task"] = origin_task

    url = f"/sessions/{session_id}/artifacts/{artifact_id}"
    if page:
        url += f"/{page}"
    return f"{url}?{urlencode(params)}" if params else url
