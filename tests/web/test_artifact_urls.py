from urllib.parse import parse_qs, urlsplit

from lore.web.features.artifacts.urls import artifact_page_url


def test_artifact_page_url_preserves_context_and_encodes_query_values():
    url = artifact_page_url(
        "session-id",
        "artifact-id",
        "explore",
        origin_task="task-id",
        adapter_key="table & chart",
    )

    parsed = urlsplit(url)
    assert parsed.path == "/sessions/session-id/artifacts/artifact-id/explore"
    assert parse_qs(parsed.query) == {
        "origin_task": ["task-id"],
        "adapter_key": ["table & chart"],
    }


def test_artifact_page_url_omits_empty_query():
    assert artifact_page_url("session-id", "artifact-id") == (
        "/sessions/session-id/artifacts/artifact-id"
    )
