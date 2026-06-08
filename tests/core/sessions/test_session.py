"""
Tests for core Session orchestrator class
"""


def test_session_init(temp_session):
    """
    Tests that the Session bootstraps correctly
    """
    assert (temp_session.dir / "manifest.json").exists()
    assert (temp_session.dir / "artifacts").is_dir()
    assert len(temp_session.list_artifacts()) == 0


def test_artifact_registration(temp_session, dummy_json_file):
    """
    Tests that artifacts can be registered with the Session
    """
    artifact = temp_session.register_artifact(
        dummy_json_file,
        name="test_artifact",
        data_type="json",
        metadata={"source": "generated", "format": "json"},
    )
    assert len(temp_session.list_artifacts()) == 1
    assert artifact.name == "test_artifact"
    assert artifact.data_type == "json"


def test_session_persistence(temp_runtime, closed_session, dummy_json_file):
    """
    Tests that the manifest survives a full close/reopen cycle.
    Registers an artifact, closes the session, reopens it, and confirms
    the artifact is still there with its metadata intact.
    """
    # Write something into the closed session and let it close again
    with temp_runtime.open_session(closed_session.id) as s:
        s.register_artifact(dummy_json_file, name="persisted_artifact", data_type="json")

    # Reopen independently — this is a brand-new Session object reading from disk
    with temp_runtime.open_session(closed_session.id) as s:
        artifacts = s.list_artifacts()
        assert len(artifacts) == 1
        assert artifacts[0].name == "persisted_artifact"
        assert artifacts[0].data_type == "json"


def test_session_rename(temp_runtime, closed_session):
    """
    Tests that renaming a Session renames the directory on disk and persists the
    new name in the manifest.
    """
    # 1. Renaming a closed Session
    original_dir = closed_session.dir
    closed_session.name = "Renamed Session"

    # closed_session._root is now stale — use the Runtime to find the actual path
    renamed_dir = temp_runtime.find_session_dir(closed_session.id)
    assert not original_dir.exists()
    assert renamed_dir.exists()
    assert renamed_dir != original_dir

    # 2. Renaming an open Session
    with temp_runtime.open_session(closed_session.id) as s:
        assert s.name == "Renamed Session"
        assert s.dir == renamed_dir  # Fresh session object has the correct path

        s.name = "Renamed Again"
        assert s.name == "Renamed Again"
        assert s.dir == renamed_dir  # Directory unchanged while open (deferred)

    # 3. After closing, sync_session_dir renames the directory to match
    final_dir = temp_runtime.find_session_dir(closed_session.id)
    assert final_dir != renamed_dir

    with temp_runtime.open_session(closed_session.id) as s:
        assert s.name == "Renamed Again"
        assert s.dir == final_dir
