"""
Tests for FastAPI app entry
"""
from hashlib import sha256

from fastapi import FastAPI

from lore.web import deps
from lore.web.api import create_app


def test_create_app_returns_fastapi(temp_runtime):
    """create_app() returns a properly configured FastAPI instance."""
    app = create_app(temp_runtime)
    assert isinstance(app, FastAPI)
    assert app.state.rt is temp_runtime


def test_health_endpoint(client):
    """The /health endpoint responds 200 with ok: true."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_expected_routes_registered(temp_runtime):
    """All feature routers are mounted under their expected path prefixes."""
    app = create_app(temp_runtime)
    paths = {route.path for route in app.routes}

    # Spot-check one route per included router to confirm registration
    assert any(p.startswith("/sessions") for p in paths)
    assert any(p.startswith("/settings") for p in paths)
    # Artifacts are nested under sessions, not top-level
    assert any("/artifacts/" in p for p in paths)


def test_main_css_url_changes_when_file_changes(client, tmp_path, monkeypatch):
    static_dir = tmp_path / "static"
    css_path = static_dir / "css" / "main.css"
    css_path.parent.mkdir(parents=True)
    css_path.write_text("body { color: red; }", encoding="utf-8")
    monkeypatch.setattr(deps, "STATIC_DIR", static_dir)

    first_response = client.get("/dashboard")
    first_hash = sha256(css_path.read_bytes()).hexdigest()
    assert f"/static/css/main.css?v={first_hash}" in first_response.text

    css_path.write_text("body { color: blue; }", encoding="utf-8")
    second_response = client.get("/dashboard")
    second_hash = sha256(css_path.read_bytes()).hexdigest()
    assert first_hash != second_hash
    assert f"/static/css/main.css?v={second_hash}" in second_response.text
