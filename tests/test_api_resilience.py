"""Gateway tests for corrupt model artifacts and the web-driven reload flow."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import main as main_module  # noqa: E402
from app.model import ModelRegistry  # noqa: E402


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    shutil.copy(ROOT / "models" / "v1.json", tmp_path / "v1.json")
    shutil.copy(ROOT / "models" / "v2.json", tmp_path / "v2.json")
    return tmp_path


@pytest.fixture()
def resilient_client(model_dir: Path):
    """Point the process-global gateway at a throwaway model directory."""
    original_registry = main_module.registry
    original_worker_registry = main_module.worker.registry
    original_active = main_module.active_version
    registry = ModelRegistry.load_dir(model_dir, "v1")
    main_module.registry = registry
    main_module.worker.registry = registry
    main_module.active_version = registry.default_version
    main_module.jobs.clear()
    main_module.switches.clear()
    main_module.worker.metrics.update({k: 0 for k in main_module.worker.metrics})
    try:
        with TestClient(main_module.app) as client:
            yield client, registry, model_dir
    finally:
        main_module.registry = original_registry
        main_module.worker.registry = original_worker_registry
        main_module.active_version = original_active


def test_startup_with_one_corrupt_file_still_serves(resilient_client):
    client, registry, model_dir = resilient_client
    (model_dir / "v3.json").write_text("{broken json", encoding="utf-8")
    result = client.post("/api/models/reload").json()
    # v3 must never be switchable; v1/v2 stay healthy.
    assert "v3" not in result["loaded"]
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert set(health["models"]) == {"v1", "v2"}
    assert health["degraded"] is True
    issue = next(i for i in health["model_issues"] if i["file"] == "v3.json")
    assert "JSON" in issue["reason"]

    # Switching to the corrupt version is impossible (404, not 500).
    blocked = client.post("/api/model", json={"version": "v3"})
    assert blocked.status_code == 404

    # Healthy classification continues.
    classify = client.post(
        "/api/classify",
        json={"label": "live", "emails": [{"text": "free winner prize", "deadline_ms": 800}]},
    ).json()
    assert classify["entries"][0]["status"] == "ok"
    assert classify["entries"][0]["model_version"] == "v1"


def test_corrupt_default_shows_actual_enabled_version(resilient_client, monkeypatch):
    client, registry, model_dir = resilient_client
    (model_dir / "v1.json").write_text("half-written {", encoding="utf-8")
    summary = registry.reload(required_versions=["v1"])
    # v1 is active and in-memory, so the reload is rejected while v1 still
    # serves; emulate the real startup path instead by rebuilding the registry.
    assert summary["status"] == "unchanged_failure"

    monkeypatch.setenv("DEFAULT_MODEL_VERSION", "v1")
    fresh = ModelRegistry.load_dir(model_dir, "v1")
    assert fresh.default_version == "v2"
    main_module.registry = fresh
    main_module.worker.registry = fresh
    main_module.active_version = fresh.default_version

    models = client.get("/api/models").json()
    assert models["active"] == "v2"
    assert models["configured_default"] == "v1"
    notice = " ".join(i["reason"] for i in models["issues"])
    assert "v2" in notice and "v1" in notice
    assert models["degraded"] is True


def test_web_reload_repairs_file_then_allows_switch(resilient_client):
    client, registry, model_dir = resilient_client
    good_v2 = (ROOT / "models" / "v2.json").read_text(encoding="utf-8")
    (model_dir / "v2.json").write_text('{"format":', encoding="utf-8")

    dropped = client.post("/api/models/reload").json()
    # The broken non-active file is excluded; the serving v1 snapshot keeps
    # classifying traffic and the active version is never dropped.
    assert dropped["status"] == "reloaded"
    assert dropped["loaded"] == ["v1"]
    assert any(i["file"] == "v2.json" for i in dropped["issues"])
    assert client.get("/api/models").json()["active"] == "v1"

    # Repair through the same flow an operator would use.
    (model_dir / "v2.json").write_text(good_v2, encoding="utf-8")
    repaired = client.post("/api/models/reload").json()
    assert repaired["status"] == "reloaded"
    assert repaired["issues"] == []
    assert "v2" in repaired["loaded"]

    switched = client.post("/api/model", json={"version": "v2"})
    assert switched.status_code == 200
    assert switched.json()["active"] == "v2"


def test_failed_reload_does_not_affect_online_classification(resilient_client):
    client, registry, model_dir = resilient_client
    first = client.post(
        "/api/classify",
        json={"label": "before", "emails": [{"text": "free spam winner", "deadline_ms": 700}]},
    ).json()
    assert first["entries"][0]["status"] == "ok"
    pinned_score = first["entries"][0]["score"]

    # Drop a brand-new invalid file into the directory and reload.
    (model_dir / "v9.json").write_text("not even json", encoding="utf-8")
    result = client.post("/api/models/reload").json()
    assert result["status"] == "reloaded"  # healthy files re-adopted fine
    assert any(i["file"] == "v9.json" for i in result["issues"])

    second = client.post(
        "/api/classify",
        json={"label": "after", "emails": [{"text": "free spam winner", "deadline_ms": 700}]},
    ).json()
    assert second["entries"][0]["status"] == "ok"
    assert second["entries"][0]["score"] == pinned_score


def test_pinned_versions_survive_reload_and_later_switch(resilient_client):
    """A reload must not retarget accepted emails or disturb the active model."""
    client, registry, model_dir = resilient_client
    client.post("/api/model", json={"version": "v1"})
    job = client.post(
        "/api/jobs",
        json={
            "label": "pinned-across-reload",
            "emails": [
                {"text": "free winner prize", "deadline_ms": 1000},
                {
                    "text": "free winner prize",
                    "deadline_ms": 1000,
                    "submit_after_ms": 500,
                },
            ],
        },
    ).json()

    # Wait until the first email is accepted under v1.
    watched = None
    for _ in range(100):
        watched = client.get(f"/api/jobs/{job['id']}").json()
        if watched["entries"][0]["model_version"]:
            break
        time.sleep(0.01)
    assert watched["entries"][0]["model_version"] == "v1"

    # Reload models (no on-disk change): serving snapshot must remain intact.
    reload_result = client.post("/api/models/reload").json()
    assert reload_result["status"] == "reloaded"
    assert set(reload_result["loaded"]) == {"v1", "v2"}

    # Email #1 was accepted before any switch: it must settle on v1 against
    # its own deadline even after a reload and a later switch to v2.
    client.post("/api/model", json={"version": "v2"})
    assert client.get("/api/state").json()["active_model"] == "v2"

    for _ in range(300):
        watched = client.get(f"/api/jobs/{job['id']}").json()
        if watched["status"] == "completed":
            break
        time.sleep(0.01)
    assert watched["status"] == "completed"
    first = watched["entries"][0]
    assert first["model_version"] == "v1"
    assert first["status"] in ("ok", "timeout")

    # Requests accepted after the switch use v2 immediately.
    after = client.post(
        "/api/classify",
        json={
            "label": "after-switch",
            "emails": [{"text": "free winner prize", "deadline_ms": 800}],
        },
    ).json()
    assert after["entries"][0]["model_version"] == "v2"
    assert after["entries"][0]["status"] == "ok"


def test_all_corrupt_directory_fails_at_startup(model_dir: Path):
    for name in ("v1.json", "v2.json"):
        (model_dir / name).write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError):
        ModelRegistry.load_dir(model_dir, "v1")
