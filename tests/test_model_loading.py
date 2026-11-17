"""Fault-tolerant model loading: corrupt artifacts never take down healthy ones.

Covers the operations contract:

* startup reads every ``*.json`` independently; one bad file is reported with
  its file name and reason while the other versions still serve;
* a broken configured default falls back to a healthy version and says so;
* only an entirely broken directory raises a startup error;
* half-written / schema-invalid files never enter the switchable list;
* reload after an on-disk repair adopts the repaired model, and a failed
  reload never disturbs the models classifying traffic online.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.model import ModelRegistry  # noqa: E402


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    shutil.copy(ROOT / "models" / "v1.json", tmp_path / "v1.json")
    shutil.copy(ROOT / "models" / "v2.json", tmp_path / "v2.json")
    return tmp_path


def test_healthy_directory_loads_both_models(model_dir: Path):
    registry = ModelRegistry.load_dir(model_dir, "v1")
    assert registry.versions == ["v1", "v2"]
    assert registry.default_version == "v1"
    assert registry.source_file("v1") == "v1.json"
    assert registry.issues == []


def test_one_corrupt_file_others_keep_serving(model_dir: Path):
    # Simulate a half-written artifact: invalid JSON truncated mid-object.
    (model_dir / "v3.json").write_text(
        '{"format": "bow-linear.v1", "version": "v3", "vocabul', encoding="utf-8"
    )
    registry = ModelRegistry.load_dir(model_dir, "v1")

    assert registry.versions == ["v1", "v2"]  # corrupt file excluded
    assert "v3" not in registry
    issue = next(i for i in registry.issues if i.file == "v3.json")
    assert issue.status == "invalid"
    assert "JSON" in issue.reason
    # Healthy models still infer.
    prediction = registry.get("v1").predict("free winner prize")
    assert prediction.label in {"spam", "ham"}


def test_invalid_schema_never_enters_switchable_list(model_dir: Path):
    (model_dir / "bad.json").write_text(
        json.dumps(
            {
                "format": "bow-linear.v1",
                "version": "broken",
                "vocabulary": ["a"],
                "idf": [1.0],
                "weights": [1.0, 2.0],  # length mismatch with vocabulary
                "bias": 0.0,
            }
        ),
        encoding="utf-8",
    )
    registry = ModelRegistry.load_dir(model_dir)
    assert registry.versions == ["v1", "v2"]
    assert any(i.file == "bad.json" and "mismatch" in i.reason for i in registry.issues)


def test_unsupported_format_rejected(model_dir: Path):
    (model_dir / "weird.json").write_text(
        json.dumps({"format": "something-else", "version": "v9"}), encoding="utf-8"
    )
    registry = ModelRegistry.load_dir(model_dir)
    assert "v9" not in registry
    assert any(i.file == "weird.json" for i in registry.issues)


def test_broken_configured_default_falls_back_to_healthy(model_dir: Path):
    (model_dir / "v1.json").write_text("{{not json", encoding="utf-8")
    registry = ModelRegistry.load_dir(model_dir, "v1")

    assert registry.versions == ["v2"]
    assert registry.default_version == "v2"  # actually enabled healthy version
    reasons = " ".join(i.reason for i in registry.issues)
    assert "v1" in reasons and "v2" in reasons


def test_all_corrupt_raises_startup_error(model_dir: Path):
    for name in ("v1.json", "v2.json"):
        (model_dir / name).write_text("truncated {", encoding="utf-8")
    with pytest.raises(RuntimeError) as excinfo:
        ModelRegistry.load_dir(model_dir, "v1")
    message = str(excinfo.value)
    assert "v1.json" in message and "v2.json" in message


def test_missing_directory_raises(model_dir: Path):
    with pytest.raises(RuntimeError):
        ModelRegistry.load_dir(model_dir / "does-not-exist")


def test_reload_repairs_file_and_keeps_others_online(model_dir: Path):
    # Start with only v1 healthy; v2 is a half-written file.
    original_v2 = (ROOT / "models" / "v2.json").read_text(encoding="utf-8")
    (model_dir / "v2.json").write_text('{"format": "bow-linear.v1", "version":', encoding="utf-8")
    registry = ModelRegistry.load_dir(model_dir, "v1")
    assert registry.versions == ["v1"]
    assert any(i.file == "v2.json" for i in registry.issues)

    # Online classification on v1 works before the repair.
    before = registry.get("v1").predict("free winner")

    # A reload while v2 is still broken re-adopts the healthy snapshot and
    # keeps excluding v2; the active v1 model is untouched.
    failed = registry.reload(required_versions=["v1"])
    assert failed["status"] == "reloaded"
    assert registry.versions == ["v1"]
    assert any(i["file"] == "v2.json" for i in failed["issues"])
    assert registry.get("v1").predict("free winner").score == before.score

    # Operator repairs the file; the next reload adopts v2.
    (model_dir / "v2.json").write_text(original_v2, encoding="utf-8")
    summary = registry.reload(required_versions=["v1"])
    assert summary["status"] == "reloaded"
    assert registry.versions == ["v1", "v2"]
    assert summary["added"] == ["v2"]
    assert registry.source_file("v2") == "v2.json"
    # v1 keeps serving exactly as before (atomic swap, same artifact).
    assert registry.get("v1").predict("free winner").score == before.score


def test_reload_rejects_snapshot_that_drops_active_model(model_dir: Path):
    registry = ModelRegistry.load_dir(model_dir, "v1")
    assert registry.versions == ["v1", "v2"]
    # Corrupt the active model's file on disk.
    (model_dir / "v1.json").write_text("garbage", encoding="utf-8")

    summary = registry.reload(required_versions=["v1"])
    assert summary["status"] == "unchanged_failure"
    # The in-process v1 must remain available for already-accepted traffic.
    assert registry.versions == ["v1", "v2"]
    assert registry.get("v1").predict("winner free").label in {"spam", "ham"}


def test_reload_with_zero_healthy_files_keeps_last_good_snapshot(model_dir: Path):
    registry = ModelRegistry.load_dir(model_dir, "v1")
    for name in ("v1.json", "v2.json"):
        (model_dir / name).write_text("nope", encoding="utf-8")
    summary = registry.reload(required_versions=["v1"])
    assert summary["status"] == "unchanged_failure"
    assert registry.versions == ["v1", "v2"]  # last good snapshot stays online
    assert registry.get("v2") is not None


def test_duplicate_version_in_two_files_is_flagged(model_dir: Path):
    payload = json.loads((model_dir / "v2.json").read_text(encoding="utf-8"))
    payload["version"] = "v1"  # collides with v1.json
    # A name sorting after the originals, so the shipped v1.json wins.
    (model_dir / "zz_copy.json").write_text(json.dumps(payload), encoding="utf-8")
    registry = ModelRegistry.load_dir(model_dir)
    assert registry.versions == ["v1", "v2"]
    assert any(i.file == "zz_copy.json" and "重复" in i.reason for i in registry.issues)
