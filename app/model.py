"""Versioned bag-of-words linear spam classifier.

Deterministic, dependency-light inference (NumPy only) so that the same
model artifacts produce byte-identical scores on the gateway, the worker
and in tests.  Two distinct English bag-of-words models ship with the repo;
each model version pins its own vocabulary, IDF weights and linear
coefficients.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")

# A small, obviously distinct stopword-ish filter.  Kept identical across
# versions on purpose: the *model* differs, not the tokenizer contract.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have i in is it its of on or that "
    "the this to was were will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization for the English bag-of-words models."""
    return [tok for tok in TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS]


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


@dataclass(frozen=True)
class Prediction:
    score: float
    label: str
    model_version: str

    def as_dict(self) -> dict:
        return {
            "score": round(float(self.score), 6),
            "label": self.label,
            "model_version": self.model_version,
        }


class LinearBoWModel:
    """Binary bag-of-words logistic model with a fixed vector vocabulary."""

    def __init__(
        self,
        version: str,
        vocabulary: list[str],
        idf: np.ndarray,
        weights: np.ndarray,
        bias: float,
        threshold: float,
        description: str = "",
    ) -> None:
        if len(vocabulary) != len(idf) or len(vocabulary) != len(weights):
            raise ValueError(f"model {version}: vocabulary/idf/weights length mismatch")
        self.version = version
        self.vocabulary = tuple(vocabulary)
        self.idf = np.asarray(idf, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        self.threshold = float(threshold)
        self.description = description
        self._index = {tok: i for i, tok in enumerate(self.vocabulary)}

    # -- serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": "bow-linear.v1",
            "version": self.version,
            "description": self.description,
            "threshold": self.threshold,
            "bias": self.bias,
            "vocabulary": list(self.vocabulary),
            "idf": self.idf.tolist(),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LinearBoWModel":
        if payload.get("format") != "bow-linear.v1":
            raise ValueError(f"unsupported model format: {payload.get('format')!r}")
        return cls(
            version=payload["version"],
            vocabulary=payload["vocabulary"],
            idf=np.asarray(payload["idf"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=payload["bias"],
            threshold=payload.get("threshold", 0.5),
            description=payload.get("description", ""),
        )

    # -- inference -----------------------------------------------------
    def featurize(self, tokens: Iterable[str]) -> np.ndarray:
        """Sublinear tf * idf over the *frozen* vocabulary of this version."""
        counts = np.zeros(len(self.vocabulary), dtype=np.float64)
        known = 0
        total = 0
        for tok in tokens:
            total += 1
            idx = self._index.get(tok)
            if idx is not None:
                counts[idx] += 1.0
                known += 1
        if known:
            present = counts > 0
            counts[present] = 1.0 + np.log(counts[present])
        counts /= max(1.0, float(np.sqrt(max(total, 1))))
        return counts * self.idf

    def score_tokens(self, tokens: list[str]) -> float:
        margin = float(np.dot(self.featurize(tokens), self.weights) + self.bias)
        # numerically stable logistic
        if margin >= 0:
            return 1.0 / (1.0 + float(np.exp(-margin)))
        exp = float(np.exp(margin))
        return exp / (1.0 + exp)

    def predict(self, text: str) -> Prediction:
        score = self.score_tokens(tokenize(text))
        label = "spam" if score >= self.threshold else "ham"
        return Prediction(score=score, label=label, model_version=self.version)


@dataclass
class ModelLoadIssue:
    """Why one model artifact could not be loaded (or was unloaded)."""

    file: str
    status: str  # invalid | missing | unloaded
    reason: str
    attempted_version: str | None = None
    at: str = ""

    def public(self) -> dict:
        return {
            "file": self.file,
            "status": self.status,
            "reason": self.reason,
            "attempted_version": self.attempted_version,
            "at": self.at,
        }


class ModelRegistry:
    """Reloadable set of models that tolerates individual corrupt artifacts.

    Design rules (see the operations runbook in README.md):

    * Each ``*.json`` in the model directory is read independently at startup
      and on every :meth:`reload`.  One bad file never prevents the healthy
      versions from being served.
    * A file only enters the switchable set after it parses and constructs a
      :class:`LinearBoWModel` successfully.  Half-written or invalid files are
      recorded in :attr:`issues` instead.
    * The registry swaps its model map atomically, so a failed reload can never
      remove a model that is still classifying traffic online.
    * If every artifact is unusable, :meth:`load_dir` raises at startup; while
      running, :meth:`reload` keeps serving the last good snapshot and reports
      the failure instead.
    """

    def __init__(
        self,
        models: dict[str, LinearBoWModel],
        directory: str | os.PathLike[str],
        preferred_default: str | None = None,
        issues: list[ModelLoadIssue] | None = None,
    ) -> None:
        if not models:
            raise ValueError("registry needs at least one healthy model")
        self.directory = Path(directory)
        self.preferred_default = preferred_default
        self._models: dict[str, LinearBoWModel] = dict(models)
        self._files: dict[str, str] = {}
        self._issues: list[ModelLoadIssue] = list(issues or [])
        self.default_version = self._resolve_default(preferred_default)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    @classmethod
    def load_dir(
        cls,
        directory: str | os.PathLike[str],
        default_version: str | None = None,
    ) -> "ModelRegistry":
        """Load every artifact, tolerating per-file failures.

        Raises ``RuntimeError`` only when no healthy model could be loaded or
        the model directory itself is unusable.
        """
        path = Path(directory)
        models, files, issues = cls._scan(path)
        if not models:
            detail = "; ".join(f"{i.file}: {i.reason}" for i in issues) or "目录中没有 JSON 文件"
            raise RuntimeError(f"模型目录 {path} 中没有可用模型：{detail}")
        registry = cls(
            models, path, preferred_default=default_version, issues=issues
        )
        registry._files = files
        return registry

    @staticmethod
    def _scan(
        path: Path,
    ) -> tuple[dict[str, LinearBoWModel], dict[str, str], list[ModelLoadIssue]]:
        models: dict[str, LinearBoWModel] = {}
        files: dict[str, str] = {}
        issues: list[ModelLoadIssue] = []
        if not path.exists():
            raise RuntimeError(f"模型目录不存在: {path}")
        if not path.is_dir():
            raise RuntimeError(f"模型路径不是目录: {path}")
        for file in sorted(path.glob("*.json")):
            payload: object = None
            try:
                # Read the whole document first so a half-written file fails
                # the JSON parse and never produces a partial model.
                with file.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                if not isinstance(payload, dict):
                    raise ValueError("顶层结构必须是 JSON 对象")
                model = LinearBoWModel.from_dict(payload)
            except json.JSONDecodeError as exc:
                issues.append(
                    ModelLoadIssue(
                        file=file.name,
                        status="invalid",
                        reason=f"JSON 解析失败（可能写入不完整）: {exc.msg} "
                        f"(行 {exc.lineno} 列 {exc.colno})",
                    )
                )
                continue
            except (OSError, UnicodeDecodeError) as exc:
                issues.append(
                    ModelLoadIssue(file=file.name, status="invalid", reason=f"文件无法读取: {exc}")
                )
                continue
            except (KeyError, TypeError, ValueError) as exc:
                attempted = payload.get("version") if isinstance(payload, dict) else None
                issues.append(
                    ModelLoadIssue(
                        file=file.name,
                        status="invalid",
                        reason=f"模型内容无效: {exc}",
                        attempted_version=attempted,
                    )
                )
                continue
            if model.version in models:
                issues.append(
                    ModelLoadIssue(
                        file=file.name,
                        status="invalid",
                        reason=f"模型版本 {model.version!r} 与已加载文件重复，本文件已忽略",
                        attempted_version=model.version,
                    )
                )
                continue
            models[model.version] = model
            files[model.version] = file.name
        return models, files, issues

    def reload(self, *, required_versions: Iterable[str] | None = None) -> dict:
        """Re-scan the model directory and atomically adopt healthy models.

        ``required_versions`` (typically the currently active version) must
        remain healthy in the new snapshot: if it is missing, the reload is
        rejected so online classification of live traffic is never disturbed
        by an operator-triggered scan.

        Returns a summary dict.  A scan that yields zero healthy models, a
        directory-level failure, or a missing required version keeps serving
        the last good snapshot and is reported via
        ``status == "unchanged_failure"``.
        """
        with self._lock:
            previous = set(self._models)
            try:
                models, files, issues = self._scan(self.directory)
            except RuntimeError as exc:
                self._issues.append(
                    ModelLoadIssue(
                        file=str(self.directory),
                        status="invalid",
                        reason=f"重新扫描失败，继续使用已加载模型: {exc}",
                    )
                )
                return {
                    "status": "unchanged_failure",
                    "loaded": list(sorted(previous)),
                    "added": [],
                    "removed": [],
                    "issues": self.issue_public(),
                    "reason": str(exc),
                }
            required = {v for v in (required_versions or []) if v}
            missing_required = sorted(required - set(models))
            if not models or missing_required:
                if missing_required:
                    reason = (
                        f"重新加载会丢失在线模型 {missing_required}，"
                        "已拒绝本次加载并继续使用当前模型"
                    )
                else:
                    reason = "重新加载后没有任何健康模型，保留当前在线模型"
                self._issues.extend(issues)
                self._issues.append(
                    ModelLoadIssue(file=str(self.directory), status="invalid", reason=reason)
                )
                return {
                    "status": "unchanged_failure",
                    "loaded": list(sorted(previous)),
                    "added": [],
                    "removed": [],
                    "issues": self.issue_public(),
                    "reason": reason,
                }
            self._models = models
            self._files = files
            self._issues = issues
            self.default_version = self._resolve_default(self.preferred_default)
            loaded = set(models)
            return {
                "status": "reloaded",
                "loaded": list(sorted(loaded)),
                "added": list(sorted(loaded - previous)),
                "removed": list(sorted(previous - loaded)),
                "issues": [i.public() for i in issues],
            }

    def _resolve_default(self, preferred: str | None) -> str:
        default = preferred or os.environ.get("DEFAULT_MODEL_VERSION")
        if default is None:
            default = "v1"
        if default not in self._models:
            fallback = sorted(self._models)[0]
            logger_note = (
                f"配置的默认模型 {default!r} 不可用，"
                f"实际启用健康版本 {fallback!r}"
            )
            self._issues.append(
                ModelLoadIssue(
                    file=str(self.directory),
                    status="invalid",
                    reason=logger_note,
                    attempted_version=default,
                )
            )
            return fallback
        return default

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------
    @property
    def versions(self) -> list[str]:
        with self._lock:
            return sorted(self._models)

    @property
    def issues(self) -> list[ModelLoadIssue]:
        with self._lock:
            return list(self._issues)

    def source_file(self, version: str) -> str | None:
        with self._lock:
            return self._files.get(version)

    def issue_public(self) -> list[dict]:
        return [issue.public() for issue in self.issues]

    def get(self, version: str) -> LinearBoWModel:
        with self._lock:
            try:
                return self._models[version]
            except KeyError:
                raise KeyError(
                    f"unknown model version {version!r}; known: {self.versions}"
                ) from None

    def __contains__(self, version: object) -> bool:
        with self._lock:
            return version in self._models
