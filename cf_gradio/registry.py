from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .params import WorkflowParameterSpec, load_workflow_parameter_spec
from .workflow import (
    KIND_UNKNOWN,
    WorkflowCapabilities,
    detect_capabilities,
    extract_prompt_and_extra,
    read_json,
)

logger = logging.getLogger(__name__)


def slugify(value: str) -> str:
    text = str(value or "").strip()
    if text.lower().endswith("_api"):
        text = text[:-4]
    text = re.sub(r"[^0-9A-Za-z]+", "-", text).strip("-").lower()
    return text or "workflow"


def pretty_name(stem: str) -> str:
    text = stem[:-4] if stem.lower().endswith("_api") else stem
    return text.replace("_", " ").strip() or stem


@dataclass(frozen=True)
class WorkflowEntry:
    name: str
    stem: str
    model_id: str
    display_name: str
    path: Path
    mtime_ns: int
    workflow_obj: Any
    capabilities: WorkflowCapabilities
    parameter_spec: WorkflowParameterSpec | None = None
    parameter_error: str | None = None

    @property
    def kind(self) -> str:
        return self.capabilities.kind

    @property
    def kinds(self) -> tuple[str, ...]:
        return self.capabilities.kinds

    def supports(self, kind: str) -> bool:
        return kind in self.capabilities.kinds

    @property
    def ready(self) -> bool:
        return self.kind != KIND_UNKNOWN


@dataclass(frozen=True)
class WorkflowLoadError:
    name: str
    path: Path
    mtime_ns: int
    error: str


class WorkflowRegistry:
    """Scans a directory of ComfyUI API workflows on demand.

    No background watcher: every `refresh()` only re-parses files whose mtime
    changed, so calling it from the UI, `/v1/models`, or a job is cheap and
    keeps newly dropped workflows usable without a restart.
    """

    def __init__(self, workflows_dir: Path) -> None:
        self.workflows_dir = Path(workflows_dir)
        self._lock = threading.RLock()
        self._entries: dict[str, WorkflowEntry] = {}
        self._errors: dict[str, WorkflowLoadError] = {}
        self._loaded = False

    def refresh(self) -> None:
        with self._lock:
            self.workflows_dir.mkdir(parents=True, exist_ok=True)
            cached = dict(self._entries)
            loaded: list[WorkflowEntry] = []
            errors: dict[str, WorkflowLoadError] = {}

            for path in sorted(self.workflows_dir.glob("*.json")):
                if not path.is_file():
                    continue
                stat = path.stat()
                mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
                previous = cached.get(path.name)
                if previous is not None and previous.mtime_ns == mtime_ns:
                    loaded.append(previous)
                    continue
                try:
                    loaded.append(self._load_one(path, mtime_ns))
                except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                    logger.warning("工作流加载失败: %s: %s", path, exc)
                    errors[path.name] = WorkflowLoadError(
                        name=path.name,
                        path=path.resolve(),
                        mtime_ns=mtime_ns,
                        error=f"{type(exc).__name__}: {exc}",
                    )

            self._assign_model_ids(loaded)
            self._entries = {entry.name: entry for entry in loaded}
            self._errors = errors
            self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.refresh()

    def _load_one(self, path: Path, mtime_ns: int) -> WorkflowEntry:
        workflow_obj = read_json(path)
        prompt, _extra = extract_prompt_and_extra(workflow_obj)
        capabilities = detect_capabilities(prompt)

        parameter_spec: WorkflowParameterSpec | None = None
        parameter_error: str | None = None
        try:
            parameter_spec = load_workflow_parameter_spec(
                workflows_dir=self.workflows_dir,
                workflow_path=path,
                expected_kinds=capabilities.kinds,
            )
        except Exception as exc:  # noqa: BLE001 - keep the workflow usable
            parameter_error = str(exc)

        display_name = (
            parameter_spec.display_name
            if parameter_spec and parameter_spec.display_name
            else pretty_name(path.stem)
        )
        return WorkflowEntry(
            name=path.name,
            stem=path.stem,
            model_id="",
            display_name=display_name,
            path=path.resolve(),
            mtime_ns=mtime_ns,
            workflow_obj=workflow_obj,
            capabilities=capabilities,
            parameter_spec=parameter_spec,
            parameter_error=parameter_error,
        )

    @staticmethod
    def _assign_model_ids(entries: list[WorkflowEntry]) -> None:
        """Fill `model_id` in place, resolving duplicates deterministically.

        Sidecars may declare an explicit `model`; otherwise a readable slug is
        derived from the filename (``moss_image_aio_api`` -> ``moss-image-aio``).
        """

        used: set[str] = set()
        for entry in sorted(entries, key=lambda item: item.name.lower()):
            preferred = (
                entry.parameter_spec.model
                if entry.parameter_spec and entry.parameter_spec.model
                else entry.stem
            )
            base = slugify(preferred)
            candidate = base
            index = 2
            while candidate in used:
                candidate = f"{base}-{index}"
                index += 1
            used.add(candidate)
            object.__setattr__(entry, "model_id", candidate)

    def entries(self) -> list[WorkflowEntry]:
        with self._lock:
            self.ensure_loaded()
            return sorted(self._entries.values(), key=lambda item: item.name.lower())

    def ready_entries(self) -> list[WorkflowEntry]:
        return [entry for entry in self.entries() if entry.ready]

    def load_errors(self) -> list[WorkflowLoadError]:
        with self._lock:
            self.ensure_loaded()
            return list(self._errors.values())

    def get(self, key: str) -> WorkflowEntry | None:
        requested = str(key or "").strip()
        if not requested:
            return None
        lowered = requested.lower()
        with self._lock:
            self.ensure_loaded()
            for entry in self._entries.values():
                if entry.model_id == lowered:
                    return entry
            for entry in self._entries.values():
                if lowered in {
                    entry.name.lower(),
                    entry.stem.lower(),
                    f"{entry.stem.lower()}.json",
                }:
                    return entry
            if not lowered.endswith(".json"):
                maybe = f"{lowered}.json"
                for entry in self._entries.values():
                    if entry.name.lower() == maybe:
                        return entry
        return None

    def by_kind(self, kinds: Iterable[str]) -> list[WorkflowEntry]:
        allowed = set(kinds)
        return [
            entry
            for entry in self.ready_entries()
            if any(kind in allowed for kind in entry.capabilities.kinds)
        ]

    def kind_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.ready_entries():
            counts[entry.kind] = counts.get(entry.kind, 0) + 1
        return counts

    def model_objects(self) -> list[dict[str, Any]]:
        objects = []
        for entry in self.ready_entries():
            objects.append(
                {
                    "id": entry.model_id,
                    "object": "model",
                    "created": int(entry.mtime_ns / 1_000_000_000),
                    "owned_by": "cf-gradio",
                    "kind": entry.kind,
                    "kinds": list(entry.capabilities.kinds),
                    "workflow": entry.name,
                    "display_name": entry.display_name,
                    "parameter_error": entry.parameter_error,
                }
            )
        return objects
