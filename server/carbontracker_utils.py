from __future__ import annotations

import contextlib
import contextvars
import json
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Iterator


_CURRENT_PARENT_SPAN_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "carbontracker_parent_span_id",
    default="process",
)
_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class ExternalUsage:
    energy_kwh: float
    emissions_g: float
    carbon_intensity_g_per_kwh: float


@dataclass(frozen=True)
class TrackingContext:
    request_id: str
    request_span_id: str

    def span(self, name: str) -> "TrackedSpan":
        return TrackedSpan(
            name=name,
            request_id=self.request_id,
            parent_span_id=self.request_span_id,
        )


def make_request_id(interaction_id: str | None = None) -> str:
    prefix = clean_id(interaction_id or "request")
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def clean_id(value: str) -> str:
    clean = _SAFE_ID_RE.sub("-", value.strip()).strip("-")
    return clean or "span"


@contextlib.contextmanager
def tracked_request(name: str, *, request_id: str) -> Iterator[TrackingContext]:
    request_span_id = f"{clean_id(name)}_{clean_id(request_id)}"
    with TrackedSpan(
        name=name,
        request_id=request_id,
        parent_span_id="process",
        span_id=request_span_id,
    ):
        yield TrackingContext(
            request_id=request_id,
            request_span_id=request_span_id,
        )


class TrackedSpan:
    def __init__(
        self,
        *,
        name: str,
        request_id: str,
        parent_span_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        self.name = clean_id(name)
        self.request_id = clean_id(request_id)
        self.span_id = span_id or f"{self.name}_{self.request_id}"
        self.parent_span_id = parent_span_id or _CURRENT_PARENT_SPAN_ID.get()
        self.external_usage: ExternalUsage | None = None
        self._parent_token: contextvars.Token[str] | None = None

    def set_external_usage(self, usage: ExternalUsage) -> None:
        self.external_usage = usage

    def __enter__(self) -> "TrackedSpan":
        self._parent_token = _CURRENT_PARENT_SPAN_ID.set(self.span_id)
        # The subprocess observer currently accepts only these start fields.
        # Extra metadata would make the marker invalid, so request details stay
        # in the HTTP response/results file instead of the marker.
        _emit_marker(
            {
                "type": "start",
                "span_id": self.span_id,
                "parent_span_id": self.parent_span_id,
                "timestamp": _now_iso(),
            }
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        payload: dict[str, object] = {
            "type": "stop",
            "span_id": self.span_id,
            "timestamp": _now_iso(),
        }
        if self.external_usage is not None:
            # External LLM accounting is attached to the stop marker because the
            # token count is only known after generation has completed.
            payload["external_energy_kwh"] = self.external_usage.energy_kwh
            payload["external_emissions_g"] = self.external_usage.emissions_g
            payload[
                "external_carbon_intensity_g_per_kwh"
            ] = self.external_usage.carbon_intensity_g_per_kwh

        _emit_marker(payload)
        if self._parent_token is not None:
            _CURRENT_PARENT_SPAN_ID.reset(self._parent_token)


def _emit_marker(payload: dict[str, object]) -> None:
    sys.stdout.write("carbontracker:" + json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _now_iso() -> str:
    return datetime.now().isoformat()
