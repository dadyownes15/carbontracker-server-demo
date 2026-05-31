from __future__ import annotations

import io
import json
import sys

from server.carbontracker_utils import ExternalUsage, make_request_id, tracked_request


def marker_payloads(buffer: io.StringIO) -> list[dict[str, object]]:
    payloads = []
    for line in buffer.getvalue().splitlines():
        assert line.startswith("carbontracker:")
        payloads.append(json.loads(line.removeprefix("carbontracker:")))
    return payloads


def test_request_ids_are_unique_and_include_clean_prefix() -> None:
    first = make_request_id("turn 001")
    second = make_request_id("turn 001")

    assert first.startswith("turn-001-")
    assert second.startswith("turn-001-")
    assert first != second


def test_markers_use_supported_fields_and_nested_parent(monkeypatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)

    with tracked_request("callbuddy_request", request_id="request-abc") as trace:
        with trace.span("llm_generate") as span:
            span.set_external_usage(
                ExternalUsage(
                    energy_kwh=0.001,
                    emissions_g=0.065,
                    carbon_intensity_g_per_kwh=65.0,
                )
            )

    request_start, generate_start, generate_stop, request_stop = marker_payloads(buffer)

    assert request_start["type"] == "start"
    assert request_start["span_id"] == "callbuddy_request_request-abc"
    assert request_start["parent_span_id"] == "process"

    assert generate_start["type"] == "start"
    assert generate_start["span_id"] == "llm_generate_request-abc"
    assert generate_start["parent_span_id"] == "callbuddy_request_request-abc"
    assert set(generate_start) == {"type", "span_id", "parent_span_id", "timestamp"}

    assert generate_stop["type"] == "stop"
    assert generate_stop["external_energy_kwh"] == 0.001
    assert generate_stop["external_emissions_g"] == 0.065
    assert generate_stop["external_carbon_intensity_g_per_kwh"] == 65.0
    assert "parent_span_id" not in generate_stop

    assert request_stop["type"] == "stop"
    assert set(request_stop) == {"type", "span_id", "timestamp"}
