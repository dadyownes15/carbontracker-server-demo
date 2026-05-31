from __future__ import annotations

import os
import signal
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from server.carbontracker_utils import ExternalUsage, make_request_id, tracked_request
from server.config import CARBON_INTENSITY_G_PER_KWH, SCENARIOS
from server.gate_model import GateModel, get_gate_model
from server.llm import generate_suggestion, zero_usage


MODE = os.environ.get("CALLBUDDY_MODE", "always")
SCENARIO = next((scenario for scenario in SCENARIOS if scenario.name == MODE), None)

if SCENARIO is None:
    supported = ", ".join(scenario.name for scenario in SCENARIOS)
    raise RuntimeError(f"Unsupported CALLBUDDY_MODE={MODE!r}; expected one of {supported}")


class SuggestRequest(BaseModel):
    interaction_id: str = Field(min_length=1)
    seller_text: str = Field(min_length=1)
    customer_text: str = Field(min_length=1)
    demo_gate_pass: bool = False


STATE: dict[str, GateModel | None] = {"gate_model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if SCENARIO.requires_gate:
        assert SCENARIO.gate_variant is not None
        model = get_gate_model(SCENARIO.gate_variant)
        model.warmup()
        STATE["gate_model"] = model
    yield


app = FastAPI(
    title="Callbuddy gated LLM CarbonTracker demo",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    gate_model = STATE["gate_model"]
    return {
        "status": "ok",
        "mode": SCENARIO.name,
        "requires_gate": SCENARIO.requires_gate,
        "gate_variant": SCENARIO.gate_variant,
        "gate_model_id": gate_model.model_id if gate_model else None,
    }


@app.post("/suggest")
def suggest(payload: SuggestRequest) -> dict[str, Any]:
    started_at = time.perf_counter()
    request_id = make_request_id(payload.interaction_id)
    generated = False
    suggestion: str | None = None
    gate_result = None
    usage = zero_usage()

    with tracked_request("callbuddy_request", request_id=request_id) as trace:
        should_generate = True
        if SCENARIO.requires_gate:
            # State variable to compare scenarious
            gate_model = STATE["gate_model"]
            if gate_model is None:
                raise HTTPException(status_code=503, detail="Gate model is not loaded")
            with trace.span("llm_gate"):
                gate_result = gate_model.predict(
                    seller_text=payload.seller_text,
                    customer_text=payload.customer_text,
                )
            # Demo control: the model is actually run for measurement, but this
            # fixture field keeps the pass rate exactly at 50% for comparison.
            should_generate = payload.demo_gate_pass

        if should_generate:
            with trace.span("llm_generate") as span:
                suggestion, usage = generate_suggestion(
                    seller_text=payload.seller_text,
                    customer_text=payload.customer_text,
                    gate_label=gate_result.label if gate_result else None,
                )
                span.set_external_usage(
                    ExternalUsage(
                        energy_kwh=usage.external_energy_kwh,
                        emissions_g=usage.external_emissions_g,
                        carbon_intensity_g_per_kwh=CARBON_INTENSITY_G_PER_KWH,
                    )
                )
            generated = True

    return {
        "request_id": request_id,
        "mode": SCENARIO.name,
        "interaction_id": payload.interaction_id,
        "generated": generated,
        "demo_gate_pass": payload.demo_gate_pass,
        "gate": gate_result.to_dict() if gate_result else None,
        "suggestion": suggestion,
        "usage": usage.to_dict(),
        "duration_ms": (time.perf_counter() - started_at) * 1000.0,
    }


@app.post("/shutdown")
def shutdown() -> dict[str, str]:
    def stop_process() -> None:
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=stop_process, daemon=True).start()
    return {"status": "shutting_down"}
