from __future__ import annotations

import os
from dataclasses import dataclass


FP32_MODEL_ID = os.environ.get(
    "CALLBUDDY_FP32_MODEL_ID",
    "Intel/bert-base-uncased-mrpc",
)
INT8_MODEL_ID = os.environ.get(
    "CALLBUDDY_INT8_MODEL_ID",
    "Intel/bert-base-uncased-mrpc-int8-static-inc",
)

# Demo constants for external LLM accounting. The local machine measures the
# gate and orchestration; these constants represent remote LLM token work.
LLM_KWH_PER_TOKEN = float(os.environ.get("CALLBUDDY_LLM_KWH_PER_TOKEN", "0.0000002"))
CARBON_INTENSITY_G_PER_KWH = float(
    os.environ.get("CALLBUDDY_CARBON_INTENSITY_G_PER_KWH", "65.0")
)

FIXTURE_PATH = os.environ.get("CALLBUDDY_FIXTURE_PATH", "data/interactions.jsonl")
RESULTS_DIR = os.environ.get("CALLBUDDY_RESULTS_DIR", "results")
LOGS_DIR = os.environ.get("CALLBUDDY_LOGS_DIR", "logs")


@dataclass(frozen=True)
class Scenario:
    name: str
    npm_script: str
    port: int
    run_name: str
    requires_gate: bool
    gate_variant: str | None = None


SCENARIOS = (
    Scenario(
        name="always",
        npm_script="server:always",
        port=8020,
        run_name="always-generate",
        requires_gate=False,
    ),
    Scenario(
        name="gate_fp32",
        npm_script="server:gate-fp32",
        port=8021,
        run_name="gated-fp32-bert",
        requires_gate=True,
        gate_variant="fp32",
    ),
    Scenario(
        name="gate_int8_static",
        npm_script="server:gate-int8",
        port=8022,
        run_name="gated-static-int8-bert",
        requires_gate=True,
        gate_variant="int8_static",
    ),
)
