# Callbuddy Gated LLM CarbonTracker Demo

This standalone demo compares three Callbuddy-style response generation modes:

1. Always generate a seller suggestion.
2. Run a local FP32 BERT gate, then generate for 50% of interactions.
3. Run a local pre-quantized INT8 BERT gate, then generate for 50% of interactions.

The gate model is always executed in the gated modes so CarbonTracker measures
real local inference work. The pass/fail decision is fixed by the demo fixture
for now so the comparison stays exactly at 50%.

## Setup

```bash
uv venv
uv pip install -e /Users/mikkeldahl/carbontracker-tui
uv pip install -e .
```

The demo intentionally uses the local `carbontracker-tui` checkout through an
editable install. It does not install CarbonTracker from PyPI.

## Models

- FP32: `Intel/bert-base-uncased-mrpc`
- INT8: `Intel/bert-base-uncased-mrpc-int8-static-inc`

The INT8 model is fetched directly from Hugging Face as a prebuilt `model.onnx`
artifact and executed with ONNX Runtime. Its model metadata is tagged
`PostTrainingStatic` and `Intel Neural Compressor`; this demo does not quantize
a model at runtime.

Prefetch and warm the models before a measured run:

```bash
npm run fetch-models
```

## Run

```bash
npm run demo
```

By default, the demo replays the 12-row balanced fixture 20 times per scenario
for 240 interactions per scenario. That is intentionally longer than a smoke
test so CarbonTracker has enough span events and runtime to show the gated
effect.

To run an exact number of simulated calls/interactions per scenario:

```bash
npm run demo -- --calls 200
```

`--calls` must be even so the gated fixtures stay exactly 50% pass / 50% skip.

For a quick sanity run:

```bash
npm run demo -- --repeat 1
```

For a longer demonstration:

```bash
npm run demo -- --repeat 50
```

You can also set `CALLBUDDY_DEMO_REPEAT=50`.

`npm run demo` runs the three servers sequentially under:

```bash
uv run carbontracker track
```

That command is the local `carbontracker-tui` package installed with
`uv pip install -e /Users/mikkeldahl/carbontracker-tui`.

The runner also passes CarbonTracker duration prediction flags:

```bash
--total-duration 86400 --predict-after-seconds 2 --predict-interval 0
```

The printed table includes measured energy/CO2 and a 24 hour projection. The
projection uses the measured total for each scenario, including the external LLM
accounting attached to `llm_generate` spans. CarbonTracker prediction events are
also preserved in `results/summary.json` when emitted.

To use Electricity Maps, set one of these before running:

```bash
export ELECTRICITY_MAPS_API_KEY="..."
# or
export CARBONTRACKER_API_KEY="..."
```

You can also put either variable in a local `.env` file. The runner passes the
key to CarbonTracker through `CARBONTRACKER_API_KEY` and does not print the key.

Each scenario writes a CarbonTracker JSONL log in `logs/` and raw HTTP responses
in `results/`.

## Server Scripts

```bash
npm run server:always
npm run server:gate-fp32
npm run server:gate-int8
```

The servers expose:

- `GET /health`
- `POST /suggest`
- `POST /shutdown`

## Instrumentation

All marker emission is isolated in `server/carbontracker_utils.py`.

Span layout:

- `callbuddy_request_<request_id>`
- `llm_gate_<request_id>` for gated variants
- `llm_generate_<request_id>` when generation happens

The LLM span stop marker includes external token-derived energy and CO2 fields.
Gate spans do not include external fields because gate inference is local compute
that CarbonTracker measures directly.
# carbontracker-server-demo
