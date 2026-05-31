from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from server.config import FIXTURE_PATH, LOGS_DIR, RESULTS_DIR, SCENARIOS


PROJECT_NAME = "callbuddy-gate-demo"
DEFAULT_REPEAT = int(os.environ.get("CALLBUDDY_DEMO_REPEAT", "20"))
DEFAULT_PREDICTION_HOURS = float(os.environ.get("CALLBUDDY_PREDICTION_HOURS", "24"))
DEFAULT_PREDICT_AFTER_SECONDS = float(
    os.environ.get("CALLBUDDY_PREDICT_AFTER_SECONDS", "2")
)
SECONDS_PER_HOUR = 3600.0


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    electricity_maps_key = resolve_electricity_maps_key()
    logs_dir = Path(LOGS_DIR)
    results_dir = Path(RESULTS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Prefetching and warming gate models outside measured runs...")
    subprocess.run([sys.executable, "scripts/prefetch_models.py"], check=True)

    base_fixture = load_fixture(Path(FIXTURE_PATH))
    fixture = build_fixture(base_fixture, repeat=args.repeat, calls=args.calls)
    print_run_context(
        fixture=fixture,
        base_fixture_size=len(base_fixture),
        repeat=args.repeat,
        calls=args.calls,
        prediction_hours=args.prediction_hours,
        electricity_maps_key=electricity_maps_key,
    )
    summaries = []
    for scenario in SCENARIOS:
        summaries.append(
            run_scenario(
                scenario,
                fixture,
                logs_dir,
                results_dir,
                repeat=args.repeat,
                prediction_hours=args.prediction_hours,
                predict_after_seconds=args.predict_after_seconds,
                electricity_maps_key=electricity_maps_key,
            )
        )

    print_table(summaries)
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"\nWrote detailed summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Callbuddy gated-generation CarbonTracker demo."
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        help=(
            "Number of times to replay the balanced fixture per scenario. "
            "Use 1 for a smoke run; the default is long enough to show a "
            "clear CarbonTracker measurement."
        ),
    )
    parser.add_argument(
        "--calls",
        type=int,
        default=None,
        help=(
            "Exact number of simulated calls/interactions per scenario. "
            "Must be even so the demo pass rate remains exactly 50%."
        ),
    )
    parser.add_argument(
        "--prediction-hours",
        type=float,
        default=DEFAULT_PREDICTION_HOURS,
        help="Duration target for CarbonTracker and summary projection.",
    )
    parser.add_argument(
        "--predict-after-seconds",
        type=float,
        default=DEFAULT_PREDICT_AFTER_SECONDS,
        help="Delay before CarbonTracker emits its once-per-run prediction.",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.calls is not None and args.calls < 2:
        raise SystemExit("--calls must be >= 2")
    if args.calls is not None and args.calls % 2 != 0:
        raise SystemExit("--calls must be even to keep the demo pass rate at 50%")
    if args.prediction_hours <= 0:
        raise SystemExit("--prediction-hours must be > 0")
    if args.predict_after_seconds < 0:
        raise SystemExit("--predict-after-seconds must be >= 0")
    return args


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def resolve_electricity_maps_key() -> str | None:
    return os.environ.get("ELECTRICITY_MAPS_API_KEY") or os.environ.get(
        "CARBONTRACKER_API_KEY"
    )


def load_fixture(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No fixture rows found in {path}")
    passes = sum(1 for row in rows if row.get("demo_gate_pass"))
    if passes * 2 != len(rows):
        raise RuntimeError(
            f"Fixture must be exactly 50% pass; got {passes}/{len(rows)}"
        )
    return rows


def expand_fixture(rows: list[dict[str, Any]], *, repeat: int) -> list[dict[str, Any]]:
    expanded = []
    for repeat_index in range(repeat):
        for row in rows:
            repeated_row = dict(row)
            repeated_row["interaction_id"] = (
                f"{row['interaction_id']}_r{repeat_index + 1:03d}"
            )
            expanded.append(repeated_row)
    return expanded


def build_fixture(
    rows: list[dict[str, Any]],
    *,
    repeat: int,
    calls: int | None,
) -> list[dict[str, Any]]:
    if calls is None:
        return expand_fixture(rows, repeat=repeat)

    pass_rows = [row for row in rows if row["demo_gate_pass"]]
    skip_rows = [row for row in rows if not row["demo_gate_pass"]]
    if not pass_rows or not skip_rows:
        raise RuntimeError("Fixture must contain both pass and skip rows")

    expanded: list[dict[str, Any]] = []
    pairs = calls // 2
    for index in range(pairs):
        skip_row = dict(skip_rows[index % len(skip_rows)])
        pass_row = dict(pass_rows[index % len(pass_rows)])
        skip_row["interaction_id"] = f"{skip_row['interaction_id']}_c{index + 1:05d}"
        pass_row["interaction_id"] = f"{pass_row['interaction_id']}_c{index + 1:05d}"
        expanded.extend([skip_row, pass_row])
    return expanded


def print_run_context(
    *,
    fixture: list[dict[str, Any]],
    base_fixture_size: int,
    repeat: int,
    calls: int | None,
    prediction_hours: float,
    electricity_maps_key: str | None,
) -> None:
    if calls is None:
        print(
            f"Running {len(fixture)} interactions per scenario "
            f"({base_fixture_size} fixture rows x {repeat} repeats)."
        )
    else:
        print(f"Running {len(fixture)} interactions per scenario (--calls {calls}).")
    print(f"Including {prediction_hours:g} hour projection.")
    if electricity_maps_key:
        print("Using Electricity Maps key from environment or .env.")
    else:
        print(
            "No Electricity Maps key found in ELECTRICITY_MAPS_API_KEY, "
            "CARBONTRACKER_API_KEY, or .env; CarbonTracker will use configured fallback."
        )


def run_scenario(
    scenario,
    fixture: list[dict[str, Any]],
    logs_dir: Path,
    results_dir: Path,
    *,
    repeat: int,
    prediction_hours: float,
    predict_after_seconds: float,
    electricity_maps_key: str | None,
) -> dict[str, Any]:
    log_path = logs_dir / f"{scenario.name}_events.jsonl"
    responses_path = results_dir / f"{scenario.name}_responses.json"
    for path in (log_path, responses_path):
        if path.exists():
            path.unlink()

    cmd = [
        "uv",
        "run",
        "carbontracker",
        "track",
        "--project-name",
        PROJECT_NAME,
        "--run-name",
        scenario.run_name,
        "--log-dir",
        str(logs_dir),
        "--jsonl",
        str(log_path),
        "--intensity-method",
        "auto",
        "--total-duration",
        str(prediction_hours * SECONDS_PER_HOUR),
        "--predict-after-seconds",
        str(predict_after_seconds),
        "--predict-interval",
        "0",
        "npm",
        "run",
        scenario.npm_script,
    ]

    env = os.environ.copy()
    if electricity_maps_key:
        env["CARBONTRACKER_API_KEY"] = electricity_maps_key

    print(f"\nStarting {scenario.name}: {' '.join(cmd)}")
    started_at = time.perf_counter()
    proc = subprocess.Popen(cmd, start_new_session=True, env=env)
    try:
        wait_for_health(scenario.port)
        responses = []
        for index, row in enumerate(fixture, start=1):
            responses.append(post_json(scenario.port, "/suggest", row))
            if index % 50 == 0 or index == len(fixture):
                print(f"{scenario.name}: sent {index}/{len(fixture)} interactions")
        responses_path.write_text(json.dumps(responses, indent=2) + "\n")
        post_json(scenario.port, "/shutdown", {})
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)

    if proc.returncode not in (0, -signal.SIGTERM):
        raise RuntimeError(f"Scenario {scenario.name} exited with {proc.returncode}")

    return summarize_scenario(
        scenario,
        fixture,
        responses_path,
        log_path,
        repeat=repeat,
        duration_s=time.perf_counter() - started_at,
        prediction_hours=prediction_hours,
    )


def wait_for_health(port: int, timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            health = get_json(port, "/health", timeout_s=5.0)
            if health.get("status") == "ok":
                return
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for server on port {port}: {last_error}")


def get_json(port: int, path: str, *, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{path}",
        timeout=timeout_s,
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(port: int, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"POST {path} failed with {exc.code}: {detail}") from exc


def summarize_scenario(
    scenario,
    fixture: list[dict[str, Any]],
    responses_path: Path,
    log_path: Path,
    *,
    repeat: int,
    duration_s: float,
    prediction_hours: float,
) -> dict[str, Any]:
    responses = json.loads(responses_path.read_text())
    generated_count = sum(1 for response in responses if response["generated"])
    total_requests = len(fixture)
    total_tokens = sum(response["usage"]["total_tokens"] for response in responses)
    span_totals = span_energy_totals(log_path)
    prediction = latest_prediction(log_path)
    projected = project_to_duration(
        energy_kwh=span_totals["local_energy_kwh"] + span_totals["external_energy_kwh"],
        co2eq_g=span_totals["local_emissions_g"] + span_totals["external_emissions_g"],
        measured_duration_s=duration_s,
        prediction_hours=prediction_hours,
    )

    return {
        "scenario": scenario.name,
        "label": label_for_scenario(scenario.name),
        "requests": total_requests,
        "fixture_repeat": repeat,
        "generated": generated_count,
        "detection_rate": generated_count / total_requests,
        "average_tokens": total_tokens / total_requests,
        "duration_s": duration_s,
        "prediction_hours": prediction_hours,
        "carbontracker_prediction": prediction,
        "projected_energy_kwh": projected["energy_kwh"],
        "projected_co2eq_g": projected["co2eq_g"],
        "local_energy_kwh": span_totals["local_energy_kwh"],
        "local_emissions_g": span_totals["local_emissions_g"],
        "external_energy_kwh": span_totals["external_energy_kwh"],
        "external_emissions_g": span_totals["external_emissions_g"],
        "energy_kwh": span_totals["local_energy_kwh"] + span_totals["external_energy_kwh"],
        "co2eq_g": span_totals["local_emissions_g"] + span_totals["external_emissions_g"],
        "log_path": str(log_path),
        "responses_path": str(responses_path),
    }


def span_energy_totals(log_path: Path) -> dict[str, float]:
    totals = {
        "local_energy_kwh": 0.0,
        "local_emissions_g": 0.0,
        "external_energy_kwh": 0.0,
        "external_emissions_g": 0.0,
    }
    prefixes = ("llm_gate_", "llm_generate_")
    if not log_path.exists():
        return totals

    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("__type__") != "SpanProfileEvent":
            continue
        span_id = event.get("span_id", "")
        if not any(span_id.startswith(prefix) for prefix in prefixes):
            continue
        stats = event.get("stats") or {}
        totals["local_energy_kwh"] += float(stats.get("power_usage_kwh") or 0.0)
        totals["local_emissions_g"] += float(stats.get("emissions_g") or 0.0)
        external = event.get("external_accounting") or {}
        totals["external_energy_kwh"] += float(external.get("energy_kwh") or 0.0)
        totals["external_emissions_g"] += float(external.get("emissions_g") or 0.0)
    return totals


def latest_prediction(log_path: Path) -> dict[str, Any] | None:
    latest = None
    if not log_path.exists():
        return None
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("__type__") == "PredictionEvent":
            latest = event.get("result")
    return latest


def project_to_duration(
    *,
    energy_kwh: float,
    co2eq_g: float,
    measured_duration_s: float,
    prediction_hours: float,
) -> dict[str, float]:
    target_duration_s = prediction_hours * SECONDS_PER_HOUR
    if measured_duration_s <= 0:
        return {"energy_kwh": 0.0, "co2eq_g": 0.0}
    multiplier = target_duration_s / measured_duration_s
    return {
        "energy_kwh": energy_kwh * multiplier,
        "co2eq_g": co2eq_g * multiplier,
    }


def label_for_scenario(name: str) -> str:
    return {
        "always": "Always generate",
        "gate_fp32": "Gated FP32 BERT",
        "gate_int8_static": "Gated Static INT8 BERT",
    }[name]


def print_table(summaries: list[dict[str, Any]]) -> None:
    print(
        "\n| Scenario | Detection rate | Average tokens | Energy | CO2eq | "
        "24h Energy | 24h CO2eq |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        print(
            f"| {summary['label']} "
            f"| {summary['detection_rate']:.0%} "
            f"| {summary['average_tokens']:.1f} "
            f"| {summary['energy_kwh']:.8f} kWh "
            f"| {summary['co2eq_g']:.6f} g "
            f"| {summary['projected_energy_kwh']:.6f} kWh "
            f"| {summary['projected_co2eq_g']:.3f} g |"
        )


if __name__ == "__main__":
    main()
