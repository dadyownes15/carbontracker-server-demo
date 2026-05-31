from __future__ import annotations

import pytest

from scripts.run_demo import build_fixture, project_to_duration


BASE_ROWS = [
    {
        "interaction_id": "skip_1",
        "seller_text": "Seller",
        "customer_text": "Customer",
        "demo_gate_pass": False,
    },
    {
        "interaction_id": "pass_1",
        "seller_text": "Seller",
        "customer_text": "Customer",
        "demo_gate_pass": True,
    },
]


def test_build_fixture_exact_calls_stays_balanced() -> None:
    rows = build_fixture(BASE_ROWS, repeat=1, calls=10)

    assert len(rows) == 10
    assert sum(1 for row in rows if row["demo_gate_pass"]) == 5
    assert len({row["interaction_id"] for row in rows}) == 10


def test_projection_to_24_hours_scales_measured_total() -> None:
    projected = project_to_duration(
        energy_kwh=0.5,
        co2eq_g=10.0,
        measured_duration_s=12 * 3600,
        prediction_hours=24,
    )

    assert projected["energy_kwh"] == pytest.approx(1.0)
    assert projected["co2eq_g"] == pytest.approx(20.0)
