from __future__ import annotations

import json
from pathlib import Path


def test_fixture_is_balanced_to_50_percent_gate_pass() -> None:
    rows = [
        json.loads(line)
        for line in Path("data/interactions.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert rows
    assert sum(1 for row in rows if row["demo_gate_pass"]) * 2 == len(rows)
    assert all(row["seller_text"] and row["customer_text"] for row in rows)
