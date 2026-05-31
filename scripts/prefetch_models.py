from __future__ import annotations

from server.gate_model import get_gate_model


def main() -> None:
    for variant in ("fp32", "int8_static"):
        model = get_gate_model(variant)
        result = model.warmup()
        print(
            f"{variant}: loaded {result.model_id} "
            f"label={result.label} score={result.score:.4f} "
            f"duration_ms={result.duration_ms:.2f}"
        )


if __name__ == "__main__":
    main()
