from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from server.config import CARBON_INTENSITY_G_PER_KWH, LLM_KWH_PER_TOKEN


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class LlmUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    external_energy_kwh: float
    external_emissions_g: float
    external_carbon_intensity_g_per_kwh: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def zero_usage() -> LlmUsage:
    return _usage_from_counts(prompt_tokens=0, completion_tokens=0)


def generate_suggestion(
    *,
    seller_text: str,
    customer_text: str,
    gate_label: str | None,
) -> tuple[str, LlmUsage]:
    prompt = _build_prompt(
        seller_text=seller_text,
        customer_text=customer_text,
        gate_label=gate_label,
    )
    suggestion = _deterministic_suggestion(customer_text)
    usage = _usage_from_counts(
        prompt_tokens=count_tokens(prompt),
        completion_tokens=count_tokens(suggestion),
    )
    return suggestion, usage


def count_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _build_prompt(
    *,
    seller_text: str,
    customer_text: str,
    gate_label: str | None,
) -> str:
    return "\n".join(
        [
            "You are Callbuddy. Suggest one short answer for a phone seller.",
            f"Gate label: {gate_label or 'not_used'}",
            f"Seller: {seller_text}",
            f"Customer: {customer_text}",
        ]
    )


def _deterministic_suggestion(customer_text: str) -> str:
    lowered = customer_text.lower()
    if "price" in lowered or "expensive" in lowered or "cost" in lowered:
        return (
            "Acknowledge the price concern, then ask what they pay today so you "
            "can compare the concrete difference."
        )
    if "time" in lowered or "busy" in lowered:
        return (
            "Respect their time and ask for permission to give the short version "
            "before continuing."
        )
    if "contract" in lowered or "binding" in lowered:
        return (
            "Answer the contract question directly, then bring the conversation "
            "back to the next small decision."
        )
    return (
        "Mirror the customer's point in one sentence, then ask a calm follow-up "
        "question that keeps the conversation moving."
    )


def _usage_from_counts(*, prompt_tokens: int, completion_tokens: int) -> LlmUsage:
    total_tokens = prompt_tokens + completion_tokens
    energy_kwh = total_tokens * LLM_KWH_PER_TOKEN
    emissions_g = energy_kwh * CARBON_INTENSITY_G_PER_KWH
    return LlmUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        external_energy_kwh=energy_kwh,
        external_emissions_g=emissions_g,
        external_carbon_intensity_g_per_kwh=CARBON_INTENSITY_G_PER_KWH,
    )
