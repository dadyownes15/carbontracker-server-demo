from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from server.config import FP32_MODEL_ID, INT8_MODEL_ID


@dataclass(frozen=True)
class GateResult:
    variant: str
    model_id: str
    label: str
    score: float
    input_tokens: int
    duration_ms: float

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


class GateModel:
    def __init__(self, *, variant: str, model_id: str) -> None:
        self.variant = variant
        self.model_id = model_id
        self._torch = _load_torch()
        self._numpy = _load_numpy()
        self.tokenizer, self.model = self._load_model()
        if hasattr(self.model, "eval"):
            self.model.eval()

    def predict(self, *, seller_text: str, customer_text: str) -> GateResult:
        started_at = time.perf_counter()
        inputs = self.tokenizer(
            seller_text,
            customer_text,
            return_tensors="np" if self.variant == "int8_static" else "pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        if self.variant == "int8_static":
            logits = self._predict_onnx(inputs)
            probs = _softmax(self._numpy, logits)
            top_index = int(probs.argmax())
            score = float(probs[top_index])
            id2label = self.model["id2label"]
        else:
            with self._torch.no_grad():
                output = self.model(**inputs)
            logits = output.logits[0]
            probs = self._torch.softmax(logits, dim=-1)
            top_index = int(self._torch.argmax(probs).item())
            score = float(probs[top_index].item())
            id2label = self.model.config.id2label
        label = id2label.get(top_index, f"LABEL_{top_index}")
        input_tokens = int(inputs["attention_mask"].sum().item())

        return GateResult(
            variant=self.variant,
            model_id=self.model_id,
            label=label,
            score=score,
            input_tokens=input_tokens,
            duration_ms=(time.perf_counter() - started_at) * 1000.0,
        )

    def warmup(self) -> GateResult:
        return self.predict(
            seller_text="I can make this quick.",
            customer_text="I am not sure I have time right now.",
        )

    def _load_model(self) -> tuple[Any, Any]:
        transformers = _load_transformers()
        tokenizer = _load_tokenizer(transformers, self.model_id)

        if self.variant == "fp32":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.model_id
            )
            return tokenizer, model

        if self.variant == "int8_static":
            return tokenizer, _load_static_int8_onnx_model(transformers, self.model_id)

        raise ValueError(f"Unsupported gate variant: {self.variant}")

    def _predict_onnx(self, inputs: dict[str, Any]) -> Any:
        session = self.model["session"]
        input_names = self.model["input_names"]
        feeds = {
            name: inputs[name]
            for name in input_names
            if name in inputs
        }
        return session.run(None, feeds)[0][0]


@lru_cache(maxsize=2)
def get_gate_model(variant: str) -> GateModel:
    if variant == "fp32":
        return GateModel(variant=variant, model_id=FP32_MODEL_ID)
    if variant == "int8_static":
        return GateModel(variant=variant, model_id=INT8_MODEL_ID)
    raise ValueError(f"Unsupported gate variant: {variant}")


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch before running the gate demo.") from exc

    if hasattr(torch.backends, "quantized") and "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    return torch


def _load_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("Install numpy before running the gate demo.") from exc
    return numpy


def _load_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError("Install transformers before running the gate demo.") from exc
    return transformers


def _load_tokenizer(transformers: Any, model_id: str) -> Any:
    try:
        return transformers.AutoTokenizer.from_pretrained(model_id)
    except Exception:
        # Some quantized INC repos omit tokenizer files; the FP32 parent uses the
        # same BERT vocabulary and keeps the demo fetch path explicit.
        return transformers.AutoTokenizer.from_pretrained(FP32_MODEL_ID)


def _load_static_int8_onnx_model(transformers: Any, model_id: str) -> dict[str, Any]:
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Install onnxruntime and huggingface-hub before running the INT8 gate."
        ) from exc

    config = transformers.AutoConfig.from_pretrained(model_id)
    model_path = hf_hub_download(repo_id=model_id, filename="model.onnx")
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    return {
        "session": session,
        "input_names": [item.name for item in session.get_inputs()],
        "id2label": config.id2label,
    }


def _softmax(numpy: Any, logits: Any) -> Any:
    shifted = logits - numpy.max(logits)
    exp = numpy.exp(shifted)
    return exp / exp.sum()
