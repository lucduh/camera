from collections.abc import Callable
from dataclasses import dataclass

from donut_camera.tasks import TaskSpec


@dataclass(frozen=True)
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def classify_value(
    prediction: str,
    reference: str,
    normalizer: Callable[[str], str] | None = None,
) -> Counts:
    if normalizer is not None:
        prediction = normalizer(prediction)
        reference = normalizer(reference)
    has_prediction = bool(prediction)
    has_reference = bool(reference)
    if has_prediction and has_reference and prediction == reference:
        return Counts(tp=1)
    if has_prediction and has_reference:
        # A wrong value is simultaneously a spurious prediction and a missed truth.
        return Counts(fp=1, fn=1)
    if has_prediction:
        return Counts(fp=1)
    if has_reference:
        return Counts(fn=1)
    return Counts()


def extraction_metrics(
    records: list[dict],
    task: TaskSpec,
    *,
    normalized: bool = False,
) -> dict:
    normalizer = normalize if normalized else None
    by_field = {field: Counts() for field in task.fields}
    exact_documents = 0
    valid_documents = 0

    for record in records:
        prediction = record["prediction"]
        reference = record["reference"]
        document_exact = True
        for field in task.fields:
            counts = classify_value(
                prediction.get(field, ""), reference.get(field, ""), normalizer
            )
            by_field[field] = by_field[field] + counts
            if counts.fp or counts.fn:
                document_exact = False
        exact_documents += int(document_exact)
        valid_documents += int(record.get("valid", False))

    micro = Counts()
    for counts in by_field.values():
        micro += counts

    supported = [counts.f1 for counts in by_field.values() if counts.tp + counts.fn > 0]
    macro_f1 = sum(supported) / len(supported) if supported else 0.0
    n_documents = len(records)
    return {
        "mode": "normalized" if normalized else "strict",
        "documents": n_documents,
        "micro": micro.as_dict(),
        "macro_field_f1": macro_f1,
        "document_exact_match": exact_documents / n_documents if n_documents else 0.0,
        "structured_validity": valid_documents / n_documents if n_documents else 0.0,
        "fields": {field: counts.as_dict() for field, counts in by_field.items()},
    }
