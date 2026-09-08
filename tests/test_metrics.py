import pytest

from donut_camera.metrics import classify_value, extraction_metrics
from donut_camera.tasks import TaskSpec


def test_wrong_value_is_false_positive_and_false_negative() -> None:
    counts = classify_value("wrong", "right")
    assert (counts.tp, counts.fp, counts.fn) == (0, 1, 1)
    assert counts.f1 == 0.0


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("value", "value", (1, 0, 0)),
        ("value", "", (0, 1, 0)),
        ("", "value", (0, 0, 1)),
        ("", "", (0, 0, 0)),
    ],
)
def test_field_classification(prediction, reference, expected) -> None:
    counts = classify_value(prediction, reference)
    assert (counts.tp, counts.fp, counts.fn) == expected


def test_micro_macro_and_document_metrics() -> None:
    task = TaskSpec(name="test", fields=("a", "b"))
    records = [
        {
            "prediction": {"a": "yes", "b": "wrong"},
            "reference": {"a": "yes", "b": "right"},
            "valid": True,
        },
        {
            "prediction": {"a": "YES", "b": ""},
            "reference": {"a": "yes", "b": ""},
            "valid": False,
        },
    ]
    strict = extraction_metrics(records, task)
    assert strict["micro"]["tp"] == 1
    assert strict["micro"]["fp"] == 2
    assert strict["micro"]["fn"] == 2
    assert strict["structured_validity"] == 0.5
    assert strict["document_exact_match"] == 0.0

    normalized = extraction_metrics(records, task, normalized=True)
    assert normalized["micro"]["tp"] == 2
    assert normalized["document_exact_match"] == 0.5
