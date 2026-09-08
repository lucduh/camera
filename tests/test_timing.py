from donut_camera.timing import summarize_ms


def test_summarize_ms() -> None:
    summary = summarize_ms([1.0, 2.0, 3.0, 4.0, 100.0])
    assert summary["median_ms"] == 3.0
    assert summary["mean_ms"] == 22.0
    assert summary["p95_ms"] == 100.0


def test_summarize_empty_values() -> None:
    assert summarize_ms([]) == {}
