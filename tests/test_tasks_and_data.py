import json
from pathlib import Path

from donut_camera.data import Sample, format_target, load_samples, parse_output
from donut_camera.tasks import TaskSpec, load_task


def test_load_task(tmp_path: Path) -> None:
    path = tmp_path / "task.toml"
    path.write_text('name = "test"\nfields = ["a", "b"]\n')
    task = load_task(path)
    assert task.fields == ("a", "b")
    assert task.structural_tokens[-1] == "</s_b>"


def test_target_and_parser_round_trip() -> None:
    task = TaskSpec(name="test", fields=("name", "date"))
    sample = Sample(
        document_id="1",
        image=Path("unused.png"),
        fields=({"field_name": "PREFIX/name", "annotator_text": "Alice"},),
    )
    target = format_target(sample, task)
    assert target == ("<s_name>Alice</s_name><s_date><missing></s_date>")
    parsed = parse_output(f"<s_donut>{target}</s>", task)
    assert parsed.valid
    assert parsed.fields == {"name": "Alice", "date": ""}


def test_parser_detects_missing_tags() -> None:
    task = TaskSpec(name="test", fields=("name", "date"))
    parsed = parse_output("<s_name>Alice</s_name>", task)
    assert not parsed.valid


def test_load_samples_resolves_relative_images(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps([{"id": "doc", "image": "images/a.png", "fields": []}]))
    [sample] = load_samples(path)
    assert sample.document_id == "doc"
    assert sample.image == tmp_path / "images/a.png"
