"""Exercise recursive discovery and structured source normalization."""

import json

from docx import Document

from support_copilot.ingestion.document_loader import discover_files, prepare_documents
from support_copilot.ingestion.loaders.docx_loader import load_docx
from support_copilot.ingestion.loaders.json_loader import flatten_json, load_json
from support_copilot.ingestion.loaders.text_loader import load_text


def test_discovery_is_recursive_and_sorted(make_config) -> None:
    config = make_config()
    (config.raw_data_dir / "z").mkdir(parents=True)
    (config.raw_data_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (config.raw_data_dir / "z" / "b.md").write_text("beta", encoding="utf-8")
    assert [path.relative_to(config.raw_data_dir).as_posix() for path in discover_files(config.raw_data_dir)] == [
        "a.txt",
        "z/b.md",
    ]


def test_duplicate_content_is_reported_without_duplicate_records(make_config) -> None:
    config = make_config()
    config.raw_data_dir.mkdir(parents=True)
    (config.raw_data_dir / "one.txt").write_text("same evidence", encoding="utf-8")
    (config.raw_data_dir / "two.txt").write_text("same evidence", encoding="utf-8")
    result = prepare_documents(config)
    assert len(result.records) == 1
    assert sorted(item.status for item in result.files) == ["duplicate", "processed"]


def test_text_loader_preserves_line_location(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Heading\nEvidence\n\nSecond block", encoding="utf-8")
    result = load_text(path, "notes.txt")
    assert [record["metadata"]["location"] for record in result.records] == ["lines:1-2", "lines:4-4"]


def test_json_loader_flattens_nested_values_and_paths(tmp_path) -> None:
    path = tmp_path / "tests.json"
    path.write_text(json.dumps([{"engine": {"temperature": 120}}, None]), encoding="utf-8")
    result = load_json(path, "tests.json")
    assert result.records[0]["metadata"]["location"] == "$[0]"
    assert "engine.temperature: 120" in result.records[0]["text"]
    assert flatten_json(None) == ["value: null"]


def test_docx_loader_preserves_paragraph_and_table_locations(tmp_path) -> None:
    path = tmp_path / "manual.docx"
    document = Document()
    document.add_paragraph("Inspection notes")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Part"
    table.cell(0, 1).text = "Limit"
    table.cell(1, 0).text = "Brake"
    table.cell(1, 1).text = "2 mm"
    document.save(path)
    records = load_docx(path, "manual.docx").records
    assert any(record["metadata"]["location"] == "paragraph:1" for record in records)
    assert any(record["metadata"]["location"] == "table:1/row:2" for record in records)
