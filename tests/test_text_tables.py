from support_copilot.ingestion.loaders.text_loader import load_text


def _records(tmp_path, text: str, name: str = "doc.md") -> list[tuple[str, str]]:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return [(record["metadata"]["location"], record["text"]) for record in load_text(path, name).records]


ASCIIDOC_EXPLICIT_HEADER = """\
=== Check LED warning codes
Intro paragraph.

[cols="1,1,4",options="header"]
|===
| Long flashes | Short flashes | Status

| 0
| 3
| Generic failure to boot

| 0
| 7
| Kernel image not found
|===

After the table.
"""


def test_asciidoc_rows_carry_headers_and_heading_with_original_locations(tmp_path) -> None:
    assert _records(tmp_path, ASCIIDOC_EXPLICIT_HEADER) == [
        ("lines:1-2", "=== Check LED warning codes\nIntro paragraph."),
        (
            "lines:8-10",
            "Check LED warning codes — Long flashes: 0 | Short flashes: 3 | Status: Generic failure to boot",
        ),
        (
            "lines:12-14",
            "Check LED warning codes — Long flashes: 0 | Short flashes: 7 | Status: Kernel image not found",
        ),
        ("lines:17-17", "After the table."),
    ]


def test_asciidoc_one_cell_per_line_header_uses_declared_columns_and_title(tmp_path) -> None:
    text = """\
== LED behaviour

[[power-leds]]
.Power LEDs by model
[cols="1,1",options="header"]
|===
| Models
| LED count

| Raspberry Pi 5
| 1
|===
"""
    assert _records(tmp_path, text) == [
        ("lines:1-1", "== LED behaviour"),
        ("lines:10-11", "Power LEDs by model — Models: Raspberry Pi 5 | LED count: 1"),
    ]


def test_asciidoc_implicit_header_and_empty_cells_are_skipped(tmp_path) -> None:
    text = """\
=== Typical power requirements

|===
| Product | Recommended PSU | Typical draw

| Raspberry Pi 5
| 5.0A
|
|===
"""
    assert _records(tmp_path, text)[-1] == (
        "lines:6-8",
        "Typical power requirements — Product: Raspberry Pi 5 | Recommended PSU: 5.0A",
    )


def test_asciidoc_table_without_header_keeps_plain_blocks(tmp_path) -> None:
    text = """\
|===
| | | Pi 1B+ | Pi 2B
| Boot | Max | 0.26 | 0.40
|===
"""
    assert _records(tmp_path, text) == [("lines:1-4", text.strip())]


def test_malformed_asciidoc_table_keeps_plain_blocks(tmp_path) -> None:
    text = """\
[cols="1,1",options="header"]
|===
| Name | Value

| only one cell
|===
"""
    assert _records(tmp_path, text) == [
        ("lines:1-3", '[cols="1,1",options="header"]\n|===\n| Name | Value'),
        ("lines:5-6", "| only one cell\n|==="),
    ]


def test_markdown_pipe_table_rows_carry_headers_and_heading(tmp_path) -> None:
    text = """\
# Error codes

| Code | Meaning |
| --- | :---: |
| E1 | Overheated |
| E2 | Low voltage |

Contact support for other codes.
"""
    assert _records(tmp_path, text) == [
        ("lines:1-1", "# Error codes"),
        ("lines:5-5", "Error codes — Code: E1 | Meaning: Overheated"),
        ("lines:6-6", "Error codes — Code: E2 | Meaning: Low voltage"),
        ("lines:8-8", "Contact support for other codes."),
    ]


def test_code_block_dashes_are_not_mistaken_for_a_markdown_table(tmp_path) -> None:
    text = "[source,console]\n----\n$ vcgencmd get_config usb_max_current_enable\n----\n"
    assert _records(tmp_path, text) == [("lines:1-4", text.strip())]
