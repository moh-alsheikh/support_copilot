from support_copilot.ingestion.ocr import AVAILABLE, BROKEN, MISSING, _program_status


def test_program_that_runs_is_available() -> None:
    assert _program_status(("true",)) == AVAILABLE


def test_program_on_path_that_fails_to_run_is_broken_not_available() -> None:
    assert _program_status(("false",)) == BROKEN


def test_program_not_on_path_is_missing() -> None:
    assert _program_status(("definitely-not-an-installed-ocr-tool",)) == MISSING


def test_first_runnable_candidate_wins() -> None:
    assert _program_status(("false",), ("true",)) == AVAILABLE
