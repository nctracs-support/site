import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "temp_anomaly.py"


def run(content: str, tmp_path):
    f = tmp_path / "data.csv"
    f.write_text(content)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(f)],
        capture_output=True,
        text=True,
    )


def test_duplicate_date(tmp_path):
    result = run(
        "Date,Temperature\n"
        "2026-01-01,70\n"
        "2026-01-01,71\n",
        tmp_path,
    )
    assert result.returncode == 1
    assert result.stdout == "ERROR: Duplicate date encountered at line 3: 2026-01-01\n"
    assert result.stderr == ""


def test_out_of_order_date(tmp_path):
    result = run(
        "Date,Temperature\n"
        "2026-01-02,70\n"
        "2026-01-01,71\n",
        tmp_path,
    )
    assert result.returncode == 1
    assert result.stdout == "ERROR: Date out of order at line 3: 2026-01-01 after 2026-01-02\n"
    assert result.stderr == ""


def test_invalid_rows_skipped_in_ordering(tmp_path):
    result = run(
        "Date,Temperature\n"
        "2026-01-02,70\n"
        "bad-date,71\n"
        "2026-01-03,72\n",
        tmp_path,
    )
    assert result.returncode == 0
    assert result.stderr == ""
