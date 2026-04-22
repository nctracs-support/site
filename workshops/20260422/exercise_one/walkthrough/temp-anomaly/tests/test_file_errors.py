import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "temp_anomaly.py"


def test_invalid_utf8_prints_error_and_exits_1(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_bytes(b"\xff\xfe\xff")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(bad_csv)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stdout == f"ERROR: Cannot decode file '{bad_csv}' as UTF-8\n"
    assert result.stderr == ""


def test_missing_file_prints_error_and_exits_1():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "does_not_exist.csv"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stdout == "ERROR: Cannot open file 'does_not_exist.csv'\n"
    assert result.stderr == ""
