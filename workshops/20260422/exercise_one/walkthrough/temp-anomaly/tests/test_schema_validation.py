import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "temp_anomaly.py"

MISSING_HEADER = "ERROR: Missing header row\n"


def run(path: Path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        capture_output=True,
        text=True,
    )


def test_empty_file(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_text("")
    result = run(f)
    assert result.returncode == 2
    assert result.stdout == MISSING_HEADER
    assert result.stderr == ""


def test_whitespace_only(tmp_path):
    f = tmp_path / "blank.csv"
    f.write_text("   \n\n  \n")
    result = run(f)
    assert result.returncode == 2
    assert result.stdout == MISSING_HEADER
    assert result.stderr == ""


def test_single_column(tmp_path):
    f = tmp_path / "onecol.csv"
    f.write_text("onlyone\n")
    result = run(f)
    assert result.returncode == 2
    assert result.stdout == MISSING_HEADER
    assert result.stderr == ""


# --- valid header variants ---

def test_valid_header_exact(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("Date,Temperature\n")
    result = run(f)
    assert result.returncode == 0
    assert result.stdout.startswith("TEMPERATURE ANOMALY REPORT\n")
    assert result.stderr == ""


def test_valid_header_trimmed(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text(" DATE , TEMPERATURE \n")
    result = run(f)
    assert result.returncode == 0
    assert result.stdout.startswith("TEMPERATURE ANOMALY REPORT\n")
    assert result.stderr == ""


def test_valid_header_lowercase(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("date,temperature\n")
    result = run(f)
    assert result.returncode == 0
    assert result.stdout.startswith("TEMPERATURE ANOMALY REPORT\n")
    assert result.stderr == ""


# --- missing required columns ---

def test_missing_temperature_column(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("Date,Temp\n")
    result = run(f)
    assert result.returncode == 2
    assert result.stdout == "ERROR: Missing required column 'Temperature'\n"
    assert result.stderr == ""


def test_missing_date_column(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("Temperature,Value\n")
    result = run(f)
    assert result.returncode == 2
    assert result.stdout == "ERROR: Missing required column 'Date'\n"
    assert result.stderr == ""


# --- duplicate columns ---

def test_duplicate_date_column(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("Date,DATE,Temperature\n")
    result = run(f)
    assert result.returncode == 2
    assert result.stdout == "ERROR: Duplicate column 'Date'\n"
    assert result.stderr == ""


def test_duplicate_temperature_column(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("Date,Temperature,TEMPERATURE\n")
    result = run(f)
    assert result.returncode == 2
    assert result.stdout == "ERROR: Duplicate column 'Temperature'\n"
    assert result.stderr == ""
