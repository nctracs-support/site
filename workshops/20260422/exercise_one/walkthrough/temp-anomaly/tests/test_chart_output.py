import subprocess
import sys
from pathlib import Path

from temp_anomaly import compute_pos

SCRIPT = Path(__file__).parent.parent / "temp_anomaly.py"


def test_min_temp_maps_to_zero():
    assert compute_pos(10.0, 10.0, 20.0) == 0


def test_max_temp_maps_to_69():
    assert compute_pos(20.0, 10.0, 20.0) == 69


def test_midpoint_rounds_to_even():
    # round((5/10)*69) = round(34.5) = 34 (Python banker's rounding)
    assert compute_pos(15.0, 10.0, 20.0) == 34


def test_clamp_below_min():
    assert compute_pos(5.0, 10.0, 20.0) == 0


def test_clamp_above_max():
    assert compute_pos(25.0, 10.0, 20.0) == 69


def test_min_equals_max_returns_center():
    assert compute_pos(15.0, 15.0, 15.0) == 35


def test_anomaly_row_uses_hash_marker(tmp_path):
    # Priors: 2026-01-01..10 at temps 50-59 (mean=54.5, stdev≈3.03)
    # Day 11 at 100.0: diff=45.5 >> 2*3.03 → anomaly → '#' marker
    lines = ["Date,Temperature"]
    for i in range(10):
        lines.append(f"2026-01-{i+1:02d},{50+i}.0")
    lines.append("2026-01-11,100.0")
    f = tmp_path / "data.csv"
    f.write_text("\n".join(lines) + "\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(f)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    # Day 11 (anomaly) -> '#' at pos 69 (max); day 01 (normal) -> '*' at pos 0
    assert "2026-01-11 |---------------------------------------------------------------------#| 100.0F" in result.stdout
    assert "2026-01-01 |*" in result.stdout


def test_chart_three_rows(tmp_path):
    # pos: 10.0->0, 15.0->34 (banker's round), 20.0->69
    f = tmp_path / "data.csv"
    f.write_text(
        "Date,Temperature\n"
        "2026-01-01,10.0\n"
        "2026-01-02,15.0\n"
        "2026-01-03,20.0\n"
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(f)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stderr == ""

    min_label, max_label = "10.0F", "20.0F"
    labels_line = min_label + " " * (72 - len(min_label) - len(max_label)) + max_label

    expected = (
        "TEMPERATURE ANOMALY REPORT\n"
        "ASCII CHART\n"
        "2026-01-01 |*---------------------------------------------------------------------| 10.0F\n"
        "2026-01-02 |----------------------------------*-----------------------------------| 15.0F\n"
        "2026-01-03 |---------------------------------------------------------------------*| 20.0F\n"
        "|----------------------------------------------------------------------|\n"
        f"{labels_line}\n"
        "ANOMALIES\n"
        "(none)\n"
        "DATA ISSUES\n"
    )
    assert result.stdout == expected
