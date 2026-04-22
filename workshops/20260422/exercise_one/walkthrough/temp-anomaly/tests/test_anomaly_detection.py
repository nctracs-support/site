import statistics
import subprocess
import sys
from pathlib import Path

from temp_anomaly import check_anomaly, update_window

SCRIPT = Path(__file__).parent.parent / "temp_anomaly.py"


def test_fewer_than_10_priors_returns_none():
    assert check_anomaly(999.0, [1.0] * 9) is None


def test_exactly_10_priors_enables_detection():
    window = [1.0] * 10  # stddev=0 -> skip, but first test with a spread window
    window = list(range(1, 11))
    assert check_anomaly(999.0, window) is not None


def test_window_not_modified_by_check_anomaly():
    # Verifies exclusion of current row: function is pure, doesn't append temp
    window = list(range(10))
    original = window.copy()
    check_anomaly(999.0, window)
    assert window == original


def test_window_max_size_30():
    window: list[float] = []
    for i in range(35):
        update_window(window, float(i))
    assert len(window) == 30
    assert window[0] == 5.0  # first 5 were evicted


def test_stddev_uses_sample_ddof1():
    window = list(range(1, 11))  # [1..10]
    result = check_anomaly(100.0, window)
    assert result is not None
    m, diff, z = result
    expected_sd = statistics.stdev(window)
    assert abs(z - diff / expected_sd) < 1e-10


def test_strict_inequality_just_below_boundary_is_not_anomaly():
    window = list(range(1, 11))
    m = statistics.mean(window)
    sd = statistics.stdev(window)
    # Slightly below 2*sd: strict > means this is NOT an anomaly
    assert check_anomaly(m + 2 * sd - 0.001, window) is None


def test_strict_inequality_just_above_boundary_is_anomaly():
    window = list(range(1, 11))
    m = statistics.mean(window)
    sd = statistics.stdev(window)
    assert check_anomaly(m + 2 * sd + 0.001, window) is not None


def test_stddev_zero_skips_detection():
    assert check_anomaly(999.0, [5.0] * 10) is None


def test_anomalies_table_format(tmp_path):
    # 10 priors 50-59, then extreme 100.0 -> exactly one anomaly
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

    # Compute expected values
    window = list(range(50, 60))
    m = statistics.mean(window)          # 54.5
    sd = statistics.stdev(window)
    diff = 100.0 - m                     # +45.5
    z = diff / sd                        # ~+15.0

    expected_header = (
        f"{'Date':<10}  {'Temp(F)':>7}  {'Mean(F)':>7}  {'Diff(F)':>7}  {'Z-Score':>7}"
    )
    expected_row = (
        f"{'2026-01-11':<10}  {100.0:>7.1f}  {m:>7.1f}  {diff:>+7.1f}  {z:>+7.1f}"
    )

    assert "ANOMALIES\n" in result.stdout
    assert expected_header in result.stdout
    assert expected_row in result.stdout


def test_anomalies_none_when_no_anomalies(tmp_path):
    # Only 3 rows -> no window for detection -> "(none)"
    f = tmp_path / "data.csv"
    f.write_text("Date,Temperature\n2026-01-01,50.0\n2026-01-02,51.0\n2026-01-03,52.0\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(f)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ANOMALIES\n(none)\n" in result.stdout
