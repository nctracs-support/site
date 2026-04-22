"""Full golden test: exact stdout match for a fixture covering all features."""
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "temp_anomaly.py"

FIXTURE = """\
Date,Temperature,Notes
2026-01-01,50.0,ok
2026-01-02,51.0,ok
2026-01-03,52.0,ok
2026-01-04,53.0,ok
2026-01-05,54.0,ok
2026-01-06,55.0,ok
2026-01-07,56.0,ok
2026-01-08,57.0,ok
2026-01-09,58.0,ok
2026-01-10,59.0,ok
2026-01-11
2026-02-30,70.0,invalid-date
2026-01-12,abc,bad-temp
2026-01-13,nan,nan-temp
2026-01-14,inf,inf-temp
2026-01-15,100.0,anomaly
2026-01-16,51.0,ok
"""

# Known eligible rows (derived independently from fixture)
_ELIGIBLE: list[tuple[date, float]] = [
    (date(2026, 1, 1),  50.0),
    (date(2026, 1, 2),  51.0),
    (date(2026, 1, 3),  52.0),
    (date(2026, 1, 4),  53.0),
    (date(2026, 1, 5),  54.0),
    (date(2026, 1, 6),  55.0),
    (date(2026, 1, 7),  56.0),
    (date(2026, 1, 8),  57.0),
    (date(2026, 1, 9),  58.0),
    (date(2026, 1, 10), 59.0),
    (date(2026, 1, 15), 100.0),  # anomaly: priors=[50..59], mean=54.5
    (date(2026, 1, 16), 51.0),
]

_DATA_ISSUES = [
    "Line 12: malformed row",
    "Line 13: invalid date: 2026-02-30",
    "Line 14: non-numeric temperature: abc",
    "Line 15: non-numeric temperature: nan",
    "Line 16: non-numeric temperature: inf",
]


def _compute_pos(temp: float, min_t: float, max_t: float) -> int:
    if min_t == max_t:
        return 35
    return max(0, min(69, round(((temp - min_t) / (max_t - min_t)) * 69)))


def _build_expected() -> str:
    """Independently compute the expected output from known inputs."""
    # --- anomaly detection (mirrors implementation logic) ---
    window: list[float] = []
    anomaly_dates: set[date] = set()
    anomaly_rows: list[tuple[date, float, float, float, float]] = []

    for d, temp in _ELIGIBLE:
        if len(window) >= 10:
            m = sum(window) / len(window)
            sd = statistics.stdev(window)
            if sd != 0.0:
                diff = temp - m
                if abs(diff) > 2 * sd:
                    anomaly_dates.add(d)
                    anomaly_rows.append((d, temp, m, diff, diff / sd))
        window.append(temp)
        if len(window) > 30:
            window.pop(0)

    # --- chart ---
    min_temp = min(t for _, t in _ELIGIBLE)
    max_temp = max(t for _, t in _ELIGIBLE)
    min_label = f"{min_temp:.1f}F"
    max_label = f"{max_temp:.1f}F"

    out: list[str] = ["TEMPERATURE ANOMALY REPORT", "ASCII CHART"]
    for d, temp in _ELIGIBLE:
        pos = _compute_pos(temp, min_temp, max_temp)
        marker = "#" if d in anomaly_dates else "*"
        bar = "-" * pos + marker + "-" * (69 - pos)
        out.append(f"{d} |{bar}| {temp:.1f}F")
    out.append("|" + "-" * 70 + "|")
    out.append(min_label + " " * (72 - len(min_label) - len(max_label)) + max_label)

    # --- anomalies ---
    out.append("ANOMALIES")
    if not anomaly_rows:
        out.append("(none)")
    else:
        out.append(
            f"{'Date':<10}  {'Temp(F)':>7}  {'Mean(F)':>7}  {'Diff(F)':>7}  {'Z-Score':>7}"
        )
        for d, temp, m, diff, z in anomaly_rows:
            out.append(
                f"{d!s:<10}  {temp:>7.1f}  {m:>7.1f}  {diff:>+7.1f}  {z:>+7.1f}"
            )

    # --- data issues ---
    out.append("DATA ISSUES")
    out.extend(_DATA_ISSUES)

    return "\n".join(out) + "\n"


def test_golden_output(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text(FIXTURE)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(f)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == _build_expected()
