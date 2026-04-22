import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "temp_anomaly.py"

FIXTURE = """\
Date,Temperature,Extra
2026-01-01,72.0,x
2026-01-02
2026-02-30,70.0,x
2026-01-04,seventy,x
2026-01-05,nan,x
2026-01-06,inf,x
2026-01-07,-inf,x
2026-01-08,73.0,x
"""

EXPECTED = (
    "TEMPERATURE ANOMALY REPORT\n"
    "ASCII CHART\n"
    # 2026-01-01,72.0 -> pos=0; 2026-01-08,73.0 -> pos=69
    "2026-01-01 |*---------------------------------------------------------------------| 72.0F\n"
    "2026-01-08 |---------------------------------------------------------------------*| 73.0F\n"
    "|----------------------------------------------------------------------|\n"
    "72.0F                                                              73.0F\n"
    "ANOMALIES\n"
    "(none)\n"
    "DATA ISSUES\n"
    "Line 3: malformed row\n"
    "Line 4: invalid date: 2026-02-30\n"
    "Line 5: non-numeric temperature: seventy\n"
    "Line 6: non-numeric temperature: nan\n"
    "Line 7: non-numeric temperature: inf\n"
    "Line 8: non-numeric temperature: -inf\n"
)


def test_data_issues_section(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text(FIXTURE)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(f)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == EXPECTED
    assert result.stderr == ""
