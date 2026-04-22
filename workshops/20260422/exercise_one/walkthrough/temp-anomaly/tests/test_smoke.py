import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "temp_anomaly.py"


def test_no_args_prints_usage_and_exits_1():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stdout == "Usage: python temp_anomaly.py <input.csv>\n"
    assert result.stderr == ""
