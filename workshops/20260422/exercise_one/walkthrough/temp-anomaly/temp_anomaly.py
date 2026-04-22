import csv
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import date


@dataclass
class Anomaly:
    date: date
    temp: float
    mean: float
    diff: float
    z: float


def check_anomaly(temp: float, window: list[float]) -> tuple[float, float, float] | None:
    """Return (mean, diff, z) if temp is anomalous vs window, else None."""
    if len(window) < 10:
        return None
    m = sum(window) / len(window)
    sd = statistics.stdev(window)
    if sd == 0.0:
        return None
    diff = temp - m
    if abs(diff) > 2 * sd:
        return (m, diff, diff / sd)
    return None


def update_window(window: list[float], temp: float) -> None:
    window.append(temp)
    if len(window) > 30:
        window.pop(0)


def compute_pos(temp: float, min_temp: float, max_temp: float) -> int:
    if min_temp == max_temp:
        return 35
    pos = round(((temp - min_temp) / (max_temp - min_temp)) * 69)
    return max(0, min(69, pos))


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv
    if len(argv) != 2:
        print("Usage: python temp_anomaly.py <input.csv>")
        sys.exit(1)

    path = argv[1]
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
    except OSError:
        print(f"ERROR: Cannot open file '{path}'")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"ERROR: Cannot decode file '{path}' as UTF-8")
        sys.exit(1)

    # Locate header: first row with >= 2 columns
    header_line = None
    header = None
    for line_num, row in enumerate(rows, start=1):
        if len(row) >= 2:
            header_line = line_num
            header = row
            break

    if header is None:
        print("ERROR: Missing header row")
        sys.exit(2)

    cols = [c.strip().lower() for c in header]
    for required, label in (("date", "Date"), ("temperature", "Temperature")):
        count = cols.count(required)
        if count > 1:
            print(f"ERROR: Duplicate column '{label}'")
            sys.exit(2)
        if count == 0:
            print(f"ERROR: Missing required column '{label}'")
            sys.exit(2)

    date_idx = cols.index("date")
    temp_idx = cols.index("temperature")
    ncols = len(header)

    data_issues: list[tuple[int, str]] = []
    valid_rows: list[tuple[date, float]] = []
    last_date: date | None = None

    for line_num, row in enumerate(rows, start=1):
        if line_num <= header_line:
            continue

        if len(row) < ncols:
            data_issues.append((line_num, "malformed row"))
            continue

        date_str = row[date_idx].strip()
        try:
            parsed_date = date.fromisoformat(date_str)
        except ValueError:
            data_issues.append((line_num, f"invalid date: {date_str}"))
            continue

        temp_str = row[temp_idx].strip()
        try:
            temp_val = float(temp_str)
            if not math.isfinite(temp_val):
                raise ValueError
        except ValueError:
            data_issues.append((line_num, f"non-numeric temperature: {temp_str}"))
            continue

        if last_date is not None:
            if parsed_date == last_date:
                print(f"ERROR: Duplicate date encountered at line {line_num}: {parsed_date}")
                sys.exit(1)
            if parsed_date < last_date:
                print(f"ERROR: Date out of order at line {line_num}: {parsed_date} after {last_date}")
                sys.exit(1)

        last_date = parsed_date
        valid_rows.append((parsed_date, temp_val))

    anomalies: list[Anomaly] = []
    window: list[float] = []
    for d, temp in valid_rows:
        result = check_anomaly(temp, window)
        if result is not None:
            m, diff, z = result
            anomalies.append(Anomaly(d, temp, m, diff, z))
        update_window(window, temp)

    print("TEMPERATURE ANOMALY REPORT")
    print("ASCII CHART")

    if valid_rows:
        min_temp = min(t for _, t in valid_rows)
        max_temp = max(t for _, t in valid_rows)
        anomaly_dates = {a.date for a in anomalies}
        for d, temp in valid_rows:
            pos = compute_pos(temp, min_temp, max_temp)
            marker = "#" if d in anomaly_dates else "*"
            bar = "-" * pos + marker + "-" * (69 - pos)
            print(f"{d} |{bar}| {temp:.1f}F")
        min_label = f"{min_temp:.1f}F"
        max_label = f"{max_temp:.1f}F"
        print("|" + "-" * 70 + "|")
        print(min_label + " " * (72 - len(min_label) - len(max_label)) + max_label)

    print("ANOMALIES")
    if not anomalies:
        print("(none)")
    else:
        print(f"{'Date':<10}  {'Temp(F)':>7}  {'Mean(F)':>7}  {'Diff(F)':>7}  {'Z-Score':>7}")
        for a in anomalies:
            print(f"{a.date!s:<10}  {a.temp:>7.1f}  {a.mean:>7.1f}  {a.diff:>+7.1f}  {a.z:>+7.1f}")
    print("DATA ISSUES")
    for line_num, issue in data_issues:
        print(f"Line {line_num}: {issue}")


if __name__ == "__main__":
    main()
