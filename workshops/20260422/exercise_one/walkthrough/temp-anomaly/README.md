# temp-anomaly

Reads a CSV file of daily temperature readings, detects statistical anomalies using a 30-day rolling window, and prints a fixed-width ASCII chart with an anomaly report.

## Requirements

- Python 3.12+

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python temp_anomaly.py <input.csv>
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Usage error, file error, or ordering error |
| 2 | Schema error |

## CSV Schema

The input file must be UTF-8 (optional BOM) with at least these two columns (case-insensitive, extra columns allowed):

| Column | Format | Notes |
|--------|--------|-------|
| `Date` | `YYYY-MM-DD` | Must be strictly increasing |
| `Temperature` | Decimal number | Must be finite (no NaN/Inf) |

## Example Input

```csv
Date,Temperature
2026-01-01,50.0
2026-01-02,51.0
2026-01-03,49.5
```

## Example Output

```
TEMPERATURE ANOMALY REPORT
ASCII CHART
2026-01-01 |*---------------------------------------------------------------------| 50.0F
2026-01-02 |---------------------------------------------------------------------*| 51.0F
2026-01-03 |----------------------------------*-----------------------------------| 49.5F
|----------------------------------------------------------------------|
49.5F                                                              51.0F
ANOMALIES
(none)
DATA ISSUES
```

## Running Tests

```bash
# Run all tests
venv/bin/python -m pytest -q

# Run with coverage
venv/bin/coverage run -m pytest
venv/bin/coverage report -m
```
