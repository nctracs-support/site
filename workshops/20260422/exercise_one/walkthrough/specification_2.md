# Developer-ready specification

## `temp_anomaly.py` — CSV Temperature Anomaly Detector

---

# 1. Goal

Build a **Python command-line tool** that:

1. Reads a CSV file containing daily temperature readings.
2. Detects anomalies where a temperature differs from a rolling mean by **more than 2 standard deviations**.
3. Uses a **rolling window of the previous 30 valid temperature rows**.
4. Outputs to **stdout only**:

   1. An ASCII chart (vertical timeline) highlighting anomalies.
   2. A chronological table of flagged anomaly dates.
   3. A chronological report of data issues.

The tool must produce **deterministic output suitable for golden testing**.

---

# 2. Implementation constraints

## Language

Python **3.12+**

## Script name

The executable script **must be named**

```
temp_anomaly.py
```

## Invocation

```
python temp_anomaly.py <input.csv>
```

Rules:

* Exactly **one positional argument** required.
* Output must be printed **only to stdout**.
* No output must be printed to stderr.

Incorrect invocation:

```
Usage: python temp_anomaly.py <input.csv>
```

Exit code: **1**

---

# 3. Dependencies

Prefer **standard library only**.

Allowed libraries:

```
csv
datetime
math
statistics
argparse
sys
pathlib
```

Development dependencies (for testing/linting):

```
pytest
pytest-cov
black
ruff
mypy
pre-commit
```

These must be listed in `requirements.txt`.

---

# 4. CSV schema

## Required columns

Headers are **case-insensitive** and **trimmed**.

Required columns:

```
Date
Temperature
```

Example valid headers:

```
Date,Temperature
DATE,TEMPERATURE
 date , temperature
```

If multiple columns match either required column (case-insensitive):

Fatal schema error.

Exit code **2**

---

## Header requirements

The CSV **must contain a header row**.

A header row is defined as the **first parseable CSV row containing at least two columns**.

If the file:

* is empty
* contains only whitespace
* has fewer than two columns

then:

```
ERROR: Missing header row
```

Exit code **2**

---

# 5. Parsing rules

## Encoding

Input file must be decoded using:

```
UTF-8 with optional BOM (utf-8-sig)
```

Decode failure:

```
ERROR: Cannot decode file '<filename>' as UTF-8
```

Exit code **1**

---

## Whitespace

Leading/trailing whitespace must be trimmed from:

* headers
* cell values

---

## Date format

Expected format:

```
YYYY-MM-DD
```

Rules:

* Must parse as a valid calendar date.
* Impossible dates (e.g. `2025-02-30`) are invalid.

Invalid dates are **non-fatal data issues**.

Rows with invalid dates:

* must be recorded in DATA ISSUES
* must **not appear in chart**
* must **not participate in rolling statistics**
* must **not participate in ordering validation**

---

## Temperature format

Temperature must parse as a finite float.

Invalid if:

* non-numeric
* contains units (e.g. `72F`)
* NaN
* +inf
* -inf

Invalid temperature rows are **data issues**.

They must:

* be recorded
* not appear in chart
* not enter rolling window

---

## Malformed rows

Rows shorter than the header column count are **data issues**.

Example:

```
2026-01-02
```

---

# 6. Chart-eligible rows

A row is **chart-eligible** if:

```
Date parses successfully
AND
Temperature parses as finite float
```

Only chart-eligible rows are used for:

* ordering validation
* anomaly detection
* ASCII chart
* min/max scaling

---

# 7. Input ordering constraints

Ordering rules apply **only to chart-eligible rows**.

Dates must be:

```
strictly increasing
```

Duplicate dates among chart-eligible rows are **fatal**.

Error:

```
ERROR: Duplicate date encountered at line <line>: <date>
```

Exit code **1**

Out-of-order:

```
ERROR: Date out of order at line <line>: <date> after <previous_date>
```

Exit code **1**

Validation must occur **during streaming parse**.

Execution stops immediately when detected.

The program **must not sort input**.

---

# 8. Rolling window specification

Maintain a list of the **previous up to 30 valid temperatures**.

Window rules:

* Contains **chart-eligible rows only**
* Invalid rows never enter window
* Window grows gradually until size 30
* Current row **excluded** from stats

---

## Minimum data requirement

Anomaly detection requires:

```
>= 10 prior temperatures
```

If fewer exist:

* row is valid
* anomaly detection skipped

---

# 9. Statistics definition

Mean:

```
mean = average(window)
```

Standard deviation:

```
sample stddev (ddof=1)
```

If:

```
stddev == 0
```

then:

* anomaly detection skipped
* z-score considered N/A

---

# 10. Anomaly definition

Constant:

```
ANOMALY_STDDEV_THRESHOLD = 2
```

Condition:

```
abs(temp - mean) > threshold * stddev
```

Strict inequality.

---

# 11. Values reported

For each anomaly:

```
diff = temp - mean
z = diff / stddev
```

---

# 12. Output structure

Sections must appear in this exact order:

```
TEMPERATURE ANOMALY REPORT
ASCII CHART
ANOMALIES
DATA ISSUES
```

No extra sections.

---

# 13. ASCII chart

## Row structure

```
YYYY-MM-DD |<70-char plot>| XX.XF
```

Example:

```
2026-03-01 |-------------------*----------------------------| 72.3F
```

Rules:

Plot width:

```
70 characters
```

Boundaries:

```
| at both ends
```

Markers:

```
* = normal
# = anomaly
```

---

# 14. Scaling

Compute:

```
min_temp
max_temp
```

Across **all chart-eligible rows**.

Mapping formula:

```
pos = round((temp - min_temp) / (max_temp - min_temp) * 69)
```

Clamp:

```
pos = max(0, min(69, pos))
```

---

## Special case

If:

```
min_temp == max_temp
```

then:

```
pos = 35
```

---

# 15. Bottom axis

After chart rows print:

```
|----------------------------------------------------------------------|
```

Then labels line:

```
<min_temp formatted>                                             <max_temp formatted>
```

Formatting:

```
1 decimal place
F suffix
```

Example:

```
12.4F                                                           97.8F
```

Max label must **end directly under final boundary**.

---

# 16. Anomalies table

Columns:

```
Date
Temp(F)
Mean(F)
Diff(F)
Z-Score
```

Formatting:

```
Temp, Mean, Diff → 1 decimal
Z-score → 1 decimal
Diff must include sign (+ or -)
Z-score must include sign
```

Example:

```
2026-03-02  89.1  71.4  +17.7  +3.1
```

---

## No anomalies case

Output:

```
(none)
```

---

# 17. Data issues section

Each entry format:

```
Line <n>: <issue_type>: <detail>
```

Issues listed in **ascending line order**.

Line numbers include header as **line 1**.

Examples:

```
Line 5: invalid date: 2025-02-30
Line 7: non-numeric temperature: seventy
Line 12: malformed row
```

---

# 18. Fatal errors

## File open failure

```
ERROR: Cannot open file '<filename>'
```

Exit code **1**

---

## Invocation error

```
Usage: python temp_anomaly.py <input.csv>
```

Exit code **1**

---

## Schema errors

Examples:

```
ERROR: Missing required column 'Temperature'
ERROR: Duplicate column 'Date'
ERROR: Missing header row
```

Exit code **2**

---

## Ordering errors

```
ERROR: Duplicate date encountered at line X: YYYY-MM-DD
```

or

```
ERROR: Date out of order at line X: YYYY-MM-DD after YYYY-MM-DD
```

Exit code **1**

---

# 19. Project structure

```
temp-anomaly/
├── temp_anomaly.py
├── requirements.txt
├── AGENTS.md
├── README.md
└── tests/
    ├── test_schema_validation.py
    ├── test_anomaly_detection.py
    ├── test_missing_values.py
    └── test_chart_output.py
```

---

# 20. AGENTS.md requirements

AGENTS.md must define:

Formatting

```
black
```

Linting

```
ruff
```

Type checking

```
mypy
```

Testing

```
pytest
pytest-cov
coverage >= 90%
```

Pre-commit hooks required.

Rules:

* Public functions must have docstrings.
* Do not print stack traces for user errors.
* All error messages must match specification exactly.

---

# 21. README.md requirements

Must include:

* overview
* Python requirement (3.12+)
* venv named `venv`
* installation instructions
* usage example
* required CSV schema
* example input
* example output
* running tests

---

# 22. Testing plan

## Unit tests

Must verify:

Schema validation

* missing columns
* duplicate columns
* header trimming
* case-insensitivity

Ordering

* duplicate dates
* out-of-order detection

Invalid data

* invalid dates
* non-numeric temperatures
* NaN/inf
* malformed rows

Rolling window

* uses previous 30 rows
* requires 10 priors
* ddof=1

Detection logic

* high anomalies
* low anomalies
* strict `>` threshold

---

## Golden output tests

At least one test must:

* generate a CSV fixture
* include anomalies
* include invalid rows
* include invalid dates
* run script via `subprocess.run`
* assert:

```
exit code == 0
stdout == expected_output
```

Additional golden tests:

* no anomalies
* fewer than 10 valid rows
* constant temperature (stddev=0)

---

# 23. Performance expectations

The tool must handle:

```
up to 5 million rows
```

Memory use must remain bounded.

Implementation must **stream input** and avoid storing unnecessary rows.

---

# 24. Determinism requirement

All outputs must be deterministic:

* chart scaling
* anomaly detection
* ordering of data issues
* numeric rounding

This guarantees reproducible golden tests.
