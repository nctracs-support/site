## Developer-ready specification: `temp_anomaly.py` CSV temperature anomaly detector

### 1. Goal

Build a command-line tool that:

* Reads a CSV file containing daily temperature readings.
* Detects anomalies where a day’s temperature differs from a rolling mean by **more than 2 standard deviations**, using a **rolling window of the previous 30 valid rows**.
* Outputs to **stdout**:

  1. An ASCII chart (vertical timeline) highlighting anomalies.
  2. A chronological table of flagged anomaly dates with deviation magnitude.
  3. A chronological report of data issues (invalid rows).

### 2. Implementation constraints and choices

* **Language**: Python **3.12+**
* **Invocation**: `python temp_anomal.py <input.csv>`

  * **Input filename** is a **single positional argument** read from `sys.argv`.
  * Output always to **stdout**.
* **Dependencies**:

  * Third-party dependencies allowed; provide `requirements.txt`.
  * Use a virtual environment named `venv` (documented in README).
* **Stats behavior**:

  * Rolling window uses **previous 30 valid temperature rows in the file** (row-based, not calendar-based).
  * Rolling stats computed from **prior rows only** (exclude current row).
  * Standard deviation uses **sample** standard deviation (**ddof = 1**).
  * Minimum requirement: **at least 10 valid prior temperatures** required to compute mean/stddev and evaluate anomaly.
  * If rolling stddev is **0**, do **not** flag anomalies for that row and report **z-score as N/A** for the anomaly logic context (but see output requirements below—only anomalies are printed in the anomaly table).
* **Schema strictness**:

  * Required columns: `Date` and `Temperature`.
  * Header matching is **case-insensitive** (e.g., `date`, `DATE` accepted).
  * Leading/trailing whitespace should be **trimmed** from headers and values before validation/parsing.
  * If multiple columns match `Date` or `Temperature` (case-insensitive), treat as **schema error** and exit.
  * Must require a **header row**.
* **Units**: Assume **Fahrenheit**; label output as `F`.

### 3. Expected CSV schema and parsing rules

#### Required columns

* **Date**: `YYYY-MM-DD`
* **Temperature**: float

Other columns may exist and must be ignored.

#### Row validity rules

A row is considered to have a **data issue** (non-fatal) if any of the following occurs:

* `Date` value is not parseable as a real calendar date in `YYYY-MM-DD` format (including impossible dates like `2025-02-30`).
* `Temperature` is not strictly numeric (no unit suffixes like `72.3F`), after trimming whitespace.
* `Temperature` parses to a float but is **NaN** or **±inf** → treat as invalid/missing.
* The row is malformed/short relative to the header (e.g., missing columns) → treat as a data issue.

**Missing temperature definition**:

* “Any value that is not a number” counts as missing/invalid.
* Invalid/missing temperature rows must be **reported as errors** (data issues) and **excluded from rolling calculations**.

#### Do not silently drop rows

* Rows with data issues must be recorded and reported at the end in a **DATA ISSUES** section.
* Rows with data issues must **not** appear in the chart and must not affect rolling stats.

### 4. Input ordering constraints

* Input must be sorted by date in **strictly increasing** order.
* **Duplicates are not allowed**, and must trigger a specific error message (distinct from “out of order”).
* If dates are out of order or duplicates are found:

  * Stop immediately (fatal).
  * Print a clear error message.
  * Exit with code **1**.
* Sorting must *not* be performed automatically.

### 5. Anomaly detection specification

#### Rolling window definition

Maintain a rolling list of the **previous up to 30 valid temperature values** encountered *before* the current valid temperature row.

* Invalid temperature rows do not enter the window.
* The window grows gradually until it reaches 30.
* For a given row, compute stats from the window **excluding** current row.

#### Eligibility to evaluate anomaly

* Only evaluate anomalies for rows with valid temperature **and** at least **10** valid prior temperatures in the rolling window.
* If fewer than 10 valid priors exist, the row is valid but simply not eligible for anomaly detection.

#### Stats and threshold

* `mean` = average of rolling window
* `stddev` = sample standard deviation of rolling window (`ddof=1`)
* If `stddev == 0`, do not flag anomalies; treat z-score as N/A for evaluation.
* An anomaly is flagged if:

  * `abs(temp - mean) > ANOMALY_STDDEV_THRESHOLD * stddev`
  * where `ANOMALY_STDDEV_THRESHOLD` is a **named constant** with hard-coded value `2`.
* Report:

  * `diff = temp - mean` (signed)
  * `z = diff / stddev` (signed), if stddev > 0

### 6. Output requirements (stdout)

Output must include sections in this order:

1. **TEMPERATURE ANOMALY REPORT** header
2. **ASCII CHART**
3. **ANOMALIES**
4. **DATA ISSUES**

No additional “summary” section.

#### 6.1 ASCII chart

* Vertical timeline: **one line per valid row** (valid Date + numeric temperature only).
* Chart line format (compact):

  * Left: date
  * Middle: boundary bars and 70-char scale with marker
  * Right: temperature aligned as a trailing label

Example (illustrative):

```
ASCII CHART
-----------
2026-03-01 |--------------------*------------------------------------| 72.3F
2026-03-02 |--------------------#------------------------------------| 89.1F
...
|----------------------------------------------------------------------|
12.4F                                                              97.8F
```

Rules:

* Width of plotting region: **70 characters**
* Include **boundary bars**: `|` at both ends on each chart line.
* Normal points use marker `*`
* Anomalies use marker `#`
* Scaling:

  * Use min/max temperature **across all valid rows** (not including invalid rows).
  * Map each temperature to an integer position within `[0, 69]` (70 slots).
  * If `min == max`, place marker at center or position 0 (implementation-defined, but must be deterministic and tested).
* Bottom axis:

  * After all chart rows, print:

    * A boundary line: `|` + 70 dashes + `|`
    * Then a line with **min value at far left and max value at far right** (Option B):

      * `min` and `max` formatted with **1 decimal** and `F` (no degree symbol required).
      * Align max label to the right edge under the final boundary.

#### 6.2 Anomalies table

* Chronological order (same as input order).
* Must include columns:

  * `Date`
  * `Temp(F)`
  * `Mean(F)`
  * `Diff(F)` (signed, include `+` for positive)
  * `Z-Score` (signed)
* Formatting:

  * Temperature, mean, diff: **1 decimal**
  * Z-score: **1 decimal**
  * Use explicit `+` for positive diff (and optionally z-score, but at least diff must have explicit sign; keep consistent).
* Only include rows that were eligible and flagged as anomalies.
* If there are **no anomalies**, show the header and either no rows or a clear “(none)” line (developer choice; must be consistent and tested).

#### 6.3 Data issues section

* Listed at the end.
* Must include CSV **line numbers including header as line 1**.
* Must not stop execution for these issues; continue processing.
* Each issue should state:

  * Line number
  * A concise description (e.g., invalid date, non-numeric temperature, NaN/inf, malformed row)
  * Preferably include the raw value encountered (optional but helpful)

### 7. Error handling and exit codes

#### Fatal errors (must terminate)

1. File cannot be opened:

   * Print: `ERROR: Cannot open file '<filename>'`
   * Exit code **1**
2. Incorrect invocation (no args, too many args):

   * Print usage line: `Usage: python temp_anomal.py <input.csv>`
   * Exit code **1**
3. CSV schema errors:

   * Missing required columns (`Date`/`Temperature`, case-insensitive match)
   * Duplicate matching columns (case-insensitive)
   * Missing header row
   * Print clear error message (actionable)
   * Exit code **2**
4. Dates out of order or duplicates:

   * Stop immediately
   * Print clear error message specifying whether:

     * duplicate date found, or
     * out-of-order date found
   * Exit code **1**

#### Non-fatal errors (must continue)

* Invalid/missing temperature
* Invalid date (even if syntactically correct but impossible)
* Malformed row/short row
* These must be recorded and shown in **DATA ISSUES**.

### 8. Project structure

Single-script layout:

```
temp-anomaly/
├── temp_anomal.py
├── requirements.txt
├── AGENTS.md
├── README.md
└── tests/
    ├── test_schema_validation.py
    ├── test_anomaly_detection.py
    ├── test_missing_values.py
    └── test_chart_output.py
```

(Exact test filenames may vary, but must cover all behaviors.)

### 9. AGENTS.md requirements (must be created)

AGENTS.md must codify:

* Formatting: **black**
* Linting: **ruff**
* Type checking: **mypy**
* Testing: **pytest** with **coverage threshold** (e.g., ≥ 90%) using `pytest-cov`
* Pre-commit hooks: **pre-commit** required
* Layout: single-script layout as specified (no `src/`)
* Docstrings required for public functions
* Dependency rule: third-party deps allowed via `requirements.txt`
* Error handling rule: “All user-facing errors must be actionable” and “Do not print stack traces for expected user errors”

Also include a short “How to run” section consistent with README:

* create venv named `venv`
* `pip install -r requirements.txt`
* `pytest`
* formatting/lint/typecheck commands
* pre-commit install/run

### 10. README.md requirements

README must include:

* Overview
* Requirements (Python 3.12+, venv name `venv`)
* Setup instructions:

  * `python -m venv venv`
  * activate venv
  * `pip install -r requirements.txt`
* Usage:

  * `python temp_anomal.py <input.csv>`
* Required CSV schema (case-insensitive headers allowed; whitespace trimmed)
* Example input CSV
* Example output (sample chart + anomaly + data issues format)
* Running tests (`pytest`, and coverage expectation)

### 11. Testing plan (pytest)

Testing must include **both** unit/component tests and end-to-end “golden output” tests.

#### 11.1 Unit/component tests

Cover at least:

* Schema validation:

  * missing Date/Temperature → exit code 2
  * duplicate matching columns (case-insensitive) → exit code 2
  * header trimming and case-insensitivity
* Date ordering validation:

  * strictly increasing enforced
  * duplicates produce specific error and exit code 1
  * out-of-order produces specific error and exit code 1
* Missing/invalid values:

  * non-numeric temperature handled as data issue
  * `nan`, `inf`, `-inf` treated as data issue
  * invalid/impossible dates treated as data issue
  * malformed/short rows treated as data issue
  * ensure invalid rows are excluded from rolling window and chart
* Rolling window mechanics:

  * window uses previous 30 valid rows only
  * window grows gradually, anomaly checks begin only at ≥10 valid priors
  * sample stddev (`ddof=1`) behavior
  * stddev == 0 behavior (no anomaly)
* Detection logic:

  * two-sided anomalies (high and low)
  * threshold strictly “more than” 2 stddev (not ≥)

#### 11.2 Golden output tests (end-to-end)

At least one test that:

* Creates a small CSV fixture with:

  * valid rows
  * at least one high anomaly and one low anomaly
  * at least one invalid temperature
  * at least one invalid date
* Runs the script (e.g., via `subprocess.run`) and asserts:

  * exit code is 0
  * stdout matches an expected “golden” output file/string exactly
  * includes chart markers `*` and `#`, correct min/max labels, and correct anomaly table formatting

Additional golden tests recommended:

* No anomalies case
* Only <10 valid temperatures case (no anomalies, still chart prints)

