## Step-by-step blueprint

### 1) Core behaviors and invariants

* **CLI contract**

  * `python temp_anomaly.py <input.csv>`
  * Exactly 1 positional arg, otherwise print usage to **stdout** and exit **1**
  * Never print to stderr
* **File decoding**

  * Open as UTF-8 with optional BOM (`utf-8-sig`)
  * Decode failure → `ERROR: Cannot decode file '<filename>' as UTF-8` exit **1**
  * Open failure → `ERROR: Cannot open file '<filename>'` exit **1**
* **CSV header + schema**

  * Header row = first parseable CSV row with **≥ 2 columns**
  * If missing/empty/whitespace/<2 cols → `ERROR: Missing header row` exit **2**
  * Headers: trim + case-insensitive match for required columns `Date`, `Temperature`
  * Missing required column → `ERROR: Missing required column 'Temperature'` exit **2** (or `Date`)
  * If multiple columns match required (duplicate match) → `ERROR: Duplicate column 'Date'` exit **2** (or `Temperature`)
* **Row parsing**

  * Trim all cell values
  * Date: must be valid `YYYY-MM-DD` calendar date; invalid date is a **data issue** (non-fatal)
  * Temperature: finite float only; invalid (units, NaN, ±inf, non-numeric) is a **data issue**
  * Malformed row: shorter than header column count is a **data issue**
* **Chart-eligible row**

  * Date valid AND temperature valid finite float
  * Only eligible rows participate in:

    * ordering validation
    * rolling stats/anomaly detection
    * chart scaling
    * chart rows
* **Ordering validation (eligible rows only)**

  * Must be strictly increasing
  * Duplicate eligible date → fatal `ERROR: Duplicate date encountered at line <line>: <date>` exit **1**
  * Out-of-order eligible date → fatal `ERROR: Date out of order at line <line>: <date> after <previous_date>` exit **1**
  * Must be detected during streaming parse; do not sort input
* **Rolling window**

  * Keep previous up to 30 eligible temperatures; current row excluded from stats
  * Need **≥ 10 priors** to compute anomalies; otherwise skip detection for that row
  * Stddev is **sample stddev (ddof=1)**; if stddev==0 → skip detection
  * Anomaly if `abs(temp - mean) > 2 * stddev` (strict)
* **Output (stdout only), deterministic, section order**

  1. `TEMPERATURE ANOMALY REPORT`
  2. `ASCII CHART`
  3. `ANOMALIES`
  4. `DATA ISSUES`

### 2) Streaming + scaling strategy (bounded memory where it matters)

ASCII chart scaling requires global `min_temp`/`max_temp` over all eligible rows. To avoid storing millions of rows:

* **Pass 1:** parse + validate ordering (eligible rows only), collect **data issues strings**, compute `min_temp`/`max_temp`

  * If an ordering fatal error occurs: stop immediately (exit 1)
  * If schema/header fatal error: exit 2
* **Pass 2:** re-parse file (same rules), re-validate ordering defensively, compute anomalies (rolling window), print chart lines as we stream, store anomalies list for table later
* Finally print anomalies table (or `(none)`), then print data issues (already collected pass 1) in ascending line order (they naturally are if appended while reading)

> Note: storing *all* data issues can be large if the input is very dirty. The spec requires listing every issue; without temporary files (not in the allowed import list), the only compliant option is to keep them in memory.

### 3) Internal design (single script, testable functions)

Inside `temp_anomaly.py`, keep everything in one file but organized:

* Constants:

  * `ANOMALY_STDDEV_THRESHOLD = 2`
  * `ROLLING_WINDOW_SIZE = 30`
  * `MIN_PRIORS_FOR_DETECTION = 10`
  * `PLOT_WIDTH = 70`
* Data structures:

  * `ParsedRow` (date, temp, line_no) for eligible rows (could be a small `dataclass`)
  * `Anomaly` (date, temp, mean, diff, z)
* Functions (pure where possible):

  * `parse_args(argv) -> Path | None`
  * `open_text(path) -> TextIO` with correct error handling and messages
  * `read_header(reader) -> (header_list, header_line_no)` implementing “first parseable CSV row with ≥2 cols”
  * `resolve_required_columns(headers) -> (date_idx, temp_idx)` with trimming/case-insensitive/duplicate detection
  * `parse_row(raw_row, line_no, header_len, date_idx, temp_idx) -> (eligible_row | None, issue_strings[])`
  * `validate_order(prev_date, curr_date, line_no) -> new_prev_date` or raise a controlled exception
  * `rolling_mean_stddev(window) -> (mean, stddev) | None`
  * `is_anomaly(temp, mean, stddev) -> bool`
  * `format_chart_line(date, pos, is_anom, temp) -> str`
  * `format_axis(min_temp, max_temp) -> (axis_line, labels_line)`
  * `format_anomaly_row(anom) -> str`
  * `run(path) -> (exit_code, stdout_text)` for easier unit testing (optional), or just print directly but keep logic testable

### 4) Testing strategy

* Unit tests (directly call helper functions)

  * Schema/header parsing, trimming, duplicates
  * Ordering rules only applied to eligible rows
  * Temperature parsing (NaN/inf/non-numeric)
  * Rolling window: size cap 30, min priors 10, ddof=1
  * Strict `>` threshold behavior
  * Chart scaling mapping and special case `min==max`
* Golden tests (subprocess)

  * Mixed good + invalid rows + anomalies + invalid dates; compare stdout exact
  * No anomalies → `(none)`
  * Fewer than 10 eligible rows → no anomalies
  * Constant temperature (stddev=0) → no anomalies, chart centered if min==max
  * Invocation errors and fatal ordering errors

---

## Iterative chunking (big → smaller → smallest implementable steps)

### Round 1: Major milestones

1. Project scaffolding: requirements, README, AGENTS, pytest wiring
2. CLI + error handling for invocation/open/decode
3. CSV header + schema validation
4. Row parsing + data issue recording
5. Ordering validation during streaming
6. Pass 1: compute min/max and collect issues
7. Pass 2: rolling window + anomaly detection + chart output
8. Anomalies table formatting + data issues section formatting
9. Golden tests for deterministic full output

### Round 2: Break milestones into smaller increments

1.1 Add `requirements.txt` dev deps, minimal README/AGENTS placeholders
1.2 Add `pytest` setup and first “smoke” test
2.1 Implement `main()` + usage handling
2.2 Implement file open + decode errors (stdout only)
3.1 Implement “missing header row” detection
3.2 Implement required column resolution (case-insensitive, trim)
3.3 Implement duplicate column match detection
4.1 Implement malformed row detection (short row)
4.2 Implement date parsing + invalid date issue
4.3 Implement temp parsing + invalid temp issues (NaN/inf)
5.1 Implement eligible-only ordering validation + fatal errors
6.1 Implement pass1 scanner: min/max over eligible; collect issues
6.2 Implement chart position mapping + axis lines
7.1 Implement rolling window + mean/stddev (ddof=1)
7.2 Implement anomaly detection rule (strict) + anomaly record
7.3 Print chart rows streaming in pass2 (store anomalies list)
8.1 Implement anomalies table rendering + `(none)` case
8.2 Implement DATA ISSUES rendering in ascending line order
9.1 Golden test with mixed conditions
9.2 Additional goldens: no anomalies, <10 rows, stddev=0
9.3 Coverage enforcement and lint/type hooks in AGENTS

### Round 3: Ensure each step is “safe-sized” (TDD-friendly, no big jumps)

* Every step introduces **one new behavior**, with unit tests first.
* Integration/golden tests only after core formatting is stable.
* No orphaned code: each prompt ends with runnable tests and/or CLI validation.

---

## Prompts for a code-generation LLM (TDD, incremental, wired each step)

> Each prompt is self-contained, but assumes the repo state from previous prompts.
> **Every prompt ends with human validation steps.**
> Prompts are tagged as `text` using code fences.

---

### Prompt 1 — Create project files and test harness

```text
You are implementing the project in a folder `temp-anomaly/` with the required structure.

Task (TDD-first):
1) Create:
- requirements.txt containing ONLY these dev deps (pinned not required): pytest, pytest-cov, black, ruff, mypy, pre-commit
- temp_anomaly.py (can be minimal placeholder)
- tests/ directory with an initial test file tests/test_smoke.py
- README.md and AGENTS.md as placeholders (you will expand them later)

2) In tests/test_smoke.py write a test that runs `python temp_anomaly.py` with no args using subprocess.run, capturing stdout/stderr.
- Assert returncode == 1
- Assert stdout == "Usage: python temp_anomaly.py <input.csv>\n"
- Assert stderr == "" (must be empty)

3) Implement the minimal temp_anomaly.py to satisfy the test:
- Exactly one positional arg required; otherwise print usage to stdout only and exit 1.
- Do not write to stderr.

Constraints:
- Python 3.12+
- Keep code small; no extra features yet.

Human validation:
- Run: pytest -q
- Run manually: python temp_anomaly.py
Expected: prints usage line to stdout and exits 1.
```

---

### Prompt 2 — File open error handling (stdout only)

```text
Add file opening behavior with correct error message.

TDD:
1) Create tests/test_file_errors.py with a subprocess test:
- Run: python temp_anomaly.py does_not_exist.csv
- Assert returncode == 1
- Assert stdout == "ERROR: Cannot open file 'does_not_exist.csv'\n"
- Assert stderr == ""

2) Implement in temp_anomaly.py:
- If exactly one arg is given, attempt to open that file path.
- If open fails (OSError / FileNotFoundError), print exactly:
  ERROR: Cannot open file '<filename>'
  to stdout, newline, exit 1.

Do NOT implement decoding rules yet; just opening.

Human validation:
- pytest -q
- python temp_anomaly.py does_not_exist.csv
```

---

### Prompt 3 — UTF-8 (with BOM) decode failure message

```text
Implement UTF-8 decoding with optional BOM and correct decode-failure handling.

TDD:
1) In tests/test_file_errors.py add a test that creates a temporary file containing invalid UTF-8 bytes.
- Use pytest tmp_path to create e.g. bad.csv
- Write bytes like b"\xff\xfe\xff"
- Run subprocess: python temp_anomaly.py <path>
- Assert returncode == 1
- Assert stdout == "ERROR: Cannot decode file '<filename>' as UTF-8\n"
  where <filename> is the basename (match exactly what the script prints; use the argv string you pass).
- Assert stderr == ""

2) Implement:
- Open the file in binary, decode as 'utf-8-sig' (or open text with encoding='utf-8-sig').
- If UnicodeDecodeError occurs, print the exact error string and exit 1.

Do not parse CSV yet.

Human validation:
- pytest -q
```

---

### Prompt 4 — Header row detection and “Missing header row” schema error

```text
Implement header detection per spec:
- A header row is the first parseable CSV row containing at least two columns.
- If the file is empty/whitespace/has fewer than two columns total, it's a schema error:
  "ERROR: Missing header row" exit code 2.

TDD:
1) Add tests/test_schema_validation.py:
- Case A: empty file -> exit 2, stdout == "ERROR: Missing header row\n"
- Case B: file with only whitespace lines -> same
- Case C: file with a single-column row like "onlyone\n" -> same
All via subprocess, assert stderr == "".

2) Implement minimal CSV reading:
- Use csv.reader over text opened with encoding='utf-8-sig'
- Find the first row where len(row) >= 2 => that's the header
- If never found => print error and exit 2

Do not validate required columns yet.

Human validation:
- pytest -q
```

---

### Prompt 5 — Required columns: trimming + case-insensitive matching

```text
Add required column resolution for Date and Temperature.

TDD:
1) Extend tests/test_schema_validation.py with subprocess tests:
- Valid headers:
  "Date,Temperature\n" (should proceed further; for now you can exit 0 after schema passes)
  " DATE , TEMPERATURE \n" (trimming)
  "date,temperature\n" (case-insensitive)
For these, since no further behavior exists yet, require:
- exit code == 0
- stdout starts with "TEMPERATURE ANOMALY REPORT\n" (you can print only that line for now)
- stderr == ""

2) Add missing column tests:
- Header "Date,Temp\n" -> exit 2 and stdout == "ERROR: Missing required column 'Temperature'\n"
- Header "Temperature,Value\n" -> exit 2 and stdout == "ERROR: Missing required column 'Date'\n"

Implementation notes:
- Trim header cells
- Compare lowercased to 'date' and 'temperature'
- If missing: exact error above, exit 2
- If schema ok: print "TEMPERATURE ANOMALY REPORT" line and exit 0 for now.

Human validation:
- pytest -q
```

---

### Prompt 6 — Duplicate column match detection

```text
Detect multiple columns matching Date or Temperature (case-insensitive), as a fatal schema error with exit code 2.

TDD:
1) In tests/test_schema_validation.py add:
- Header "Date,DATE,Temperature\n" -> exit 2 and stdout == "ERROR: Duplicate column 'Date'\n"
- Header "Date,Temperature,TEMPERATURE\n" -> exit 2 and stdout == "ERROR: Duplicate column 'Temperature'\n"
stderr must be empty.

2) Implement:
- When scanning headers, count matches for each required column.
- If match count > 1, print exact duplicate message, exit 2.

Keep previous behaviors intact.

Human validation:
- pytest -q
```

---

### Prompt 7 — Parse rows: malformed rows + invalid date/temp become data issues (not fatal)

```text
Introduce row parsing rules and the DATA ISSUES section, but no ordering/anomalies/chart yet.

TDD:
1) Add tests/test_missing_values.py with a subprocess test using a CSV fixture like:
Date,Temperature,Extra
2026-01-01,72.0,x
2026-01-02            (malformed row: fewer columns than header count)
2026-02-30,70.0,x     (invalid date)
2026-01-04,seventy,x  (non-numeric temp)
2026-01-05,nan,x      (NaN invalid)
2026-01-06,inf,x      (+inf invalid)
2026-01-07,-inf,x     (-inf invalid)
2026-01-08,73.0,x

Expected (for now): program exits 0 and prints these sections in exact order:
TEMPERATURE ANOMALY REPORT
ASCII CHART
ANOMALIES
DATA ISSUES

For now, ASCII CHART can be empty (no rows yet), and ANOMALIES can print "(none)".

But DATA ISSUES must list entries in ascending line order (line numbers include header line as 1):
Line 3: malformed row
Line 4: invalid date: 2026-02-30
Line 5: non-numeric temperature: seventy
Line 6: non-numeric temperature: nan
Line 7: non-numeric temperature: inf
Line 8: non-numeric temperature: -inf

Each on its own line.

2) Implement:
- Malformed row: len(row) < header_column_count => data issue
- Date parse: YYYY-MM-DD using datetime.date.fromisoformat after validating structure; impossible dates invalid
- Temp parse: float(value) then reject non-finite via math.isfinite
- Trim cell values
- Record issues as strings; do not include invalid rows in later processing
- Emit the 4 required section headers always (even if content minimal)

Human validation:
- pytest -q
- Run script on the fixture and visually confirm DATA ISSUES formatting.
```

---

### Prompt 8 — Eligible-only ordering validation (fatal, streaming)

```text
Add ordering validation for chart-eligible rows only. Must stop immediately when detected. Must NOT sort input.

TDD:
1) Create tests/test_ordering_validation.py with subprocess tests:

Case A: Duplicate eligible dates:
Date,Temperature
2026-01-01,70
2026-01-01,71
Expect exit 1 and stdout:
ERROR: Duplicate date encountered at line 3: 2026-01-01

Case B: Out of order eligible dates:
Date,Temperature
2026-01-02,70
2026-01-01,71
Expect exit 1 and stdout:
ERROR: Date out of order at line 3: 2026-01-01 after 2026-01-02

Case C: Invalid rows do NOT affect ordering:
Date,Temperature
2026-01-02,70
bad-date,71        (invalid date)
2026-01-03,72
Expect exit 0 (no ordering error)

2) Implement:
- While reading rows, for each eligible row only:
  - enforce strictly increasing dates
  - if duplicate/out-of-order: print exact error, exit 1 immediately
- Do not include invalid-date or invalid-temp rows in ordering checks

Keep DATA ISSUES behavior.

Human validation:
- pytest -q
```

---

### Prompt 9 — Pass 1: compute min/max over eligible rows (streaming) + chart scaling helpers

```text
Implement pass-1 scanning to compute min_temp and max_temp over eligible rows and build chart-scaling helpers, but still don’t print chart rows.

TDD:
1) Add unit tests (not subprocess) in tests/test_chart_output.py for a pure function:
- Given min_temp=10.0 max_temp=20.0:
  - temp=10.0 => pos 0
  - temp=20.0 => pos 69
  - temp=15.0 => pos round((5/10)*69)=round(34.5)=34 (Python round)
- Clamp behavior at extremes
- Special case min==max => pos 35

2) Implement in temp_anomaly.py:
- A function like compute_pos(temp, min_temp, max_temp) -> int following spec.

No change to CLI output yet besides internal plumbing.

Human validation:
- pytest -q
```

---

### Prompt 10 — Print full ASCII chart rows (markers default to normal for now) + bottom axis

```text
Print the ASCII CHART section fully for eligible rows. For now, treat every eligible row as normal (marker '*') and anomalies will still be "(none)".

TDD (subprocess golden-ish but small):
1) In tests/test_chart_output.py add a subprocess test with CSV:
Date,Temperature
2026-01-01,10.0
2026-01-02,15.0
2026-01-03,20.0

Expect:
- exit 0
- Output sections in order
- Under ASCII CHART, three rows formatted:
YYYY-MM-DD |<70 chars>| XX.XF
with '*' at the computed position, '-' elsewhere.
- Then the axis line exactly:
|----------------------------------------------------------------------|
- Then a labels line with:
"10.0F" on the left and "20.0F" aligned so the max label ends under the final boundary.

Keep ANOMALIES as "(none)" and DATA ISSUES empty (just the header line, no entries).

2) Implement:
- Two-pass approach:
  - Pass 1: parse, validate ordering, collect issues, compute min/max
  - Print headers:
    TEMPERATURE ANOMALY REPORT
    ASCII CHART
  - Pass 2: parse again, validate ordering, for each eligible row:
    - compute pos
    - print chart row with '*' marker
  - print axis + labels line per spec
  - print ANOMALIES + "(none)"
  - print DATA ISSUES + all issues

Human validation:
- pytest -q
- Manually run the script and confirm chart width is exactly 70 between bars.
```

---

### Prompt 11 — Rolling window stats (ddof=1) + anomaly detection (but don’t change chart marker yet)

```text
Implement rolling window and anomaly detection logic; record anomalies, but keep chart markers as '*' for now.

TDD:
1) Add tests/test_anomaly_detection.py with unit tests for:
- rolling window excludes current row
- requires >=10 priors
- window max size 30 (use a sequence and verify size)
- stddev uses sample ddof=1 (compare to statistics.stdev)
- strict inequality: abs(diff) > 2*stddev (not >=)
- stddev==0 => skip detection

2) Implement:
- Maintain a list of prior temps (eligible only), pop oldest when >30
- For each eligible row:
  - if priors < 10: no detection
  - else mean = sum(priors)/len(priors)
  - stddev = statistics.stdev(priors) (sample)
  - if stddev==0: skip
  - if anomaly: store an Anomaly record with date,temp,mean,diff,z
- Append current temp to window after processing

No output changes required yet beyond internal anomaly list accumulation.

Human validation:
- pytest -q
```

---

### Prompt 12 — Mark anomalies in the chart with '#', normal with '*'

```text
Change chart markers:
- '#' if the row is an anomaly
- '*' otherwise

TDD:
1) Add a subprocess test (tests/test_chart_output.py or a new file) with a constructed dataset that guarantees at least one anomaly after >=10 priors.
Example idea:
- 10 days at temp 50.0
- day 11 at temp 100.0 (stddev==0 for priors, so anomaly detection skipped!)
So instead create priors with small variance, e.g. temps 50..59 (10 values), then an extreme 100.
Verify:
- Exit 0
- Chart line for anomaly date contains '#' at correct position.

2) Implement:
- During pass 2, after you compute anomaly status for the current row, use '#' marker if anomaly else '*'.
- Still print anomalies table as "(none)" for now (next prompt will format it).

Human validation:
- pytest -q
- Manually inspect the output to see '#' on anomaly row.
```

---

### Prompt 13 — Render the ANOMALIES table with exact formatting + (none) case

```text
Implement the ANOMALIES section table rendering.

TDD:
1) In tests/test_anomaly_detection.py add a subprocess test with a dataset that produces exactly one anomaly.
Assert:
- The ANOMALIES section includes header row:
Date  Temp(F)  Mean(F)  Diff(F)  Z-Score
(Spacing can be flexible ONLY if you choose fixed-width; but the numeric formats must match exactly.)
- The anomaly line contains:
- Temp/Mean/Diff to 1 decimal
- Diff with explicit sign (+/-)
- Z-Score with explicit sign (+/-) and 1 decimal
- Chronological order (they will be encountered in order)

2) Keep "(none)" exactly when there are no anomalies.

Implementation:
- Store anomalies during pass2, then after chart+axis print:
ANOMALIES
<either (none) or table>

Human validation:
- pytest -q
- Run on a sample file and confirm numeric formats.
```

---

### Prompt 14 — Full deterministic golden test (mixed anomalies + invalid rows + invalid dates)

```text
Add at least one full golden test per spec.

TDD:
1) Create tests/test_golden_output.py that:
- Writes a CSV fixture including:
  - valid eligible rows (enough to create anomalies)
  - malformed row
  - invalid date row
  - non-numeric temp row
  - NaN/inf temp rows
- Runs subprocess: python temp_anomaly.py <fixture>
- Asserts:
  - returncode == 0
  - stderr == ""
  - stdout == expected_output (exact match)

2) In the expected_output string, ensure the four sections appear in exact order with exact formatting.

Implementation adjustments:
- Fix any minor formatting drift uncovered by the golden test:
  - chart width 70
  - axis and labels alignment (max label ends under final boundary)
  - deterministic rounding via Python round() and format to 1 decimal
  - DATA ISSUES in ascending line order with line numbers including header as line 1

Human validation:
- pytest -q
- Re-run the golden test twice to ensure identical stdout both times.
```

---

### Prompt 15 — Documentation + AGENTS.md requirements + coverage target

```text
Finalize README.md and AGENTS.md exactly per spec.

TDD-ish:
1) Add a simple test (optional) that checks README.md and AGENTS.md exist; keep it lightweight.
2) Update AGENTS.md to specify:
- black formatting
- ruff linting
- mypy type checking
- pytest + pytest-cov
- coverage >= 90%
- pre-commit hooks required
- rules: public functions have docstrings; no stack traces for user errors; error messages match spec exactly
3) Update README.md with:
- overview
- Python 3.12+ requirement
- venv named venv
- install instructions
- usage example
- required CSV schema
- example input
- example output
- running tests

Human validation:
- Open README.md and verify it contains every required bullet.
- Run: pytest -q --cov --cov-fail-under=90
- Run: ruff check . ; black --check . ; mypy .
```

---

If you want, I can also include a ready-to-paste **expected_output** template for the golden test (with careful spacing), but the prompts above already force the LLM to make the formatting exact and verifiable step by step.
