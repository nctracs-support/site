"""Direct in-process tests for main() via dependency-injected argv."""
import pytest
from temp_anomaly import main


def _run(capsys, argv: list[str]):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return exc.value.code, capsys.readouterr()


# --- usage / arg count ---

def test_no_args_usage(capsys):
    code, out = _run(capsys, ["prog"])
    assert code == 1
    assert out.out == "Usage: python temp_anomaly.py <input.csv>\n"
    assert out.err == ""


def test_too_many_args_usage(capsys):
    code, out = _run(capsys, ["prog", "a.csv", "b.csv"])
    assert code == 1
    assert out.out == "Usage: python temp_anomaly.py <input.csv>\n"


# --- file errors ---

def test_missing_file_error(capsys):
    code, out = _run(capsys, ["prog", "no_such_file.csv"])
    assert code == 1
    assert out.out == "ERROR: Cannot open file 'no_such_file.csv'\n"


def test_invalid_utf8_error(capsys, tmp_path):
    f = tmp_path / "bad.csv"
    f.write_bytes(b"\xff\xfe\xff")
    code, out = _run(capsys, ["prog", str(f)])
    assert code == 1
    assert out.out == f"ERROR: Cannot decode file '{f}' as UTF-8\n"


# --- schema errors ---

def test_missing_header(capsys, tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("only_one_column\n")
    code, out = _run(capsys, ["prog", str(f)])
    assert code == 2
    assert out.out == "ERROR: Missing header row\n"


def test_duplicate_date_column(capsys, tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("Date,DATE,Temperature\n")
    code, out = _run(capsys, ["prog", str(f)])
    assert code == 2
    assert out.out == "ERROR: Duplicate column 'Date'\n"


def test_missing_temperature_column(capsys, tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("Date,Other\n")
    code, out = _run(capsys, ["prog", str(f)])
    assert code == 2
    assert out.out == "ERROR: Missing required column 'Temperature'\n"


# --- ordering errors ---

def test_duplicate_date_ordering(capsys, tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("Date,Temperature\n2026-01-01,70\n2026-01-01,71\n")
    code, out = _run(capsys, ["prog", str(f)])
    assert code == 1
    assert out.out == "ERROR: Duplicate date encountered at line 3: 2026-01-01\n"


def test_out_of_order_date(capsys, tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("Date,Temperature\n2026-01-02,70\n2026-01-01,71\n")
    code, out = _run(capsys, ["prog", str(f)])
    assert code == 1
    assert out.out == "ERROR: Date out of order at line 3: 2026-01-01 after 2026-01-02\n"


# --- successful output paths ---

def test_valid_no_data_issues(capsys, tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("Date,Temperature\n2026-01-01,50.0\n2026-01-02,51.0\n")
    main(["prog", str(f)])
    out = capsys.readouterr()
    assert out.out.startswith("TEMPERATURE ANOMALY REPORT\n")
    assert "ANOMALIES\n(none)\n" in out.out
    assert "DATA ISSUES\n" in out.out
    assert out.err == ""


def test_valid_with_data_issues(capsys, tmp_path):
    f = tmp_path / "d.csv"
    f.write_text(
        "Date,Temperature\n"
        "2026-01-01,50.0\n"
        "bad-date,70\n"
        "2026-01-02,nan\n"
        "2026-01-03,51.0\n"
    )
    main(["prog", str(f)])
    out = capsys.readouterr()
    assert "Line 3: invalid date: bad-date\n" in out.out
    assert "Line 4: non-numeric temperature: nan\n" in out.out


def test_valid_with_anomaly_and_hash_marker(capsys, tmp_path):
    lines = ["Date,Temperature"]
    for i in range(10):
        lines.append(f"2026-01-{i+1:02d},{50+i}.0")
    lines.append("2026-01-11,100.0")
    f = tmp_path / "d.csv"
    f.write_text("\n".join(lines) + "\n")
    main(["prog", str(f)])
    out = capsys.readouterr()
    assert "#" in out.out
    assert "2026-01-11" in out.out
    assert "(none)" not in out.out


def test_malformed_row_in_data_issues(capsys, tmp_path):
    f = tmp_path / "d.csv"
    # Row with fewer columns than header -> malformed
    f.write_text("Date,Temperature\n2026-01-01\n2026-01-02,51.0\n")
    main(["prog", str(f)])
    out = capsys.readouterr()
    assert "Line 2: malformed row\n" in out.out


def test_argv_none_uses_sys_argv(monkeypatch, capsys):
    # Exercises the argv=None -> sys.argv branch
    monkeypatch.setattr("sys.argv", ["prog"])
    with pytest.raises(SystemExit) as exc:
        main()  # called with no argument
    assert exc.value.code == 1


def test_min_equals_max_uses_center(capsys, tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("Date,Temperature\n2026-01-01,50.0\n")
    main(["prog", str(f)])
    out = capsys.readouterr()
    assert "|" + "-" * 35 + "*" + "-" * 34 + "|" in out.out
