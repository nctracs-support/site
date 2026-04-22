# AGENTS

Guidelines for AI agents and contributors working on this project.

## Toolchain

| Tool | Purpose | Command |
|------|---------|---------|
| `black` | Code formatting | `black .` |
| `ruff` | Linting | `ruff check .` |
| `mypy` | Static type checking | `mypy .` |
| `pytest` + `pytest-cov` | Testing and coverage | `coverage run -m pytest && coverage report -m` |
| `pre-commit` | Pre-commit hooks | `pre-commit run --all-files` |

## Quality Gates

- **Coverage**: `coverage run -m pytest` must achieve ≥ 90% on `temp_anomaly.py`
- **Formatting**: `black --check .` must pass with no changes
- **Linting**: `ruff check .` must report zero violations
- **Types**: `mypy .` must report zero errors
- **Pre-commit hooks**: must be installed (`pre-commit install`) and pass before every commit

## Coding Rules

1. **Docstrings**: every public function must have a docstring describing its purpose, parameters, and return value.
2. **No stack traces for user errors**: all user-facing errors must be caught and printed as a clean message to stdout; `sys.exit` with the correct code. Never let an unhandled exception reach the user.
3. **Error messages match spec exactly**: the text of every error message (usage, file errors, schema errors, ordering errors) must match the specification character-for-character, including punctuation and capitalisation.

## Running the Full Check

```bash
source venv/bin/activate
black --check .
ruff check .
mypy .
coverage run -m pytest -q
coverage report -m --fail-under=90
```
