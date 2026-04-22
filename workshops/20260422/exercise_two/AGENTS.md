# AGENTS.md

## Environment & Tooling
- Dependency management: Use `pip` (via `pyproject.toml`). Do NOT use `requirements.txt`.
- Python version: 3.12+
- Virtual environment named `venv`

## Engineering Standards
- Type hints: Required on all public function signatures and routes.
- Error handling: Never use a bare `except:` or `except Exception: pass`. Handle specific errors explicitly.

## Security & Safety
- SQL: Never construct SQL queries using string formatting (e-strings, `.format()`, or `%s`). Always use database driver parameterization.
- Git: Never commit virtual environments, caches, or secrets.
