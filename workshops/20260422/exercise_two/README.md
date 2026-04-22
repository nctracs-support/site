# Exercise Two: Red Teaming Code

In this exercise you will use Claude Code as a code reviewer — giving it a deliberately flawed codebase and evaluating how well it finds real problems. The codebase contains six intentional issues spanning security, error handling, test quality, and project hygiene. Your job is to prompt the agent, judge its findings, and then use it to apply fixes.

## Known Issues

| # | Issue | Category |
|---|-------|----------|
| 1 | requirements.txt instead of pyproject.toml | Toolchain compliance |
| 2 | SQL query built with f-string formatting | Security (injection) |
| 3 | Bare except: pass swallowing errors | Error handling |
| 4 | Missing type hints on public functions | Code standards |
| 5 | Test that passes but asserts the wrong variable | Test quality |
| 6 | No .gitignore — venv/cache would be committed | Git hygiene |

## Tasks

1. **Agent review** — Prompt Claude Code with the following:

   > Review this codebase against AGENTS.md for compliance. Check for: security issues, exception handling problems, test quality, dependency management, type hint coverage, and Git hygiene. For each issue, explain the risk and provide the exact file and line number.

2. **Manual verification** — Read the agent's findings and cross-reference with the code. Did it catch all six? Did it miss any? Did it flag anything that is actually fine (false positive)?

3. **Fix** — Ask the agent to fix the highest-severity issue: apply the fix, run tests, and commit. Alternatively, enter planning mode, review a plan to fix all issues, then execute it.

4. **Approval gating** — If the fix requires a dependency change, observe the approval gate in action — the agent should pause and ask before installing anything.

> Alternative: use `/simplify` in Claude Code or `/review` in Codex.
>
> This launches three subagents:
> * Code resuse review
> * Code quality review
> * Efficiency review
