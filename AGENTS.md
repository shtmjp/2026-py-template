# Repository maintenance instructions

- Develop under the assumption that Python files are executed from the repository root, e.g., `uv run src/xxx.py`.
- Use `uv add` to add packages, e.g., `uv add numpy`.
- Do not edit `pyproject.toml`, `ruff.toml`, or `uv.lock` directly.
- Ensure the test suite passes with `uv run pytest`.
- Write docstring in numpy style for all public functions/classes.
- Run linting and type checking as described in the **Lint and type checking** section below.

## IMPORTANT: Research-first implementation policy

Use the minimum complexity that keeps scientific correctness auditable.

- **Assume Trusted Inputs:** This is a private research repository. Assume inputs to functions and methods are correct and valid.
- **NO Defensive Programming:** Do NOT write excessive runtime validation (e.g., `if not isinstance(...)`, `if val is None: raise...`, bounds checking).
- **Focus on Core Logic:** Prioritize the readability and mathematical correctness of the core algorithms (e.g., numerical optimization, inference logic) over edge-case exception handling.
- Prefer direct, linear implementations over many tiny helper functions.
- Add abstractions only when they clearly improve readability or are reused.
  - Rule of thumb: extract only when logic is reused, or when one block is too long to scan.

## Lint and type checking

- Format Python code with `ruff format` and ensure `ruff check` produces no
  warnings.
- Run static type checking with `uv run ty check`.

## Directory layout

- `src/` holds the Python package.
- `tests/` stores the pytest suite.
- `notebooks/` contains marimo notebooks.
- `data/` is for raw data files.

## Repo-local agent assets

- Repo-local skills live under `.agents/my-skills`.
- `https://github.com/shtmjp/my-skills.git` is the single source of truth.
- In this repository, update only the `.agents/my-skills` submodule pointer. Do not edit skill contents in place.
- For a fresh clone, initialize submodules with `git submodule update --init --recursive`.
- To move to the latest published skills on `main`, run `git submodule update --init --remote .agents/my-skills` and commit the gitlink change.

## Available repo-local skills

- `joblib-numpy-rng-spawn`: Implement reproducible NumPy RNG spawning for `joblib.Parallel` pipelines. (file: `.agents/my-skills/joblib-numpy-rng-spawn/SKILL.md`)
- `marimo-notebook-rules`: Apply repository-specific guardrails when editing marimo notebooks. (file: `.agents/my-skills/marimo-notebook-rules/SKILL.md`)
