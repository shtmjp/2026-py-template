# Repository maintenance instructions

- Develop under the assumption that Python files are executed from the repository root, e.g., `uv run --env-file .env src/xxx.py`.
- Use `uv add` to add packages, e.g., `uv add numpy`.
- Do not edit `pyproject.toml`, `ruff.toml`, or `uv.lock` directly.
- Ensure the test suite passes with `uv run pytest`.
- Run linting and type checking as described in the **Lint and type checking** section below.
- See the **Directory layout** section for local conventions in each major directory.

## Lint and type checking

- Format Python code with `ruff format` and ensure `ruff check` produces no
  warnings.
- Run static type checking with `uv run ty check`.

## Directory layout

- `src/` holds the Python package.
- `tests/` stores the pytest suite.
- `notebooks/` contains marimo notebooks (see **Marimo notebooks rule** section).
- `data/` is for raw data files.
Check the file in each directory before modifying code there.

## Marimo notebooks rule

## Non-negotiables

- Do not remove or rewrite the marimo scaffold (`import marimo`, `__generated_with`, `app = marimo.App(...)`, and the `if __name__ == "__main__": app.run()` footer).
- Cells are stored as `@app.cell` functions.
  - Cell signature = referenced globals
  - `return` values = globals defined/exported by the cell
  - Final statement/expression = displayed output
- Don’t micromanage cell signatures/returns; marimo may rewrite them.
- Never use `from X import *` (star imports).
- Nothing inside `with app.setup:` may depend on variables defined in other cells.

## Reactive dataflow constraints (DAG, not linear)

- Execution order is based on variable references (not file order).
- Every global name (imports, functions, classes included) must be defined by exactly one cell.
- Do not introduce circular dependencies between cells.
- Mutations are not tracked: don’t define in one cell and mutate in another.
  - If mutation is required, mutate in the defining cell or create a new variable.
- Use underscore-prefixed locals (`_x`) for cell-local temporaries (safe to reuse across cells).

## Reusable top-level API (only if needed)

- If you need importable functions/classes, use `@app.function` / `@app.class_definition` and keep them pure
  (only close over imports/constants from `with app.setup:`).

### After edits

- Validate: `marimo check <file_or_dir>` (use `--fix` for fixable formatting issues).

### Ask before

- Renaming/removing exported globals, splitting/merging cells, changing `.py`/`.md` format, or adding side effects.
