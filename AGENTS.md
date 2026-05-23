# Instructions for AI agents

## Project

- Python 3.13
- Package manager: `uv`
- Dependencies: `uv sync`, `uv add <package>` etc.
- Directory structure:
  - 'playgrad/' - Playgrad visualization library (no training)
  - `examples/` - Runnable Python examples (fully contains training logic, each example in a separate subdirectory)
  - `tests/` - Tests for both examples and the playgrad library
  - `README.md` - How to run examples and the playgrad library API usage
  - `INTERNALS.md` - High level overview of playgrad library internals

## Code quality

- Consider moving files to subdirectories if a large number of files appear in `lib/` or `tests/`
- Proactively refactor clearly redundant or suboptimal code. Refactor big functions into smaller ones if reasonable.

## Type hints

- All function signatures must have type hints (parameters and return types).
- Variables whose type cannot be inferred from the during initialization must have type hints (e.g. `items: list[str] = []`).
- Do not annotate variables where the type can be inferred from the right-hand side.

## Testing

- Framework: pytest
- Keep tests reasonably fast: no sleeps, no many-batch neural network training, use small tensors etc.
- Use `pytest.mark.parametrize` for testing multiple inputs instead of duplicating test functions.

## Commit discipline

- Every user requested change should be accompanied by a commit. No need to ask for permission.
- Most commits should include corresponding test additions or changes. High level changes should incorporate documentation changes.
- Before committing: `uv run pytest && uv run ty check`
- Before committing: run the code. For UI testing, you can run some of the examples and use the playwright MCP.
- `README.md` and `INTERNALS.md` should be kept up to date

