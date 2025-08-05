# PANIC-TDA Project Overview

## Project Description

PANIC-TDA is a Python tool for computing recursive text-to-image and
image-to-text model trajectories and analyzing them using
[topological data analysis](https://en.wikipedia.org/wiki/Topological_data_analysis).
It systematically explores how information flows through networks of generative
AI models by feeding outputs recursively back as inputs, creating "trajectories"
through semantic space.

- use `uv` for _everything_
- use modern python: we support 3.12 and above
- use type hints
- to execute ad-hoc python code but in the correct context (including db
  session) add code to `script()` in @src/panic_tda/main.py, then run it with
  `uv run panic-tda script` (and remove the code when done)
- always do the above (use `script()` rather than `uv run python -c ...`)
- run the full test suite in parallel with `uv run pytest -n 8` which takes
  about 3 minutes (so set timeouts accordingly)
- don't use try/except for anything outside the top-level functions: it's fine
  for most functions to not handle any exceptions that occur
- use the logging module for logging (and don't overuse the INFO level - DEBUG
  is fine in most cases)
- use `sqlmodel` for database operations (models in @src/panic_tda/schema.py)
- when writing Alembic migrations that add constraints to existing tables,
  always use `batch_alter_table` context manager since SQLite doesn't support
  most ALTER TABLE operations directly (it will recreate the table with the new
  schema)

The project implements a three-stage computational pipeline (in
@src/panic_tda/engine.py):

1. **Runs Stage**: Execute networks of genAI models where outputs become inputs
2. **Embeddings Stage**: Embed text outputs into high-dimensional semantic space
3. **Persistence Diagram Stage**: Apply topological data analysis to
   characterize trajectory structure

For detailed design rationale, see @DESIGN.md.
