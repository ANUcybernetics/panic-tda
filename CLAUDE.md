# PANIC-TDA Project Overview

## Project Description

PANIC-TDA is a Python tool for computing recursive text-to-image and
image-to-text model trajectories and analyzing them using
[topological data analysis](https://en.wikipedia.org/wiki/Topological_data_analysis).
It systematically explores how information flows through networks of generative
AI models by feeding outputs recursively back as inputs, creating "trajectories"
through semantic space.

## Architecture Overview

The project implements a three-stage computational pipeline:

1. **Runs Stage**: Execute networks of genAI models where outputs become inputs
   (@src/panic_tda/engine.py:114-200)
2. **Embeddings Stage**: Embed text outputs into high-dimensional semantic space
   (@src/panic_tda/engine.py:250-300)
3. **Persistence Diagram Stage**: Apply topological data analysis to
   characterize trajectory structure (@src/panic_tda/engine.py:350-400)

## Key Libraries

- **Ray** (@src/panic_tda/engine.py:9): Distributed computing framework for
  parallel execution
- **Pydantic/SQLModel** (@src/panic_tda/schemas.py): Data validation and ORM for
  SQLite persistence
- **PyTorch/Diffusers/Transformers**: ML frameworks for genAI and embedding
  models
- **Giotto-TDA/Giotto-PH** (@src/panic_tda/tda.py): Topological data analysis
  computations
- **Altair/Plotnine** (@src/panic_tda/datavis.py): Visualization libraries for
  results
- **Polars**: High-performance dataframe operations
- **Typer/Rich** (@src/panic_tda/main.py): CLI framework with enhanced output

## Development Philosophy

The project prioritizes:

- **Self-hosted execution**: Uses open-weight models to avoid cloud API costs
  (@DESIGN.md:44-49)
- **Reproducible research**: Complete pipeline from experiments to
  visualizations (@DESIGN.md:62-68)
- **Commodity hardware**: Optimized for single high-end GPU setups
  (@DESIGN.md:56-61)
- **Parallel computation**: Ray actors manage GPU resources efficiently
  (@src/panic_tda/engine.py:145-160)

## Core Components

### Models

- **GenAI Models** (@src/panic_tda/genai_models.py): Text-to-image and
  image-to-text model implementations
- **Embedding Models** (@src/panic_tda/embeddings.py): Semantic embedding model
  implementations

### Data Schema

- **ExperimentConfig** (@src/panic_tda/schemas.py:200-250): Experiment
  specification
- **Run** (@src/panic_tda/schemas.py:150-180): Single trajectory through a
  network
- **Invocation** (@src/panic_tda/schemas.py:100-140): Individual model inference
  event
- **Embedding** (@src/panic_tda/schemas.py:250-280): Semantic embedding result
- **PersistenceDiagram** (@src/panic_tda/schemas.py:300-350): TDA computation
  result

### CLI Commands

Main entry point is `panic-tda` (@src/panic_tda/main.py:79):

- `perform-experiment`: Run experiments from config file
- `experiment-status`: Check experiment progress
- `list-experiments`: View all experiments in database
- `export-video`: Generate mosaic visualizations
- `paper-charts`: Create publication-ready figures

## Testing Approach

The project uses pytest with comprehensive test coverage:

- **Unit tests** for all major components (@tests/)
- **Slow tests** marked for GPU-intensive operations (@pyproject.toml:57-61)
- **Benchmark tests** for performance measurement
- Run all tests: `uv run pytest`
- Run including slow tests: `uv run pytest -m slow`

## Development Workflow

1. Install with uv: `uv pip install -e .`
2. Create experiment config JSON (@README.md:100-109)
3. Run experiment: `uv run panic-tda perform-experiment config.json`
4. Monitor progress: `uv run panic-tda experiment-status`
5. Export results: `uv run panic-tda export-video <experiment-id>`

## Key Insights

The tool reveals patterns in how generative AI models transform information:

- Semantic stability and "stuck" states in trajectories
- Sensitivity to initial prompts and random seeds
- Emergent properties from model network interactions
- Quantifiable trajectory structure via persistent homology

For detailed design rationale, see @DESIGN.md.

## Warnings and Gotchas

- Don't ever run a clustering (e.g. `uv run panic-tda clustering-*`) command
  with output to stdout - it crashes claude code for some reason - redirect to a
  file

### SQLite3 CLI Crash Issue

**CRITICAL**: Claude Code crashes with `std::bad_alloc` when running `sqlite3`
CLI commands. This is a known issue (see
[GitHub #3660](https://github.com/anthropics/claude-code/issues/3660)).

**DO NOT** run sqlite3 commands at all - even redirecting output still causes
crashes:

```bash
# DON'T DO THIS - IT WILL CRASH:
sqlite3 ./db/trajectory_data.sqlite "SELECT * FROM table;"
# DON'T DO THIS EITHER - STILL CRASHES:
sqlite3 ./db/trajectory_data.sqlite "SELECT * FROM table;" > output.txt
```

**INSTEAD**, use one of these workarounds:

1. **Use Python's sqlite3 module**:

   ```python
   import sqlite3
   conn = sqlite3.connect('./db/trajectory_data.sqlite')
   for row in conn.execute("SELECT * FROM table LIMIT 10"):
       print(row)
   conn.close()
   ```

2. **Use the project's SQLModel infrastructure**:

   ```python
   from panic_tda.schemas import Run
   from panic_tda.engine import open_session

   with open_session("./db/trajectory_data.sqlite") as session:
       runs = session.query(Run).limit(10).all()
   ```

Also, **DO NOT** `tail` a file - this can causes crashes as well.
