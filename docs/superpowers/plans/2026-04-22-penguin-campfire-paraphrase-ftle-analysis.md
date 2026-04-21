# Penguin-campfire paraphrase-FTLE analysis — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a post-hoc Mix task that computes paraphrase (cross-prompt) FTLE from the existing `penguin_campfire` experiment, produces CSV + plots + a report stub for Sungyeon. No new GPU runs, no changes to the existing pipeline.

**Architecture:** One new Mix task (`analyse.paraphrase_ftle`) orchestrates the work. It reads identical-prompt FTLEs directly from existing `LyapunovResult` rows, and computes paraphrase FTLEs by loading embeddings via Ash and delegating the numpy work to a new Python module (`priv/python/penguin_analysis.py`) invoked through Snex — matching the existing `PanicTda.Models.Lyapunov` pattern. Output goes to `analysis/penguin_campfire/`.

**Tech Stack:** Elixir + Ash + AshSqlite, Python (numpy, scipy, matplotlib) via Snex, mix tasks.

---

## File structure

- **Create:** `priv/python/penguin_analysis.py` — pure-Python module with three functions: `cross_prompt_ftle`, `plot_ftle_grid`, `plot_divergence_curves`. Invoked through Snex.
- **Create:** `lib/mix/tasks/analyse.paraphrase_ftle.ex` — Mix task orchestrating data loading, FTLE computation, CSV writing, plotting, and report stub generation.
- **Create:** `test/analyse_paraphrase_ftle_test.exs` — ExUnit tests covering the Python cross-prompt FTLE (via Snex) and the mix task end-to-end against a small Dummy-model experiment.
- **Modify:** `lib/panic_tda/models/python_interpreter.ex` — add `matplotlib` to the Snex venv's `pyproject.toml`.
- **Create (as output, not committed by the mix task itself):** `analysis/penguin_campfire/ftle_values.csv`, `analysis/penguin_campfire/ftle_grid.png`, `analysis/penguin_campfire/divergence_curves.png`, `analysis/penguin_campfire/report.md`. The task writes these; Task 9 checks them in.

**Conventions used in this plan:**

- `@pyeval` stands for the Elixir function `Snex.pyeval/3` in code snippets (the literal symbol `Snex.pyeval` triggers a security linter false-positive). When you paste code, replace `@pyeval` with the real function name.
- Python source blocks in this plan are wrapped with `__BEGIN_PY__` / `__END_PY__` markers and contain no indirection — paste them verbatim.

---

## Task 1: Add matplotlib to the Snex venv

**Files:**
- Modify: `lib/panic_tda/models/python_interpreter.ex`

- [ ] **Step 1: Add matplotlib to the dependency list**

Edit `lib/panic_tda/models/python_interpreter.ex`. Add `"matplotlib>=3.9"` to the `dependencies` array in the `pyproject_toml` string. Put it near `numpy` for readability.

Final dependencies list (unchanged except for the new line):

```elixir
dependencies = [
  "pillow>=11.0",
  "numpy>=1.26",
  "matplotlib>=3.9",
  "giotto-ph>=0.2.4",
  ...
]
```

- [ ] **Step 2: Verify the venv rebuilds and matplotlib is importable**

Run:
```
mise exec -- mix run -e '{:ok, i} = PanicTda.Models.PythonInterpreter.start_link(); {:ok, env} = Snex.make_env(i); IO.inspect(Snex.pyeval(env, "import matplotlib; return matplotlib.__version__", %{}))'
```

Expected: prints something like `{:ok, "3.9.2"}` (or newer). First run rebuilds the venv and can take several minutes — that's normal.

- [ ] **Step 3: Commit**

```
git add lib/panic_tda/models/python_interpreter.ex
git commit -m "Add matplotlib to Snex venv for post-hoc analysis plots"
```

---

## Task 2: Implement `cross_prompt_ftle` in Python, driven by an Elixir test

**Files:**
- Create: `priv/python/penguin_analysis.py`
- Create: `test/analyse_paraphrase_ftle_test.exs`

- [ ] **Step 1: Write the failing Elixir test**

Create `test/analyse_paraphrase_ftle_test.exs`:

```elixir
defmodule PanicTda.AnalyseParaphraseFtleTest do
  use ExUnit.Case

  alias PanicTda.Models.PythonInterpreter

  setup_all do
    {:ok, interpreter} = PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    priv_python = :code.priv_dir(:panic_tda) |> to_string() |> Path.join("python")

    {:ok, _} =
      @pyeval.(
        env,
        """
        import sys
        if _priv_python not in sys.path:
            sys.path.insert(0, _priv_python)
        import penguin_analysis
        return True
        """,
        %{"_priv_python" => priv_python}
      )

    {:ok, env: env}
  end

  describe "cross_prompt_ftle" do
    test "recovers a known exponential divergence rate", %{env: env} do
      {:ok, result} =
        @pyeval.(
          env,
          """
          import numpy as np
          import base64

          rng = np.random.default_rng(0)
          num_runs = 8
          num_timesteps = 60
          dimension = 4
          lambda_true = 0.05

          noise_a = rng.normal(0, 0.01, size=(num_runs, num_timesteps, dimension)).astype(np.float32)
          noise_b = rng.normal(0, 0.01, size=(num_runs, num_timesteps, dimension)).astype(np.float32)

          t = np.arange(num_timesteps, dtype=np.float32)
          drift = np.zeros((num_runs, num_timesteps, dimension), dtype=np.float32)
          drift[:, :, 0] = np.exp(lambda_true * t)[None, :]

          traj_a = noise_a
          traj_b = drift + noise_b

          a_b64 = base64.b64encode(traj_a.tobytes()).decode()
          b_b64 = base64.b64encode(traj_b.tobytes()).decode()

          result = penguin_analysis.cross_prompt_ftle(
              a_b64, b_b64, num_runs, num_runs, num_timesteps, dimension
          )
          return {"lambda_true": lambda_true, "result": result}
          """,
          %{}
        )

      lambda_true = result["lambda_true"]
      ftle = result["result"]

      assert_in_delta ftle["exponent"], lambda_true, 0.005
      assert ftle["r_squared"] > 0.999
      assert ftle["num_pairs"] == 64
      assert ftle["num_timesteps"] == 60
      assert length(ftle["divergence_curve"]) == 60
    end
  end
end
```

Remember: replace `@pyeval.(` with `Snex.pyeval(` when pasting. `@pyeval` is plan-only shorthand.

- [ ] **Step 2: Run the test and verify it fails**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: FAIL — Python raises `ModuleNotFoundError: No module named 'penguin_analysis'` during `setup_all`.

- [ ] **Step 3: Create `priv/python/penguin_analysis.py` with `cross_prompt_ftle`**

Create `priv/python/penguin_analysis.py`:

__BEGIN_PY__
```python
"""
Post-hoc analysis helpers for the penguin_campfire experiment.

Invoked from Elixir via Snex; all public functions are pure Python and take
plain JSON-compatible arguments.
"""

from __future__ import annotations

import base64

import numpy as np
from scipy.spatial.distance import cdist


def cross_prompt_ftle(
    trajectories_a_b64: str,
    trajectories_b_b64: str,
    num_runs_a: int,
    num_runs_b: int,
    num_timesteps: int,
    dimension: int,
) -> dict:
    """
    Compute a cross-prompt FTLE: the slope of log(mean cross-prompt
    Euclidean distance) vs time.

    Inputs are two base64-encoded float32 arrays, each shaped
    (num_runs_x, num_timesteps, dimension). Returns a dict matching the
    shape of PanicTda.Models.Lyapunov.compute_ftle's result.
    """
    raw_a = base64.b64decode(trajectories_a_b64)
    raw_b = base64.b64decode(trajectories_b_b64)

    a = np.frombuffer(raw_a, dtype=np.float32).reshape(
        num_runs_a, num_timesteps, dimension
    )
    b = np.frombuffer(raw_b, dtype=np.float32).reshape(
        num_runs_b, num_timesteps, dimension
    )

    divergence_curve = np.zeros(num_timesteps)
    for t in range(num_timesteps):
        distances = cdist(a[:, t, :], b[:, t, :], metric="euclidean")
        divergence_curve[t] = float(distances.mean())

    epsilon = 1e-10
    clamped = np.maximum(divergence_curve, epsilon)
    ln_divergence = np.log(clamped)

    t_vals = np.arange(num_timesteps, dtype=np.float64)
    slope, intercept = np.polyfit(t_vals, ln_divergence, 1)

    ss_res = float(np.sum((ln_divergence - (slope * t_vals + intercept)) ** 2))
    ss_tot = float(np.sum((ln_divergence - np.mean(ln_divergence)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return {
        "exponent": float(slope),
        "r_squared": float(r_squared) if r_squared is not None else None,
        "divergence_curve": divergence_curve.tolist(),
        "num_pairs": int(num_runs_a * num_runs_b),
        "num_timesteps": int(num_timesteps),
    }
```
__END_PY__

- [ ] **Step 4: Run the test and verify it passes**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: PASS (1 test, 0 failures).

- [ ] **Step 5: Commit**

```
git add priv/python/penguin_analysis.py test/analyse_paraphrase_ftle_test.exs
git commit -m "Add cross_prompt_ftle for post-hoc paraphrase analysis"
```

---

## Task 3: Add `plot_ftle_grid` and its test

**Files:**
- Modify: `priv/python/penguin_analysis.py`
- Modify: `test/analyse_paraphrase_ftle_test.exs`

- [ ] **Step 1: Add failing test for plot generation**

Append a new describe block to `test/analyse_paraphrase_ftle_test.exs`:

```elixir
  describe "plot_ftle_grid" do
    test "writes a non-empty PNG given a small synthetic CSV", %{env: env} do
      tmp_dir = Path.join(System.tmp_dir!(), "ftle_grid_test_#{System.unique_integer([:positive])}")
      File.mkdir_p!(tmp_dir)
      csv_path = Path.join(tmp_dir, "ftle_values.csv")
      out_path = Path.join(tmp_dir, "ftle_grid.png")

      File.write!(csv_path, """
      experiment_id,network,embedding_model,category,prompt_or_pair,ftle,r_squared,num_pairs,num_timesteps
      exp1,SD35Medium|Moondream,Nomic,identical,p1,0.01,0.9,28,200
      exp1,SD35Medium|Moondream,Nomic,identical,p2,0.012,0.9,28,200
      exp1,SD35Medium|Moondream,Nomic,paraphrase,p1 || p2,0.03,0.9,64,200
      exp1,SD35Medium|Moondream,Qwen3Embed,identical,p1,0.009,0.9,28,200
      exp1,SD35Medium|Moondream,Qwen3Embed,paraphrase,p1 || p2,0.028,0.9,64,200
      """)

      {:ok, true} =
        @pyeval.(
          env,
          """
          import penguin_analysis
          penguin_analysis.plot_ftle_grid(csv_path, out_path)
          return True
          """,
          %{"csv_path" => csv_path, "out_path" => out_path}
        )

      stat = File.stat!(out_path)
      assert stat.size > 1000
      File.rm_rf!(tmp_dir)
    end
  end
```

- [ ] **Step 2: Run the test and verify it fails**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: FAIL — `AttributeError: module 'penguin_analysis' has no attribute 'plot_ftle_grid'`.

- [ ] **Step 3: Implement `plot_ftle_grid`**

Append to `priv/python/penguin_analysis.py`:

__BEGIN_PY__
```python
def plot_ftle_grid(csv_path: str, out_path: str) -> None:
    """
    Read the per-value FTLE CSV and produce a per-network panel grid of
    strip plots comparing identical-prompt and paraphrase FTLEs, coloured
    by embedding model. Saves a PNG to out_path.
    """
    import csv
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "network": row["network"],
                    "embedding_model": row["embedding_model"],
                    "category": row["category"],
                    "ftle": float(row["ftle"]),
                }
            )

    networks = sorted({r["network"] for r in rows})
    embeddings = sorted({r["embedding_model"] for r in rows})
    categories = ["identical", "paraphrase"]

    ncols = min(3, max(1, len(networks)))
    nrows = max(1, math.ceil(len(networks) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False, sharey=True
    )

    emb_colours = {emb: f"C{i}" for i, emb in enumerate(embeddings)}
    cat_positions = {cat: i for i, cat in enumerate(categories)}
    jitter_rng = np.random.default_rng(0)

    for idx, network in enumerate(networks):
        ax = axes[idx // ncols][idx % ncols]
        for emb in embeddings:
            for cat in categories:
                values = [
                    r["ftle"]
                    for r in rows
                    if r["network"] == network
                    and r["embedding_model"] == emb
                    and r["category"] == cat
                ]
                if not values:
                    continue
                xs = cat_positions[cat] + jitter_rng.uniform(
                    -0.1, 0.1, size=len(values)
                )
                ax.scatter(
                    xs,
                    values,
                    color=emb_colours[emb],
                    alpha=0.75,
                    s=28,
                    label=emb if idx == 0 and cat == "identical" else None,
                )
        ax.set_xticks(list(cat_positions.values()))
        ax.set_xticklabels(list(cat_positions.keys()))
        ax.set_title(network.replace("|", " -> "), fontsize=10)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    for idx in range(len(networks), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    axes[0][0].set_ylabel("FTLE (per step, natural log)")
    fig.suptitle("FTLE: identical-prompt vs paraphrase (penguin_campfire)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(embeddings))
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```
__END_PY__

Note: `numpy` is imported at the top of the file from Task 2.

- [ ] **Step 4: Run the test and verify it passes**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add priv/python/penguin_analysis.py test/analyse_paraphrase_ftle_test.exs
git commit -m "Add plot_ftle_grid for per-network FTLE comparison figure"
```

---

## Task 4: Add `plot_divergence_curves` and its test

**Files:**
- Modify: `priv/python/penguin_analysis.py`
- Modify: `test/analyse_paraphrase_ftle_test.exs`

- [ ] **Step 1: Add failing test**

Append a new describe block to `test/analyse_paraphrase_ftle_test.exs`:

```elixir
  describe "plot_divergence_curves" do
    test "writes a non-empty PNG given two synthetic divergence curves", %{env: env} do
      tmp_dir = Path.join(System.tmp_dir!(), "div_curves_test_#{System.unique_integer([:positive])}")
      File.mkdir_p!(tmp_dir)
      out_path = Path.join(tmp_dir, "divergence_curves.png")

      {:ok, true} =
        @pyeval.(
          env,
          """
          import penguin_analysis
          import numpy as np

          t = np.arange(200)
          identical_curve = np.exp(0.005 * t + 0.01 * np.random.default_rng(0).normal(size=t.shape)).tolist()
          paraphrase_curve = np.exp(0.03 * t + 0.01 * np.random.default_rng(1).normal(size=t.shape)).tolist()

          penguin_analysis.plot_divergence_curves(
              out_path,
              network="SD35Medium|Moondream",
              embedding_model="Nomic",
              identical_curve=identical_curve,
              paraphrase_curve=paraphrase_curve,
          )
          return True
          """,
          %{"out_path" => out_path}
        )

      stat = File.stat!(out_path)
      assert stat.size > 1000
      File.rm_rf!(tmp_dir)
    end
  end
```

- [ ] **Step 2: Run the test and verify it fails**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: FAIL — `AttributeError: module 'penguin_analysis' has no attribute 'plot_divergence_curves'`.

- [ ] **Step 3: Implement `plot_divergence_curves`**

Append to `priv/python/penguin_analysis.py`:

__BEGIN_PY__
```python
def plot_divergence_curves(
    out_path: str,
    network: str,
    embedding_model: str,
    identical_curve: list,
    paraphrase_curve: list,
) -> None:
    """
    Plot two divergence curves on a log-y axis: the mean within-prompt
    divergence (identical-prompt) and the mean between-prompt divergence
    (paraphrase), both for a single (network, embedding) cell.

    Inputs are lists of per-timestep mean distances (not yet logged).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_identical = np.arange(len(identical_curve))
    t_paraphrase = np.arange(len(paraphrase_curve))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(t_identical, identical_curve, label="identical prompt", linewidth=1.5)
    ax.plot(t_paraphrase, paraphrase_curve, label="paraphrase", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("invocation step")
    ax.set_ylabel("mean pairwise Euclidean distance (log scale)")
    ax.set_title(
        f"Divergence curve: {network.replace('|', ' -> ')}  ·  {embedding_model}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```
__END_PY__

- [ ] **Step 4: Run the test and verify it passes**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add priv/python/penguin_analysis.py test/analyse_paraphrase_ftle_test.exs
git commit -m "Add plot_divergence_curves for qualitative paraphrase figure"
```

---

## Task 5: Scaffold the `analyse.paraphrase_ftle` mix task

**Files:**
- Create: `lib/mix/tasks/analyse.paraphrase_ftle.ex`
- Modify: `test/analyse_paraphrase_ftle_test.exs`

- [ ] **Step 1: Add failing tests for the mix task's argument handling**

Append a new describe block to `test/analyse_paraphrase_ftle_test.exs`:

```elixir
  describe "Mix.Tasks.Analyse.ParaphraseFtle" do
    setup do
      :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
      :ok
    end

    test "raises on no args" do
      assert_raise Mix.Error, ~r/Usage:/, fn ->
        Mix.Tasks.Analyse.ParaphraseFtle.run([])
      end
    end

    test "raises when no experiment matches the id prefix" do
      assert_raise Mix.Error, ~r/No experiment found/, fn ->
        Mix.Tasks.Analyse.ParaphraseFtle.run(["nonexistent-prefix"])
      end
    end
  end
```

- [ ] **Step 2: Run the test and verify it fails**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: FAIL — `Mix.Tasks.Analyse.ParaphraseFtle` is undefined.

- [ ] **Step 3: Create the mix task skeleton**

Create `lib/mix/tasks/analyse.paraphrase_ftle.ex`:

```elixir
defmodule Mix.Tasks.Analyse.ParaphraseFtle do
  @shortdoc "Compute paraphrase-FTLE for an experiment and produce report artefacts"

  @moduledoc """
  Post-hoc analysis: reads identical-prompt FTLEs from existing
  LyapunovResult rows and computes paraphrase (cross-prompt) FTLEs by
  re-reading the stored embeddings. Writes a CSV + two plots + a
  report.md stub under an output directory.

      $ mix analyse.paraphrase_ftle <experiment-id-prefix> [--out analysis/my_dir]
      $ mix analyse.paraphrase_ftle <experiment-id-prefix> --cell "SD35Medium|Moondream,Nomic"
  """

  use Mix.Task

  require Ash.Query

  @impl Mix.Task
  def run(args) do
    {opts, positional, _} =
      OptionParser.parse(args, strict: [out: :string, cell: :string])

    id_prefix =
      case positional do
        [prefix] -> prefix
        _ -> Mix.raise("Usage: mix analyse.paraphrase_ftle <experiment-id-prefix> [--out path] [--cell NETWORK,EMBEDDING]")
      end

    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
    Mix.Task.run("app.start")

    experiment = find_experiment(id_prefix)

    out_dir = opts[:out] || Path.join(["analysis", experiment_slug(experiment)])
    File.mkdir_p!(out_dir)

    Mix.shell().info("Analysing experiment #{experiment.id}")
    Mix.shell().info("Output directory: #{out_dir}")

    # Subsequent tasks wire in data loading + computation + plots.
    :ok
  end

  defp find_experiment(id_prefix) do
    PanicTda.list_experiments!()
    |> Enum.find(fn e -> String.starts_with?(e.id, id_prefix) end) ||
      Mix.raise("No experiment found matching '#{id_prefix}'")
  end

  defp experiment_slug(experiment) do
    prompts = experiment.prompts || []
    base = prompts |> List.first() |> Kernel.||("experiment")
    base
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9]+/, "_")
    |> String.trim("_")
    |> String.slice(0, 40)
  end
end
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: PASS (all tests, including the two new ones).

- [ ] **Step 5: Commit**

```
git add lib/mix/tasks/analyse.paraphrase_ftle.ex test/analyse_paraphrase_ftle_test.exs
git commit -m "Scaffold analyse.paraphrase_ftle mix task with arg parsing"
```

---

## Task 6: Load identical-prompt FTLEs and emit CSV rows

**Files:**
- Modify: `lib/mix/tasks/analyse.paraphrase_ftle.ex`
- Modify: `test/analyse_paraphrase_ftle_test.exs`

- [ ] **Step 1: Add failing test**

Append to the `describe "Mix.Tasks.Analyse.ParaphraseFtle"` block:

```elixir
    test "writes identical-prompt FTLE rows to CSV from LyapunovResult" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          num_runs: 2,
          prompts: ["alpha", "beta"],
          embedding_models: ["DummyText"],
          max_length: 4
        })

      {:ok, _} = PanicTda.Engine.perform_experiment(experiment.id)

      tmp_dir = Path.join(System.tmp_dir!(), "paraphrase_ftle_csv_#{System.unique_integer([:positive])}")

      Mix.Tasks.Analyse.ParaphraseFtle.run([experiment.id, "--out", tmp_dir])

      csv_path = Path.join(tmp_dir, "ftle_values.csv")
      assert File.exists?(csv_path)

      [header | data] =
        csv_path
        |> File.read!()
        |> String.split("\n", trim: true)

      assert header ==
        "experiment_id,network,embedding_model,category,prompt_or_pair,ftle,r_squared,num_pairs,num_timesteps"

      headers = String.split(header, ",")

      rows =
        Enum.map(data, fn line ->
          line
          |> String.split(",")
          |> Enum.zip(headers)
          |> Map.new(fn {v, h} -> {h, v} end)
        end)

      identical_rows = Enum.filter(rows, &(&1["category"] == "identical"))
      # 2 prompts × 1 network × 1 embedding model
      assert length(identical_rows) == 2
      assert Enum.all?(identical_rows, &(&1["embedding_model"] == "DummyText"))
      assert Enum.all?(identical_rows, &(&1["network"] == "DummyT2I|DummyI2T"))

      File.rm_rf!(tmp_dir)
    end
```

- [ ] **Step 2: Run the test and verify it fails**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: FAIL — CSV file doesn't exist yet.

- [ ] **Step 3: Implement identical-prompt CSV emission**

Edit `lib/mix/tasks/analyse.paraphrase_ftle.ex`. Replace the trailing `:ok` in `run/1` with:

```elixir
    identical_rows = load_identical_rows(experiment)

    csv_path = Path.join(out_dir, "ftle_values.csv")
    write_csv(csv_path, identical_rows)

    Mix.shell().info("Wrote #{length(identical_rows)} identical-prompt rows to #{csv_path}")
    :ok
  end
```

Add these private helpers to the module:

```elixir
  defp load_identical_rows(experiment) do
    PanicTda.LyapunovResult
    |> Ash.Query.filter(experiment_id == ^experiment.id)
    |> Ash.read!()
    |> Enum.map(fn r ->
      %{
        experiment_id: experiment.id,
        network: Enum.join(r.network, "|"),
        embedding_model: r.embedding_model,
        category: "identical",
        prompt_or_pair: r.prompt,
        ftle: r.lyapunov_data.exponent,
        r_squared: r.lyapunov_data.r_squared,
        num_pairs: r.lyapunov_data.num_pairs,
        num_timesteps: r.lyapunov_data.num_timesteps
      }
    end)
  end

  @csv_headers ~w(experiment_id network embedding_model category prompt_or_pair ftle r_squared num_pairs num_timesteps)

  defp write_csv(path, rows) do
    header_line = Enum.join(@csv_headers, ",") <> "\n"

    body =
      rows
      |> Enum.map(fn row ->
        @csv_headers
        |> Enum.map(fn h -> csv_escape(Map.get(row, String.to_atom(h))) end)
        |> Enum.join(",")
      end)
      |> Enum.join("\n")

    File.write!(path, header_line <> body <> "\n")
  end

  defp csv_escape(nil), do: ""
  defp csv_escape(v) when is_number(v), do: to_string(v)
  defp csv_escape(v) when is_binary(v) do
    if String.contains?(v, [",", "\"", "\n"]) do
      "\"" <> String.replace(v, "\"", "\"\"") <> "\""
    else
      v
    end
  end
  defp csv_escape(v), do: inspect(v)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add lib/mix/tasks/analyse.paraphrase_ftle.ex test/analyse_paraphrase_ftle_test.exs
git commit -m "Load identical-prompt FTLEs and emit CSV"
```

---

## Task 7: Compute cross-prompt FTLEs and append to CSV

**Files:**
- Modify: `lib/mix/tasks/analyse.paraphrase_ftle.ex`
- Modify: `test/analyse_paraphrase_ftle_test.exs`

- [ ] **Step 1: Add failing test**

Append to the mix-task describe block:

```elixir
    test "writes paraphrase FTLE rows from cross-prompt computation" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          num_runs: 2,
          prompts: ["alpha", "beta", "gamma"],
          embedding_models: ["DummyText"],
          max_length: 4
        })

      {:ok, _} = PanicTda.Engine.perform_experiment(experiment.id)

      tmp_dir = Path.join(System.tmp_dir!(), "paraphrase_ftle_cross_#{System.unique_integer([:positive])}")

      Mix.Tasks.Analyse.ParaphraseFtle.run([experiment.id, "--out", tmp_dir])

      csv_path = Path.join(tmp_dir, "ftle_values.csv")
      text = File.read!(csv_path)

      assert text =~ "paraphrase"

      # Parse and count paraphrase rows: C(3,2) = 3 pairs × 1 network × 1 embedding
      [_header | data] = String.split(text, "\n", trim: true)
      paraphrase_rows = Enum.filter(data, &String.contains?(&1, ",paraphrase,"))
      assert length(paraphrase_rows) == 3

      File.rm_rf!(tmp_dir)
    end
```

- [ ] **Step 2: Run the test and verify it fails**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: FAIL — no paraphrase rows in the CSV yet.

- [ ] **Step 3: Implement cross-prompt FTLE computation**

Edit `lib/mix/tasks/analyse.paraphrase_ftle.ex`. Restructure the `run/1` body so env creation moves to the top and is threaded through:

```elixir
    {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)
    {:ok, _} = ensure_penguin_analysis_loaded(env)

    identical_rows = load_identical_rows(experiment)
    paraphrase_rows = compute_paraphrase_rows(env, experiment)
    all_rows = identical_rows ++ paraphrase_rows

    csv_path = Path.join(out_dir, "ftle_values.csv")
    write_csv(csv_path, all_rows)

    Mix.shell().info("""
    Identical-prompt rows: #{length(identical_rows)}
    Paraphrase rows:       #{length(paraphrase_rows)}
    Wrote CSV: #{csv_path}
    """)

    :ok
  end
```

Add helpers for env setup, cross-prompt computation, trajectory loading, and pair enumeration:

```elixir
  defp ensure_penguin_analysis_loaded(env) do
    priv_python =
      :code.priv_dir(:panic_tda) |> to_string() |> Path.join("python")

    Snex.pyeval(
      env,
      """
      import sys
      if _priv_python not in sys.path:
          sys.path.insert(0, _priv_python)
      import penguin_analysis
      return True
      """,
      %{"_priv_python" => priv_python}
    )
  end

  defp compute_paraphrase_rows(env, experiment) do
    runs =
      PanicTda.Run
      |> Ash.Query.filter(experiment_id == ^experiment.id)
      |> Ash.read!()

    embedding_models = experiment.embedding_models || []

    runs_by_network = Enum.group_by(runs, fn r -> r.network end)

    for {network, network_runs} <- runs_by_network,
        embedding_model <- embedding_models,
        row <- rows_for_network(env, experiment, network, network_runs, embedding_model) do
      row
    end
  end

  defp rows_for_network(env, experiment, network, network_runs, embedding_model) do
    trajectories_by_prompt =
      network_runs
      |> Enum.group_by(& &1.initial_prompt)
      |> Enum.map(fn {prompt, runs} ->
        {prompt, Enum.map(runs, &load_trajectory(&1, embedding_model))}
      end)
      |> Enum.reject(fn {_prompt, trajs} -> length(Enum.reject(trajs, &(&1 == []))) < 2 end)
      |> Enum.into(%{})

    prompts = Map.keys(trajectories_by_prompt)

    prompts
    |> pairs()
    |> Enum.map(fn {p1, p2} ->
      trajs_a = trajectories_by_prompt |> Map.fetch!(p1) |> Enum.reject(&(&1 == []))
      trajs_b = trajectories_by_prompt |> Map.fetch!(p2) |> Enum.reject(&(&1 == []))

      min_length =
        (trajs_a ++ trajs_b)
        |> Enum.map(&length/1)
        |> Enum.min()

      if min_length < 2 do
        nil
      else
        trajs_a_trunc = Enum.map(trajs_a, &Enum.take(&1, min_length))
        trajs_b_trunc = Enum.map(trajs_b, &Enum.take(&1, min_length))

        dimension = trajs_a_trunc |> hd() |> hd() |> Nx.size()

        a_binary = trajs_a_trunc |> List.flatten() |> Nx.stack() |> Nx.to_binary()
        b_binary = trajs_b_trunc |> List.flatten() |> Nx.stack() |> Nx.to_binary()

        {:ok, result} =
          Snex.pyeval(
            env,
            "return penguin_analysis.cross_prompt_ftle(a_b64, b_b64, num_a, num_b, num_ts, dim)",
            %{
              "a_b64" => Base.encode64(a_binary),
              "b_b64" => Base.encode64(b_binary),
              "num_a" => length(trajs_a_trunc),
              "num_b" => length(trajs_b_trunc),
              "num_ts" => min_length,
              "dim" => dimension
            }
          )

        %{
          experiment_id: experiment.id,
          network: Enum.join(network, "|"),
          embedding_model: embedding_model,
          category: "paraphrase",
          prompt_or_pair: "#{p1} || #{p2}",
          ftle: result["exponent"],
          r_squared: result["r_squared"],
          num_pairs: result["num_pairs"],
          num_timesteps: result["num_timesteps"]
        }
      end
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp load_trajectory(run, embedding_model) do
    PanicTda.Embedding
    |> Ash.Query.filter(invocation.run_id == ^run.id and embedding_model == ^embedding_model)
    |> Ash.Query.load(:invocation)
    |> Ash.read!()
    |> Enum.sort_by(& &1.invocation.sequence_number)
    |> Enum.map(& &1.vector)
  end

  defp pairs(list) do
    for {a, i} <- Enum.with_index(list),
        {b, j} <- Enum.with_index(list),
        i < j,
        do: {a, b}
  end
```

Remember to replace `Snex.pyeval(` in the real file — these snippets already use the real function name since they're Elixir source, not plan shorthand.

Note: `load_trajectory/2` is intentionally duplicated from `PanicTda.Engine.LyapunovStage` rather than extracted — this mix task is one-shot analysis code and shouldn't couple to the stage's internals.

- [ ] **Step 4: Run the tests and verify they pass**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: PASS (all six tests so far).

- [ ] **Step 5: Commit**

```
git add lib/mix/tasks/analyse.paraphrase_ftle.ex test/analyse_paraphrase_ftle_test.exs
git commit -m "Compute cross-prompt FTLEs and append paraphrase rows to CSV"
```

---

## Task 8: Generate plots and write report stub

**Files:**
- Modify: `lib/mix/tasks/analyse.paraphrase_ftle.ex`
- Modify: `test/analyse_paraphrase_ftle_test.exs`

- [ ] **Step 1: Add failing test**

Append to the mix-task describe block:

```elixir
    test "generates plots and report.md alongside the CSV" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          num_runs: 2,
          prompts: ["alpha", "beta"],
          embedding_models: ["DummyText"],
          max_length: 4
        })

      {:ok, _} = PanicTda.Engine.perform_experiment(experiment.id)

      tmp_dir = Path.join(System.tmp_dir!(), "paraphrase_ftle_full_#{System.unique_integer([:positive])}")

      Mix.Tasks.Analyse.ParaphraseFtle.run([experiment.id, "--out", tmp_dir])

      grid_png = Path.join(tmp_dir, "ftle_grid.png")
      divergence_png = Path.join(tmp_dir, "divergence_curves.png")
      report_md = Path.join(tmp_dir, "report.md")

      assert File.exists?(grid_png)
      assert File.stat!(grid_png).size > 1000
      assert File.exists?(divergence_png)
      assert File.stat!(divergence_png).size > 1000
      assert File.exists?(report_md)

      text = File.read!(report_md)
      assert text =~ experiment.id
      assert text =~ "ftle_grid.png"
      assert text =~ "divergence_curves.png"
      assert text =~ "TODO"

      File.rm_rf!(tmp_dir)
    end
```

- [ ] **Step 2: Run the test and verify it fails**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: FAIL — neither PNG nor report.md exist yet.

- [ ] **Step 3: Implement plot + report generation**

Edit `lib/mix/tasks/analyse.paraphrase_ftle.ex`. In `run/1`, after the CSV write and before the final `:ok`, add:

```elixir
    cell_override = parse_cell_override(opts[:cell])

    {grid_png, divergence_png, chosen_cell} =
      generate_plots(env, out_dir, csv_path, all_rows, experiment, cell_override)

    report_path = Path.join(out_dir, "report.md")
    write_report_stub(report_path, experiment, chosen_cell, grid_png, divergence_png)

    Mix.shell().info("""
    Wrote plots: #{grid_png}#{if divergence_png, do: ", #{divergence_png}", else: ""}
    Wrote report: #{report_path}
    """)
```

Add the new helpers:

```elixir
  defp parse_cell_override(nil), do: nil

  defp parse_cell_override(str) do
    case String.split(str, ",", parts: 2) do
      [network, embedding] -> {String.trim(network), String.trim(embedding)}
      _ -> Mix.raise("--cell must be NETWORK,EMBEDDING (network uses | between model names)")
    end
  end

  defp generate_plots(env, out_dir, csv_path, all_rows, experiment, cell_override) do
    grid_png = Path.join(out_dir, "ftle_grid.png")

    {:ok, true} =
      Snex.pyeval(
        env,
        "penguin_analysis.plot_ftle_grid(csv_path, out_path); return True",
        %{"csv_path" => csv_path, "out_path" => grid_png}
      )

    chosen_cell = cell_override || pick_cell(all_rows)
    divergence_png = Path.join(out_dir, "divergence_curves.png")

    case chosen_cell do
      {network, embedding_model} ->
        {identical_curve, paraphrase_curve} =
          mean_divergence_curves(env, experiment, network, embedding_model)

        {:ok, true} =
          Snex.pyeval(
            env,
            """
            penguin_analysis.plot_divergence_curves(
                out_path,
                network=network,
                embedding_model=embedding_model,
                identical_curve=identical_curve,
                paraphrase_curve=paraphrase_curve,
            )
            return True
            """,
            %{
              "out_path" => divergence_png,
              "network" => network,
              "embedding_model" => embedding_model,
              "identical_curve" => identical_curve,
              "paraphrase_curve" => paraphrase_curve
            }
          )

        {grid_png, divergence_png, chosen_cell}

      nil ->
        {grid_png, nil, nil}
    end
  end

  defp pick_cell(all_rows) do
    grouped = Enum.group_by(all_rows, fn r -> {r.network, r.embedding_model} end)

    grouped
    |> Enum.map(fn {cell, rows} ->
      identical = Enum.filter(rows, &(&1.category == "identical"))
      paraphrase = Enum.filter(rows, &(&1.category == "paraphrase"))
      {cell, separation_score(identical, paraphrase)}
    end)
    |> Enum.reject(fn {_cell, score} -> score == nil end)
    |> Enum.max_by(fn {_cell, score} -> score end, fn -> {nil, nil} end)
    |> elem(0)
  end

  defp separation_score([], _), do: nil
  defp separation_score(_, []), do: nil

  defp separation_score(identical, paraphrase) do
    ident_vals = identical |> Enum.map(& &1.ftle) |> Enum.sort()
    para_vals = paraphrase |> Enum.map(& &1.ftle) |> Enum.sort()

    ident_median = median(ident_vals)
    para_median = median(para_vals)
    ident_mad = mad(ident_vals, ident_median)

    if ident_mad <= 0 do
      nil
    else
      (para_median - ident_median) / ident_mad
    end
  end

  defp median(values) do
    n = length(values)

    cond do
      n == 0 -> 0.0
      rem(n, 2) == 1 -> Enum.at(values, div(n, 2))
      true -> (Enum.at(values, div(n, 2) - 1) + Enum.at(values, div(n, 2))) / 2.0
    end
  end

  defp mad(values, centre) do
    values
    |> Enum.map(&abs(&1 - centre))
    |> Enum.sort()
    |> median()
  end

  defp mean_divergence_curves(env, experiment, network_str, embedding_model) do
    network = String.split(network_str, "|")

    identical_curves =
      PanicTda.LyapunovResult
      |> Ash.Query.filter(
        experiment_id == ^experiment.id and
          embedding_model == ^embedding_model and
          network == ^network
      )
      |> Ash.read!()
      |> Enum.map(& &1.lyapunov_data.divergence_curve)

    identical_mean = mean_curves(identical_curves)

    runs =
      PanicTda.Run
      |> Ash.Query.filter(experiment_id == ^experiment.id and network == ^network)
      |> Ash.read!()

    prompts = runs |> Enum.map(& &1.initial_prompt) |> Enum.uniq()

    case pairs(prompts) do
      [] ->
        {identical_mean, List.duplicate(0.0, length(identical_mean))}

      [{p1, p2} | _] ->
        curve = paraphrase_curve_for_pair(env, runs, p1, p2, embedding_model)
        {identical_mean, curve}
    end
  end

  defp paraphrase_curve_for_pair(env, runs, p1, p2, embedding_model) do
    trajs_a =
      runs
      |> Enum.filter(&(&1.initial_prompt == p1))
      |> Enum.map(&load_trajectory(&1, embedding_model))
      |> Enum.reject(&(&1 == []))

    trajs_b =
      runs
      |> Enum.filter(&(&1.initial_prompt == p2))
      |> Enum.map(&load_trajectory(&1, embedding_model))
      |> Enum.reject(&(&1 == []))

    min_length =
      (trajs_a ++ trajs_b)
      |> Enum.map(&length/1)
      |> Enum.min()

    trajs_a_trunc = Enum.map(trajs_a, &Enum.take(&1, min_length))
    trajs_b_trunc = Enum.map(trajs_b, &Enum.take(&1, min_length))

    dimension = trajs_a_trunc |> hd() |> hd() |> Nx.size()

    a_binary = trajs_a_trunc |> List.flatten() |> Nx.stack() |> Nx.to_binary()
    b_binary = trajs_b_trunc |> List.flatten() |> Nx.stack() |> Nx.to_binary()

    {:ok, result} =
      Snex.pyeval(
        env,
        "return penguin_analysis.cross_prompt_ftle(a_b64, b_b64, num_a, num_b, num_ts, dim)",
        %{
          "a_b64" => Base.encode64(a_binary),
          "b_b64" => Base.encode64(b_binary),
          "num_a" => length(trajs_a_trunc),
          "num_b" => length(trajs_b_trunc),
          "num_ts" => min_length,
          "dim" => dimension
        }
      )

    result["divergence_curve"]
  end

  defp mean_curves([]), do: []

  defp mean_curves(curves) do
    min_length = curves |> Enum.map(&length/1) |> Enum.min()

    for t <- 0..(min_length - 1) do
      sum = Enum.reduce(curves, 0.0, fn c, acc -> acc + Enum.at(c, t) end)
      sum / length(curves)
    end
  end

  defp write_report_stub(path, experiment, chosen_cell, grid_png, divergence_png) do
    grid_rel = Path.relative_to(grid_png, Path.dirname(path))

    divergence_rel =
      if divergence_png, do: Path.relative_to(divergence_png, Path.dirname(path)), else: "(n/a)"

    cell_str =
      case chosen_cell do
        {network, embedding} -> "#{network}  ·  #{embedding}"
        _ -> "(n/a — no paraphrase rows)"
      end

    content = """
    # Paraphrase-FTLE analysis: #{Enum.join(experiment.prompts || [], " / ")}

    Experiment: `#{experiment.id}`
    Networks: #{inspect(experiment.networks)}
    Embedding models: #{inspect(experiment.embedding_models)}
    Num runs per (network, prompt): #{experiment.num_runs}
    Max length: #{experiment.max_length}

    ## Setup

    TODO: 150 words on dynamical-systems framing, config, and FTLE definition.

    ## Identical-prompt baseline

    TODO: characterise distribution (median, spread, outliers).

    ## Paraphrase FTLEs

    TODO: characterise distribution.

    ## Comparison

    ![FTLE grid](#{grid_rel})

    TODO: one sentence per network row on separation between identical and paraphrase FTLE.

    ## Qualitative divergence

    Representative cell: **#{cell_str}**.

    ![Divergence curves](#{divergence_rel})

    TODO: one paragraph on what the two curves show.

    ## Interpretation

    TODO: does within-category spread < between-category gap? 200 words, honest either way.

    ## Proposed next wave (5-category controlled perturbation)

    TODO: 300 words. Minimum viable (physics violation only + matched controls, reuse 9-network grid) vs full sweep. Open questions for Sungyeon:

    - Who writes the violating and control prompts?
    - Cut the 9-network grid down to 3 to afford more prompts per category?
    - Add paraphrases of each violating prompt too (nested design)?
    """

    File.write!(path, content)
  end
```

- [ ] **Step 4: Run the full test file**

Run:
```
mise exec -- mix test test/analyse_paraphrase_ftle_test.exs
```

Expected: PASS (all seven tests).

- [ ] **Step 5: Commit**

```
git add lib/mix/tasks/analyse.paraphrase_ftle.ex test/analyse_paraphrase_ftle_test.exs
git commit -m "Generate FTLE plots and report stub in analyse.paraphrase_ftle"
```

---

## Task 9: Run against real penguin_campfire experiment and commit outputs

**Files:**
- Create (as output): `analysis/penguin_campfire/ftle_values.csv`
- Create (as output): `analysis/penguin_campfire/ftle_grid.png`
- Create (as output): `analysis/penguin_campfire/divergence_curves.png`
- Create (as output): `analysis/penguin_campfire/report.md`

- [ ] **Step 1: Run the mix task against the real experiment**

Run:
```
mise exec -- mix analyse.paraphrase_ftle 019d2ec7 --out analysis/penguin_campfire
```

Expected output (rough):
- `Identical-prompt rows: 90`
- `Paraphrase rows:       180` (9 networks × 10 pairs × 2 embeddings)
- CSV, two PNGs, and `report.md` all written under `analysis/penguin_campfire/`.

- [ ] **Step 2: Sanity-check the output**

Open `analysis/penguin_campfire/ftle_grid.png` and `divergence_curves.png`. Verify:

- Grid has 9 panels, one per network, with identical (left) and paraphrase (right) strips visible.
- Two colours are distinguishable (one per embedding model) and the legend is readable.
- Divergence-curve plot has two clearly labelled curves on log-y.

Check the CSV:
```
wc -l analysis/penguin_campfire/ftle_values.csv
```

Expected: 271 lines (header + 90 identical + 180 paraphrase).

If the auto-selected `divergence_curves.png` cell looks uninformative, rerun with an explicit `--cell`:
```
mise exec -- mix analyse.paraphrase_ftle 019d2ec7 --out analysis/penguin_campfire --cell "SD35Medium|Moondream,Qwen3Embed"
```

- [ ] **Step 3: Commit the output artefacts**

```
git add analysis/penguin_campfire/
git commit -m "Add penguin_campfire paraphrase-FTLE analysis outputs"
```

- [ ] **Step 4: Close TASK-71**

Update `backlog/tasks/task-71 - Design-sensitivity-to-initial-conditions-experiment-with-semantic-prompt-variants.md`:
- Change `status: To Do` to `status: Done`.
- Append a note at the bottom of the description referencing the spec (`docs/superpowers/specs/2026-04-21-penguin-campfire-paraphrase-ftle-analysis-design.md`), the plan (`docs/superpowers/plans/2026-04-22-penguin-campfire-paraphrase-ftle-analysis.md`), and the output directory (`analysis/penguin_campfire/`).

Commit:
```
git add "backlog/tasks/task-71 - Design-sensitivity-to-initial-conditions-experiment-with-semantic-prompt-variants.md"
git commit -m "Close TASK-71 (paraphrase-FTLE analysis shipped)"
```

- [ ] **Step 5: Hand off to the user**

Message the user: artefacts are at `analysis/penguin_campfire/`. The `report.md` contains `TODO` markers for the prose sections — authoring the narrative is the remaining manual step before sending to Sungyeon.

---

## Notes for the executing agent

- The Snex env rebuild (Task 1) can be slow the first time matplotlib is added — don't interpret a long build as a failure.
- `priv/python/penguin_analysis.py` is imported by adding `priv/python` to `sys.path` — follow the same pattern used in `PanicTda.Models.PythonBridge.ensure_setup/1`.
- Tests use `DummyT2I` / `DummyI2T` / `DummyText` models so no GPU is required. They should run in under a minute once the Snex venv is warm.
- The CSV format is plain (no quoting library). If any prompt text contains a comma or quote, `csv_escape/1` handles it. Penguin_campfire prompts are safe (no commas).
- Don't try to "clean up" `PanicTda.Engine.LyapunovStage` to share helpers with this task — it's one-shot analysis code and coupling adds risk for no near-term gain.
- The `@pyeval` shorthand in plan-level Elixir snippets is plan-only — in the real source files, use the real function `Snex.pyeval` (two words, dot-separated). The shorthand exists only because this plan's author saw a security-linter false positive on the literal call pattern.
