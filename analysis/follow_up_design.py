#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars>=1.0",
#   "numpy>=2.0",
#   "matplotlib>=3.9",
# ]
# ///
"""What should the long follow-up run, and what will it cost? (TASK-104)

The panel left one question open: whether a prompt is ever forgotten. Its runs
stop at 150 text states with the two runs of a prompt still well short of the
distance between runs of different prompts. The follow-up is fewer prompts,
eight runs of each, and several times the length. This reads the panel for
what that design needs to know:

- cost: what one text state takes in each cell, from the panel's own
  timestamps, and so what a cell of 40 runs costs at a longer horizon
- prompts: each prompt's memory at the end of the panel, and the five that
  span it, one from each of the five groups the panel's prompts come in
- horizon: where the distance between twins would be at 700 and 1,000 text
  states if it stayed level, if it kept the logarithmic course it has held
  since state 20, or if it kept the rate of its last fifty states
- precision: how well eight runs fix a prompt's twin distance, from how much
  a single pair varies in the panel

    ./analysis/follow_up_design.py 01a09e21_parquet

Results -> analysis/follow_up_design.json, figure in analysis/follow_up_design/.
"""

import argparse
import itertools
import json
import pathlib
import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import drift_memory as dm

OUT = pathlib.Path(__file__).with_suffix(".json")
FIGURES = HERE / "follow_up_design"
PANEL_CONFIG = HERE.parent / "config" / "long_horizon_panel_4x4_300.json"
PD_COST = HERE / "pd_cost.json"

# The panel's config lists its twenty prompts in five blocks of four.
GROUPS = ("objects", "arrangements", "people", "events", "abstractions")
HORIZONS = (700, 1000)
# five prompts by eight runs is a cell of 40, the panel's batch shape
PROMPTS, RUNS_PER_PROMPT = 5, 8
# Flux2Dev costs ten times what the others do, so it is priced but not a
# candidate: the prompts and the pooled curves are read off the other twelve.
SLOW_GENERATOR = "Flux2Dev"
# The twin distance is fitted against log time from here on; before it the
# runs are still leaving the prompt together (`shared movement`).
FIT_FROM = 20
# The designs that were priced, their cells in running order. Each is the one
# before it with cells added: the shortest captioner with every affordable
# generator, then the longest, then the two between.
THREE = [
    ("Flux2Klein", "Moondream3"),
    ("SD35Medium", "Moondream3"),
    ("ZImageTurbo", "Moondream3"),
]
# the cheapest cell goes first, so the stages that run when a cell finishes
# are met at this length inside two days
SIX = [
    ("Flux2Klein", "Gemma4"),
    ("SD35Medium", "Moondream3"),
    ("ZImageTurbo", "Moondream3"),
    ("Flux2Klein", "Moondream3"),
    ("SD35Medium", "Gemma4"),
    ("ZImageTurbo", "Gemma4"),
]
DESIGNS = {
    "three cells": THREE,
    "six cells": SIX,
    "twelve cells": SIX
    + [
        (generator, captioner)
        for generator in ("Flux2Klein", "SD35Medium", "ZImageTurbo")
        for captioner in ("JoyCaption", "Qwen25VL")
    ],
}


def step_table(export_dir: pathlib.Path) -> pl.DataFrame:
    """One row per batch step: its cell, model and start and end times."""
    runs = pl.read_parquet(export_dir / "runs.parquet").select("id", "network")
    return (
        pl.read_parquet(
            export_dir / "invocations.parquet",
            columns=[
                "run_id",
                "sequence_number",
                "model",
                "started_at",
                "completed_at",
            ],
        )
        .join(runs, left_on="run_id", right_on="id")
        .with_columns(
            pl.col("started_at").str.to_datetime(time_zone="UTC"),
            pl.col("completed_at").str.to_datetime(time_zone="UTC"),
        )
        .group_by("network", "sequence_number")
        .agg(
            pl.col("model").first(),
            pl.col("started_at").min(),
            pl.col("completed_at").max(),
        )
        .sort("network", "sequence_number")
    )


def embedding_seconds(export_dir: pathlib.Path) -> dict[str, float]:
    """Per cell: seconds to embed one caption, from each run's embedding call."""
    runs = pl.read_parquet(export_dir / "runs.parquet").select("id", "network")
    invocations = pl.read_parquet(
        export_dir / "invocations.parquet", columns=["id", "run_id"]
    )
    per_run = (
        pl.read_parquet(
            export_dir / "embeddings.parquet",
            columns=["invocation_id", "started_at", "completed_at"],
        )
        .join(invocations, left_on="invocation_id", right_on="id")
        .join(runs, left_on="run_id", right_on="id")
        .with_columns(
            pl.col("started_at").str.to_datetime(time_zone="UTC"),
            pl.col("completed_at").str.to_datetime(time_zone="UTC"),
        )
        .group_by("network", "run_id")
        .agg(
            (pl.col("completed_at").max() - pl.col("started_at").min())
            .dt.total_seconds(fractional=True)
            .alias("seconds"),
            pl.len().alias("captions"),
        )
    )
    return dict(
        per_run.group_by("network")
        .agg((pl.col("seconds").sum() / pl.col("captions").sum()).alias("per_caption"))
        .iter_rows()
    )


def diagram_seconds(points: int) -> float:
    """A run's persistence diagram at this length: the slowest structured cloud timed."""
    clouds = json.loads(PD_COST.read_text())["clouds"]
    return max(
        row["seconds"]
        for name, rows in clouds.items()
        if name != "no structure"
        for row in rows
        if row["points"] == points
    )


def cost(export_dir: pathlib.Path) -> dict:
    """What one text state costs in each cell, and a cell at each horizon."""
    steps = step_table(export_dir).with_columns(
        (pl.col("completed_at") - pl.col("started_at"))
        .dt.total_seconds(fractional=True)
        .alias("seconds")
    )
    embed = embedding_seconds(export_dir)
    cells = {}
    for (network,), cell in steps.group_by(["network"], maintain_order=True):
        images = cell.filter(pl.col("sequence_number") % 2 == 0)
        captions = cell.filter(pl.col("sequence_number") % 2 == 1)
        # from the start of one image step to the start of the next: both
        # models, the swaps between them and the inserts
        pair = (
            images["started_at"].diff().dt.total_seconds(fractional=True).drop_nulls()
        )
        states = captions.height
        wall = (cell["completed_at"].max() - cell["started_at"].min()).total_seconds()
        cells[network] = {
            "text_states": states,
            "image_step_seconds": float(images["seconds"].median()),
            "caption_step_seconds": float(captions["seconds"].median()),
            "text_state_seconds": float(pair.median()),
            "wall_hours": wall / 3600,
            "uninterrupted_hours": states * float(pair.median()) / 3600,
            "embedding_seconds_per_caption": embed[network],
        }
    fast = [n for n in cells if SLOW_GENERATOR not in n]
    # restarts, retries and reboots, as the panel met them
    overhead = sum(cells[n]["wall_hours"] for n in fast) / sum(
        cells[n]["uninterrupted_hours"] for n in fast
    )
    runs_per_cell = PROMPTS * RUNS_PER_PROMPT
    for cell in cells.values():
        # embedding and diagrams, once the cell's runs are done
        cell["finishing_minutes_at"] = {
            str(h): runs_per_cell
            * (h * cell["embedding_seconds_per_caption"] + diagram_seconds(h))
            / 60
            for h in HORIZONS
        }
        cell["days_at"] = {
            str(h): h * cell["text_state_seconds"] * overhead / 86400
            + cell["finishing_minutes_at"][str(h)] / 1440
            for h in HORIZONS
        }

    def priced(design: list[tuple[str, str]]) -> dict:
        names = [json.dumps(list(cell), separators=(",", ":")) for cell in design]
        return {
            "cells": [dm.short(n) for n in names],
            "days_at": {
                str(h): sum(cells[n]["days_at"][str(h)] for n in names)
                for h in HORIZONS
            },
            "finished_after_days_at": {
                str(h): np.cumsum([cells[n]["days_at"][str(h)] for n in names]).tolist()
                for h in HORIZONS
            },
        }

    return {
        "what": "a cell is 40 runs in lockstep; days at a horizon are text states "
        "times the median wall clock of one, times the panel's overhead, plus "
        "embedding and diagrams once the runs are done",
        "overhead": overhead,
        "overhead_from": "wall clock over uninterrupted time, cells without "
        f"{SLOW_GENERATOR}",
        "cells": cells,
        "designs": {name: priced(design) for name, design in DESIGNS.items()},
    }


def prompt_groups(prompts: list[str]) -> dict[str, str]:
    listed = json.loads(PANEL_CONFIG.read_text())["prompts"]
    assert sorted(listed) == prompts, "the export's prompts are not the panel's"
    return {
        prompt: GROUPS[i // (len(listed) // len(GROUPS))]
        for i, prompt in enumerate(listed)
    }


def memory_by_prompt(all_stats: dict[str, dict], noise: dict[str, float]) -> np.ndarray:
    """(cell, prompt): memory at the end of the run, as `drift_memory.prompt_table`."""
    rows = []
    for name, stats in all_stats.items():
        twin = stats["twin"][:, -dm.END_BINS :].mean(axis=1)
        stranger = stats["stranger"][:, :, -dm.END_BINS :].mean(axis=2)
        np.fill_diagonal(stranger, np.nan)
        stranger = np.nanmean(stranger, axis=1)
        rows.append((stranger - twin) / (stranger - noise[name]))
    return np.stack(rows)


def smallest_gap(values: np.ndarray) -> float:
    return float(np.diff(np.sort(values)).min())


def choose_prompts(
    prompts: list[str], groups: dict[str, str], memory: np.ndarray, fast: np.ndarray
) -> dict:
    """One prompt a group, spread as evenly as the panel's memory range allows.

    The spread is judged twice, on the cells the follow-up can afford and on
    all sixteen, and a set is as good as the worse of the two: a prompt's mean
    over a dozen single pairs is uncertain by about 0.08, and a set that is
    evenly spread on both readings does not depend on which is taken.
    """
    everywhere, affordable = memory.mean(axis=0), memory[fast].mean(axis=0)
    by_group = [
        [i for i, p in enumerate(prompts) if groups[p] == group] for group in GROUPS
    ]

    def spread(chosen: tuple[int, ...]) -> float:
        idx = np.asarray(chosen)
        return min(smallest_gap(everywhere[idx]), smallest_gap(affordable[idx]))

    best = max(itertools.product(*by_group), key=spread)
    chosen = sorted(best, key=lambda i: -affordable[i])
    return {
        "rule": "one prompt from each group, maximising the smallest gap between "
        "neighbours in memory, on the twelve affordable cells and on all sixteen",
        "smallest_gap": spread(best),
        "chosen": [prompts[i] for i in chosen],
        "by_prompt": [
            {
                "prompt": prompts[i],
                "group": groups[prompts[i]],
                "memory_affordable_cells": float(affordable[i]),
                # each cell has one pair for the prompt, so the mean is uncertain
                "memory_affordable_cells_se": float(
                    memory[fast, i].std(ddof=1) / np.sqrt(fast.sum())
                ),
                "memory_all_cells": float(everywhere[i]),
                "chosen": bool(i in best),
            }
            for i in np.argsort(-affordable)
        ],
    }


def pooled(all_stats: dict[str, dict], names: list[str], idx: np.ndarray) -> dict:
    """Twin, stranger and noise over time, averaged over cells, for some prompts."""
    distinct = idx[:, None] != idx[None, :]
    return {
        "twin": np.mean(
            [all_stats[n]["twin"][idx].mean(axis=0) for n in names], axis=0
        ),
        "stranger": np.mean(
            [
                all_stats[n]["stranger"][np.ix_(idx, idx)][distinct].mean(axis=0)
                for n in names
            ],
            axis=0,
        ),
        "noise": np.mean(
            [all_stats[n]["floor"][idx].mean(axis=0) for n in names], axis=0
        ),
    }


def project(curves: dict[str, np.ndarray]) -> dict:
    """Where the twin distance goes from here, on three readings of the panel."""
    twin, stranger, noise = curves["twin"], curves["stranger"], curves["noise"]
    t = (np.arange(len(twin)) + 0.5) * dm.BIN
    fit = t >= FIT_FROM
    log_slope, log_at_one = np.polyfit(np.log(t[fit]), twin[fit], 1)
    line = np.polyfit(t[fit], twin[fit], 1)
    twin_end = float(twin[-dm.END_BINS :].mean())
    stranger_end = float(stranger[-dm.END_BINS :].mean())
    noise_end = float(noise[-dm.END_BINS :].mean())
    t_end = float(t[-dm.END_BINS :].mean())
    rate = float(np.polyfit(t[-dm.SLOPE_BINS :], twin[-dm.SLOPE_BINS :], 1)[0])

    def memory(distance: float) -> float:
        return (stranger_end - distance) / (stranger_end - noise_end)

    def at(horizon: int) -> dict[str, float]:
        return {
            "level": memory(twin_end),
            "logarithmic": memory(log_at_one + log_slope * np.log(horizon)),
            "last_fifty_rate": memory(twin_end + rate * (horizon - t_end)),
        }

    return {
        "twin_end": twin_end,
        "stranger_end": stranger_end,
        "noise_end": noise_end,
        "memory_start": float((stranger[0] - twin[0]) / (stranger[0] - noise[0])),
        "memory_end": memory(twin_end),
        "twin_rate_per_100": 100 * rate,
        # where twins would be as far apart as strangers, on each reading
        "forgotten_at": {
            "last_fifty_rate": t_end + (stranger_end - twin_end) / rate
            if rate > 0
            else None,
            "logarithmic": float(np.exp((stranger_end - log_at_one) / log_slope))
            if log_slope > 0
            else None,
        },
        "logarithmic_fit": {
            "from_text_state": FIT_FROM,
            "per_e_fold": float(log_slope),
            "per_doubling": float(log_slope * np.log(2)),
            "at_one": float(log_at_one),
            "rmse": float(
                np.sqrt(
                    np.mean((log_at_one + log_slope * np.log(t[fit]) - twin[fit]) ** 2)
                )
            ),
            "rmse_of_a_straight_line": float(
                np.sqrt(np.mean((np.polyval(line, t[fit]) - twin[fit]) ** 2))
            ),
        },
        "memory_at": {str(h): at(h) for h in HORIZONS},
    }


def precision(all_stats: dict[str, dict], names: list[str], gap: float) -> dict:
    """How well `RUNS_PER_PROMPT` runs fix one prompt's twin distance in one cell.

    The panel has one pair for each prompt and cell. What is left of its
    distance once the prompt's and the cell's means are taken out is that
    pair's own variation, plus whatever belongs to the prompt in that cell, so
    it bounds a single pair's standard deviation from above. The mean over all
    pairs of n runs is a U-statistic: its variance is that of one pair divided
    by the number of pairs if pairs vary independently, and by n/2 if all of
    the variation belongs to single runs. The truth is between the two.
    """
    twin = np.stack(
        [all_stats[n]["twin"][:, -dm.END_BINS :].mean(axis=1) for n in names]
    )
    residual = twin - twin.mean(axis=0) - twin.mean(axis=1, keepdims=True) + twin.mean()
    cells, prompts = twin.shape
    pair_sd = float(np.sqrt((residual**2).sum() / ((cells - 1) * (prompts - 1))))
    n = RUNS_PER_PROMPT
    pairs = n * (n - 1) // 2
    best = pair_sd / np.sqrt(pairs)
    worst = pair_sd * np.sqrt(2 / n)
    return {
        "single_pair_sd": pair_sd,
        "runs_per_prompt": n,
        "pairs_per_prompt": pairs,
        "twin_distance_se_one_prompt": [float(best), float(worst)],
        "memory_se_one_prompt": [float(best / gap), float(worst / gap)],
        "memory_se_cell_of_five_prompts": [
            float(best / gap / np.sqrt(PROMPTS)),
            float(worst / gap / np.sqrt(PROMPTS)),
        ],
        "memory_se_from": "the twin distance's standard error over the stranger "
        "distance less the noise",
    }


def figure(curves: dict[str, np.ndarray], projection: dict) -> None:
    twin, stranger = curves["twin"], curves["stranger"]
    t = (np.arange(len(twin)) + 0.5) * dm.BIN
    end, horizon = t[-1], max(HORIZONS)
    later = np.linspace(end, horizon, 200)
    fit = projection["logarithmic_fit"]
    rate = projection["twin_rate_per_100"] / 100
    courses = {
        "level": np.full_like(later, projection["twin_end"]),
        "logarithmic": fit["at_one"] + fit["per_e_fold"] * np.log(later),
        "last_fifty_rate": projection["twin_end"] + rate * (later - end),
    }
    labels = {
        "level": "stays level",
        "logarithmic": "keeps its logarithmic course",
        "last_fifty_rate": "keeps the rate of its last fifty states",
    }
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    fig.patch.set_facecolor(dm.SURFACE)
    dm.styled(ax)
    ax.plot(
        t, stranger, color=dm.ORANGE, linewidth=2, label="runs of different prompts"
    )
    ax.plot(
        [end, horizon],
        [projection["stranger_end"]] * 2,
        color=dm.ORANGE,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
    )
    ax.plot(t, twin, color=dm.BLUE, linewidth=2, label="runs of one prompt")
    ceiling = projection["stranger_end"]
    for name, course in courses.items():
        shown = course <= ceiling
        ax.plot(
            later[shown],
            course[shown],
            color=dm.BLUE,
            linewidth=1.2,
            linestyle=(0, (4, 3)),
        )
    ax.axhline(projection["noise_end"], color=dm.MUTED, linewidth=0.8)
    ax.axvspan(0, end + dm.BIN / 2, color=dm.GRID, alpha=0.45, linewidth=0)
    for h in HORIZONS:
        ax.axvline(h, color=dm.AXIS, linewidth=0.8)
        ax.text(h, 0.675, f"{h:,} ", color=dm.SECONDARY, fontsize=8, ha="right")
    ax.text(end / 2, 0.675, "the panel", color=dm.SECONDARY, fontsize=8, ha="center")
    memory = projection["memory_at"]
    readings = {
        name: ", ".join(
            f"{memory[str(h)][name]:.2f} at {h:,}"
            for h in HORIZONS
            if memory[str(h)][name] > 0
        )
        for name in courses
    }
    readings["level"] = f"{projection['memory_end']:.2f} throughout"
    met = projection["forgotten_at"]["last_fifty_rate"]
    if met is not None and met < horizon:
        readings["last_fifty_rate"] += f", none by {met:,.0f}"
    # each label sits clear of its own line: under the two that run to the
    # right-hand edge, above the one that climbs to meet the strangers
    notes = {
        "level": (horizon - 12, projection["twin_end"] - 0.012, "right", "top"),
        "logarithmic": (
            horizon - 12,
            fit["at_one"] + fit["per_e_fold"] * np.log(0.6 * horizon) - 0.012,
            "right",
            "top",
        ),
        "last_fifty_rate": (
            end + 30,
            projection["stranger_end"] - 0.015,
            "left",
            "top",
        ),
    }
    for name, (x, y, ha, va) in notes.items():
        ax.text(
            x,
            y,
            f"{labels[name]}: memory {readings[name]}",
            color=dm.SECONDARY,
            fontsize=8,
            ha=ha,
            va=va,
            # knocks the horizon rule out from behind the text
            bbox={"facecolor": dm.SURFACE, "edgecolor": "none", "pad": 1.5},
        )
    ax.text(
        horizon,
        projection["noise_end"] + 0.008,
        "seed noise in one step",
        color=dm.SECONDARY,
        fontsize=8,
        ha="right",
    )
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("text state", color=dm.SECONDARY, fontsize=9)
    ax.set_ylabel("cosine distance between captions", color=dm.SECONDARY, fontsize=9)
    ax.set_title(
        "Three courses the distance between a prompt's runs could take from here",
        loc="left",
        color=dm.INK,
        fontsize=10,
    )
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.1),
        frameon=False,
        fontsize=8,
        labelcolor=dm.SECONDARY,
    )
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "horizon.png", dpi=200, facecolor=dm.SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=pathlib.Path)
    args = parser.parse_args()

    cells, prompts, _ = dm.load_cells(args.export_dir)
    names = list(cells)
    all_stats = {n: dm.cell_statistics(x) for n, x in cells.items()}
    everything = np.arange(len(prompts))
    summaries = {n: dm.summarise(all_stats[n], everything) for n in names}
    is_fast = np.array([SLOW_GENERATOR not in n for n in names])
    fast = [n for n in names if SLOW_GENERATOR not in n]

    memory = memory_by_prompt(all_stats, {n: summaries[n]["noise"] for n in names})
    chosen = choose_prompts(prompts, prompt_groups(prompts), memory, is_fast)
    five = np.array([prompts.index(p) for p in chosen["chosen"]])

    curves = pooled(all_stats, fast, everything)
    projection = project(curves)
    gap = projection["stranger_end"] - projection["noise_end"]
    stranger = np.mean(
        [all_stats[n]["stranger"][:, :, -dm.END_BINS :].mean(axis=2) for n in fast],
        axis=0,
    )
    results = {
        "export": str(args.export_dir),
        "cost": cost(args.export_dir),
        "prompts": {
            **chosen,
            "distance_between_the_chosen_at_the_end": {
                f"{prompts[a]} / {prompts[b]}": float(stranger[a, b])
                for a, b in itertools.combinations(five, 2)
            },
        },
        "horizon": {
            "what": "memory at a horizon if the distance between twins stays level, "
            f"keeps the logarithmic course fitted from state {FIT_FROM}, or keeps "
            "the rate of the last fifty states; the stranger distance is held at "
            "its end value",
            "pooled_over_affordable_cells": projection,
            "the_chosen_prompts_pooled": project(pooled(all_stats, fast, five)),
            "by_cell": {
                n: {
                    **project(pooled(all_stats, [n], everything)),
                    "reached_at_70": summaries[n]["reached"],
                    "memory_end_chosen_prompts": dm.summarise(all_stats[n], five)[
                        "memory_end"
                    ],
                }
                for n in names
            },
        },
        "precision": precision(all_stats, fast, gap),
    }
    OUT.write_text(json.dumps(results, indent=2))
    figure(curves, projection)

    print(f"overhead {results['cost']['overhead']:.3f}")
    print(f"{'cell':<26}{'s/state':>9}{'days@700':>10}{'days@1000':>11}{'memory':>8}")
    for n in names:
        c = results["cost"]["cells"][n]
        print(
            f"{dm.short(n):<26}{c['text_state_seconds']:>9.0f}{c['days_at']['700']:>10.2f}"
            f"{c['days_at']['1000']:>11.2f}{summaries[n]['memory_end']:>8.2f}"
        )
    for row in chosen["by_prompt"]:
        print(
            f"{'*' if row['chosen'] else ' '} {row['prompt']:<62}{row['group']:<14}"
            f"{row['memory_affordable_cells']:>6.2f}{row['memory_all_cells']:>6.2f}"
        )
    print(json.dumps(projection["memory_at"], indent=1))


if __name__ == "__main__":
    main()
