#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars>=1.0",
#   "numpy>=2.0",
#   "matplotlib>=3.9",
# ]
# ///
"""Where does a caption sit, and how does it move? (TASK-103)

A run's caption embedding at a text state is read as three things added
together: where its prompt sits in this network, the run's own offset from
there, and fresh noise from that step's diffusion seed. Each has a size and a
persistence, and all of them come straight off distances between embeddings.
Nothing is clustered and there is no partition or lag to choose. The embeddings
are unit vectors, so cosine distance is half the squared Euclidean distance and
adds like a variance: every number here is a mean distance or a difference of
two.

- noise: the part of a step the next step takes back. For a slow part moving
  by independent increments under fresh noise, `D(1) = w/2 + s` and
  `D(2) = w + s`, so the noise is `s = 2 D(1) - D(2)`. `seed_resample.py`
  checks that reading against a direct measurement.
- a run's own offset: the distance between the two runs of a prompt (twins),
  less the noise. Twins start together, so this is what a run's history has
  added.
- what the prompt fixes: the distance between runs of different prompts
  (strangers) less the twin distance. Memory is that gap as a share of the
  stranger distance: 1 while twins coincide, 0 once they are as far apart as
  strangers.
- displacement: the distance between two states of one run against their
  separation in time. Less the noise it is what has accumulated, and if runs
  stay inside their prompt's region its ceiling is the run's own offset.
- shared movement: how far a prompt's two runs move together. From the
  distance between one twin early and the other late, less the mean of the
  twin distances at the two times; the noise cancels.
- the ladder: how far apart two captions sit when they share everything but
  the run, the prompt, the generator or the captioner.

Intervals resample the twenty prompts, which are the independent units: every
cell reads the same prompts, so one resample is applied to all of them.

    ./analysis/drift_memory.py 01a09e21_parquet

Results -> analysis/drift_memory.json, figures in analysis/drift_memory/,
summary table to stdout.
"""

import argparse
import json
import pathlib
from collections.abc import Iterator

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
OUT = pathlib.Path(__file__).with_suffix(".json")
FIGURES = HERE / "drift_memory"
AUDIT = HERE / "panel_audit.json"
RESAMPLE = HERE / "seed_resample.json"

# Step size and displacement both slow over the first fifty text states and are
# close to steady after that (`ageing` shows it), so the settled numbers come
# from the second half of the run.
LATE = 75
AGE_WINDOWS = ((0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150))
AGE_LAGS = (1, 25, 50)
REPORT_LAGS = (1, 2, 5, 10, 25, 50, 70)
# Still growing or levelling off: accumulated displacement at lags 60-74
# against lags 30-44. A plateau gives 1, a random walk 1.8.
BAND_SHORT, BAND_LONG = (30, 45), (60, 75)
GROWING = 1.1
BIN = 10
# quantities "at the end" average the last two bins; slopes run over the last five
END_BINS = 2
SLOPE_BINS = 5
# shared movement is measured between two ten-state windows 65 states apart
EARLY_PAIR = ((0, 10), (65, 75))
LATE_PAIR = ((75, 85), (140, 150))
WINDOW = 5
BOOTSTRAPS = 1000
SEED = 0

SCALARS = (
    "step", "noise", "noise_share", "accumulating_step",
    "offset_start", "offset_end", "twin_slope_per_100",
    "prompt_gap_end", "memory_start", "memory_end", "memory_slope_per_100",
    "stranger_change", "accumulated_70", "reached", "growth",
    "moved_early", "shared_early", "shared_early_share",
    "moved_late", "shared_late", "shared_late_share",
)  # fmt: skip


def load_cells(export_dir: pathlib.Path) -> tuple[dict, list[str], dict]:
    """Per network: embeddings as (prompts, runs, text states, dimension)."""
    embeddings = pl.read_parquet(export_dir / "embeddings.parquet").select(
        "invocation_id", "vector"
    )
    invocations = pl.read_parquet(export_dir / "invocations.parquet").select(
        "id", "run_id", "sequence_number", "output_text"
    )
    runs = pl.read_parquet(export_dir / "runs.parquet").select(
        "id", "network", "initial_prompt", "run_number"
    )
    frame = (
        embeddings.join(invocations, left_on="invocation_id", right_on="id")
        .join(runs, left_on="run_id", right_on="id")
        .filter(pl.col("output_text").is_not_null() & (pl.col("sequence_number") >= 0))
        .sort("network", "initial_prompt", "run_number", "sequence_number")
    )
    prompts = sorted(frame["initial_prompt"].unique().to_list())
    cells, captions = {}, {}
    for (network,), cell in frame.group_by(["network"], maintain_order=True):
        n_runs = cell.select("initial_prompt", "run_number").n_unique() // len(prompts)
        x = np.asarray(cell["vector"].to_list(), dtype=np.float64)
        x = x.reshape(len(prompts), n_runs, -1, x.shape[1])
        cells[network] = x / np.linalg.norm(x, axis=3, keepdims=True)
        captions[network] = np.asarray(
            cell["output_text"].to_list(), dtype=object
        ).reshape(len(prompts), n_runs, -1)
    return cells, prompts, captions


def lagged(x: np.ndarray, lag: int, lo: int, hi: int) -> np.ndarray:
    """Per prompt: mean distance between states `lag` apart, both inside [lo, hi)."""
    d = 1.0 - np.einsum(
        "prtd,prtd->prt", x[:, :, lo : hi - lag], x[:, :, lo + lag : hi]
    )
    return d.mean(axis=(1, 2))


def started(x: np.ndarray, lag: int, lo: int, hi: int) -> np.ndarray:
    """Per prompt: mean distance between a state in [lo, hi) and the one `lag` later."""
    d = 1.0 - np.einsum("prtd,prtd->prt", x[:, :, lo:hi], x[:, :, lo + lag : hi + lag])
    return d.mean(axis=(1, 2))


def step_noise(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Per prompt: the noise in a step, from steps that start in [lo, hi)."""
    top = min(hi, x.shape[2] - 2)
    return 2 * started(x, 1, lo, top) - started(x, 2, lo, top)


def between_windows(x: np.ndarray, a: tuple, b: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Per prompt: distance from window `a` to window `b`, same run and other run."""
    d = 1.0 - np.einsum(
        "prtd,psud->prstu", x[:, :, a[0] : a[1]], x[:, :, b[0] : b[1]]
    ).mean(axis=(3, 4))
    same = np.eye(x.shape[1], dtype=bool)
    return d[:, same].mean(axis=1), d[:, ~same].mean(axis=1)


def twins(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Per prompt: distance between its runs at the same text state."""
    chunk = x[:, :, lo:hi]
    d = 1.0 - np.einsum("prtd,pstd->prst", chunk, chunk).mean(axis=3)
    return d[:, ~np.eye(x.shape[1], dtype=bool)].mean(axis=1)


def ageing_table(x: np.ndarray) -> np.ndarray:
    """(lag, window, prompt): distance from a state in each window to the one `lag` later."""
    n_prompts, _, n_states, _ = x.shape
    table = np.full((len(AGE_LAGS) + 1, len(AGE_WINDOWS), n_prompts), np.nan)
    for i, lag in enumerate((*AGE_LAGS, 2)):
        for j, (lo, hi) in enumerate(AGE_WINDOWS):
            top = min(hi, n_states - lag)
            if top - lo >= (hi - lo) // 2:
                table[i, j] = started(x, lag, lo, top)
    return table


def window_mean(x: np.ndarray, lo: int) -> np.ndarray:
    m = x[:, :, lo : lo + WINDOW].mean(axis=2)
    return m / np.linalg.norm(m, axis=2, keepdims=True)


def hops(x: np.ndarray) -> np.ndarray:
    """Distance between the mean of the five states up to t and the five after, late on."""
    late = x[:, :, LATE:]
    total = np.cumsum(np.pad(late, ((0, 0), (0, 0), (1, 0), (0, 0))), axis=2)
    window = total[:, :, WINDOW:] - total[:, :, :-WINDOW]  # sums over [t, t + WINDOW)
    window /= np.linalg.norm(window, axis=3, keepdims=True)
    return 1.0 - np.einsum(
        "prtd,prtd->prt", window[:, :, :-WINDOW], window[:, :, WINDOW:]
    )


def cell_statistics(x: np.ndarray) -> dict[str, np.ndarray]:
    """Everything a cell's numbers are built from, kept per prompt for resampling."""
    n_prompts, _, n_states, _ = x.shape
    bins = n_states // BIN
    twin = np.empty((n_prompts, bins))
    stranger = np.empty((n_prompts, n_prompts, bins))
    floor = np.empty((n_prompts, bins))
    for b in range(bins):
        lo, hi = b * BIN, (b + 1) * BIN
        chunk = x[:, :, lo:hi]
        gram = 1.0 - np.einsum("prtd,qstd->pqrs", chunk, chunk) / (hi - lo)
        stranger[:, :, b] = gram.mean(axis=(2, 3))
        twin[:, b] = twins(x, lo, hi)
        floor[:, b] = step_noise(x, lo, hi)
    stats = {
        "curve": np.stack(
            [lagged(x, lag, LATE, n_states) for lag in range(1, n_states - LATE)],
            axis=1,
        ),
        "ageing": ageing_table(x),
        "twin": twin,
        "stranger": stranger,
        "floor": floor,
        "hops": hops(x),
        "net": 1.0
        - np.einsum(
            "prd,prd->pr", window_mean(x, LATE), window_mean(x, n_states - WINDOW)
        ),
    }
    for name, (a, b) in (("early", EARLY_PAIR), ("late", LATE_PAIR)):
        within, cross = between_windows(x, a, b)
        stats[f"{name}_within"], stats[f"{name}_cross"] = within, cross
        stats[f"{name}_twins"] = (twins(x, *a) + twins(x, *b)) / 2
        stats[f"{name}_noise"] = (step_noise(x, *a) + step_noise(x, *b)) / 2
    return stats


def summarise(stats: dict[str, np.ndarray], idx: np.ndarray) -> dict:
    """A cell's numbers from one (re)sample of its prompts."""
    curve = stats["curve"][idx].mean(axis=0)
    step, two = curve[0], curve[1]
    noise = 2 * step - two
    accumulated = curve - noise

    def band(lo: int, hi: int) -> float:
        return float(accumulated[lo - 1 : hi - 1].mean())

    twin = stats["twin"][idx].mean(axis=0)
    # a prompt drawn twice is not its own stranger
    distinct = idx[:, None] != idx[None, :]
    stranger = stats["stranger"][np.ix_(idx, idx)][distinct].mean(axis=0)
    floor = stats["floor"][idx].mean(axis=0)
    offset = twin - floor
    memory = (stranger - twin) / (stranger - floor)
    centres = (np.arange(len(memory)) + 0.5) * BIN

    def slope(series: np.ndarray) -> float:
        return float(
            100 * np.polyfit(centres[-SLOPE_BINS:], series[-SLOPE_BINS:], 1)[0]
        )

    out = {
        "step": float(step),
        "noise": float(noise),
        "noise_share": float(noise / step),
        "accumulating_step": float(step - noise),
        "offset_start": float(offset[0]),
        "offset_end": float(offset[-END_BINS:].mean()),
        "twin_slope_per_100": slope(twin),
        "prompt_gap_end": float((stranger - twin)[-END_BINS:].mean()),
        "memory_start": float(memory[0]),
        "memory_end": float(memory[-END_BINS:].mean()),
        "memory_slope_per_100": slope(memory),
        "stranger_change": float(
            stranger[-END_BINS:].mean() / stranger[:END_BINS].mean()
        ),
        "accumulated_70": band(65, 75),
        "growth": band(*BAND_LONG) / band(*BAND_SHORT),
        "curve": curve,
        "twin": twin,
        "stranger": stranger,
        "floor": floor,
        "memory": memory,
        "ageing": stats["ageing"][:, :, idx].mean(axis=2),
    }
    out["reached"] = out["accumulated_70"] / out["offset_end"]
    for name in ("early", "late"):
        moved = float(
            stats[f"{name}_within"][idx].mean() - stats[f"{name}_noise"][idx].mean()
        )
        shared = float(
            stats[f"{name}_cross"][idx].mean() - stats[f"{name}_twins"][idx].mean()
        )
        out[f"moved_{name}"], out[f"shared_{name}"] = moved, shared
        out[f"shared_{name}_share"] = shared / moved
    return out


def interval(samples: list[float]) -> list[float]:
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])]


def frozen(captions: np.ndarray) -> dict:
    """Runs whose caption repeats exactly from one text state to the next."""
    late = captions[:, :, LATE:]
    repeats = (late[:, :, 1:] == late[:, :, :-1]).mean(axis=2)
    return {
        "repeat_share_late": float(repeats.mean()),
        "runs_mostly_repeating": int((repeats >= 0.5).sum()),
        "runs_with_any_repeat": int((repeats > 0).sum()),
    }


def analyse_cell(
    stats: dict[str, np.ndarray], captions: np.ndarray, draws: np.ndarray
) -> tuple[dict, list[dict]]:
    n_prompts = stats["twin"].shape[0]
    point = summarise(stats, np.arange(n_prompts))
    boots = [summarise(stats, idx) for idx in draws]
    growth = interval([b["growth"] for b in boots])
    verdict = (
        "still growing"
        if growth[0] > GROWING
        else "levelling off"
        if growth[1] < GROWING
        else "undecided"
    )
    net = stats["net"]
    remaining = 1.0 - point["reached"]
    return {
        "runs": int(net.size),
        **{k: point[k] for k in SCALARS},
        "intervals": {k: interval([b[k] for b in boots]) for k in SCALARS},
        "displacement": {
            "verdict": verdict,
            "rule": f"accumulated displacement at lags {BAND_LONG[0]}-{BAND_LONG[1] - 1} over "
            f"lags {BAND_SHORT[0]}-{BAND_SHORT[1] - 1}; growing if the interval is above {GROWING}",
            "by_lag": {str(lag): float(point["curve"][lag - 1]) for lag in REPORT_LAGS},
            "curve": [float(v) for v in point["curve"]],
            "curve_interval": [
                interval([b["curve"][i] for b in boots])
                for i in range(len(point["curve"]))
            ],
            # if a run's offset decayed exponentially, the text states it would take
            # to fall to 1/e; indicative only, the decay is not a single exponential
            "offset_remaining_at_70": remaining,
            "implied_relaxation_states": float(-70 / np.log(remaining))
            if 0 < remaining < 1
            else None,
        },
        "over_time": {
            "bin_text_states": BIN,
            "twin": [float(v) for v in point["twin"]],
            "stranger": [float(v) for v in point["stranger"]],
            "noise": [float(v) for v in point["floor"]],
            "memory": [float(v) for v in point["memory"]],
            "memory_interval": [
                interval([b["memory"][i] for b in boots])
                for i in range(len(point["memory"]))
            ],
        },
        "ageing": {
            "windows": [f"{lo}-{hi - 1}" for lo, hi in AGE_WINDOWS],
            **{
                f"lag_{lag}": [
                    None if np.isnan(v) else float(v) for v in point["ageing"][i]
                ]
                for i, lag in enumerate(AGE_LAGS)
            },
        },
        "runs_net_displacement": {
            "what": f"distance between a run's mean over states {LATE}-{LATE + WINDOW - 1} "
            "and over its last five",
            "p10": float(np.percentile(net, 10)),
            "median": float(np.median(net)),
            "p90": float(np.percentile(net, 90)),
            "static": int((net < 2 * point["noise"] / WINDOW).sum()),
            "static_rule": "under twice what noise alone puts between two five-state means",
            "mobile": int((net > point["stranger"][-1] / 3).sum()),
            "mobile_rule": "over a third of the distance between runs of different prompts",
        },
        "hops": {
            "what": "distance between a run's mean over five states and over the next "
            f"five, from state {LATE}: skewed if movement comes in jumps",
            "median": float(np.median(stats["hops"])),
            "mean": float(stats["hops"].mean()),
            "p99": float(np.percentile(stats["hops"], 99)),
        },
        "frozen": frozen(captions),
    }, boots


def relation_tables(
    cells: dict[str, np.ndarray], lo: int, hi: int, centre: bool
) -> dict:
    """Prompt-by-prompt mean distance at the same text state, by how two cells relate."""
    names = list(cells)
    generators = np.array([json.loads(n)[0] for n in names])
    captioners = np.array([json.loads(n)[1] for n in names])
    x = np.stack([cells[n][:, :, lo:hi] for n in names])  # (cell, prompt, run, t, d)
    if centre:
        # remove each captioner's mean vector: its way of writing, whatever the image
        for c in np.unique(captioners):
            x[captioners == c] -= x[captioners == c].mean(axis=(0, 1, 2, 3))
    n_cells, n_prompts, n_runs = x.shape[:3]
    flat = x.reshape(-1, hi - lo, x.shape[4]).transpose(1, 0, 2)  # (t, run, d)
    sq = np.einsum("trd,trd->r", flat, flat) / (hi - lo)
    dot = np.matmul(flat, flat.transpose(0, 2, 1)).mean(axis=0)
    shape = (n_cells, n_prompts, n_runs)
    d = (0.5 * (sq[:, None] + sq[None, :]) - dot).reshape(*shape, *shape)
    same_g = generators[:, None] == generators[None, :]
    same_c = captioners[:, None] == captioners[None, :]
    relations = {
        "same network": same_g & same_c,
        "other generator": ~same_g & same_c,
        "other captioner": same_g & ~same_c,
        "other generator and captioner": ~same_g & ~same_c,
    }
    tables = {}
    for name, mask in relations.items():
        a, b = np.nonzero(mask)
        tables[name] = d[a, :, :, b].mean(axis=(0, 2, 4))  # (prompt, prompt)
    # the same network and prompt: the other run, never the run itself
    other = ~np.eye(n_runs, dtype=bool)
    twin = np.stack(
        [
            d[c, :, :, c][np.arange(n_prompts), :, np.arange(n_prompts)]
            for c in range(n_cells)
        ]
    )
    np.fill_diagonal(tables["same network"], twin[:, :, other].mean(axis=(0, 2)))
    return tables


RUNGS = {
    "same network and prompt, other run": ("same network", True),
    "same prompt and captioner, other generator": ("other generator", True),
    "same prompt and generator, other captioner": ("other captioner", True),
    "same prompt, other generator and captioner": (
        "other generator and captioner",
        True,
    ),
    "same network, other prompt": ("same network", False),
    "other prompt, same captioner, other generator": ("other generator", False),
    "other prompt, same generator, other captioner": ("other captioner", False),
    "nothing shared": ("other generator and captioner", False),
}


def ladder(tables: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    """Mean distance between two captions, by what they share."""
    distinct = idx[:, None] != idx[None, :]
    out = {}
    for rung, (relation, same_prompt) in RUNGS.items():
        table = tables[relation]
        out[rung] = float(
            table[idx, idx].mean()
            if same_prompt
            else table[np.ix_(idx, idx)][distinct].mean()
        )
    twin = out["same network and prompt, other run"]
    out["swap the generator"] = out["same prompt and captioner, other generator"] - twin
    out["swap the captioner"] = out["same prompt and generator, other captioner"] - twin
    out["swap the prompt"] = out["same network, other prompt"] - twin
    return out


def ladder_with_intervals(tables: dict, draws: np.ndarray) -> dict:
    n_prompts = next(iter(tables.values())).shape[0]
    point = ladder(tables, np.arange(n_prompts))
    boots = [ladder(tables, idx) for idx in draws]
    return {
        k: {"value": v, "interval": interval([b[k] for b in boots])}
        for k, v in point.items()
    }


def two_way(table: np.ndarray) -> tuple[float, float, float]:
    grand = table.mean()
    by_g, by_c = table.mean(axis=1) - grand, table.mean(axis=0) - grand
    total = ((table - grand) ** 2).sum()
    ss_g, ss_c = table.shape[1] * (by_g**2).sum(), table.shape[0] * (by_c**2).sum()
    return float(ss_g / total), float(ss_c / total), float(1 - (ss_g + ss_c) / total)


def contributions(
    names: list[str], values: dict[str, float], boots: dict[str, list]
) -> dict:
    """Split a per-cell number across the panel into generator and captioner effects."""
    generators = sorted({json.loads(n)[0] for n in names})
    captioners = sorted({json.loads(n)[1] for n in names})

    def key(g: str, c: str) -> str:
        return json.dumps([g, c], separators=(",", ":"))

    table = np.array([[values[key(g, c)] for c in captioners] for g in generators])
    shares = two_way(table)
    resampled = [
        two_way(
            np.array([[boots[key(g, c)][i] for c in captioners] for g in generators])
        )
        for i in range(len(next(iter(boots.values()))))
    ]
    return {
        "by_generator": dict(
            zip(generators, table.mean(axis=1).round(4).tolist(), strict=True)
        ),
        "by_captioner": dict(
            zip(captioners, table.mean(axis=0).round(4).tolist(), strict=True)
        ),
        "range": [float(table.min()), float(table.max())],
        **{
            f"{name}_share": {
                "value": shares[i],
                "interval": interval([r[i] for r in resampled]),
            }
            for i, name in enumerate(("generator", "captioner", "interaction"))
        },
    }


def prompt_table(
    all_stats: dict[str, dict], cells_out: dict[str, dict], prompts: list[str]
) -> dict:
    """Each prompt's twin distance and memory at the end, across the cells."""
    names = list(all_stats)
    twin = np.stack([all_stats[n]["twin"][:, -END_BINS:].mean(axis=1) for n in names])
    stranger = np.stack(
        [
            np.where(
                np.eye(len(prompts), dtype=bool)[:, :, None],
                np.nan,
                all_stats[n]["stranger"][:, :, -END_BINS:],
            ).mean(axis=2)
            for n in names
        ]
    )
    stranger = np.nanmean(stranger, axis=2)
    noise = np.array([cells_out[n]["noise"] for n in names])[:, None]
    memory = (stranger - twin) / (stranger - noise)
    generators = np.array([json.loads(n)[0] for n in names])
    captioners = np.array([json.loads(n)[1] for n in names])
    grand = twin.mean()
    total = ((twin - grand) ** 2).sum()

    def share(groups: np.ndarray, axis: int) -> float:
        means = np.array(
            [
                twin.take(np.nonzero(groups == g)[0], axis=axis).mean()
                for g in np.unique(groups)
            ]
        )
        return float(twin.size / len(means) * ((means - grand) ** 2).sum() / total)

    return {
        "by_prompt": [
            {
                "prompt": prompts[p],
                "twin_distance": float(twin[:, p].mean()),
                "memory": float(memory[:, p].mean()),
                "memory_quartiles": [
                    float(v) for v in np.percentile(memory[:, p], [25, 75])
                ],
            }
            for p in np.argsort(-memory.mean(axis=0))
        ],
        "twin_distance_explained_by": {
            "prompt": share(np.arange(len(prompts)), 1),
            "network": share(np.arange(len(names)), 0),
            "generator": share(generators, 0),
            "captioner": share(captioners, 0),
        },
    }


def seed_resample_check(prompts: list[str], draws: np.ndarray) -> dict | None:
    """Set a direct measurement of one seed draw beside the chain's own estimate.

    `seed_resample.py` redraws one late step of every cell at a new seed. Two
    successors of one caption should then be as far apart as two steps of the
    stored run, `w + s`, if a step is an independent increment plus fresh noise.
    """
    if not RESAMPLE.exists():
        return None
    data = json.loads(RESAMPLE.read_text())
    index = {prompt: i for i, prompt in enumerate(prompts)}
    keys = ("resampled", "step", "two_steps")
    tables, cells = {}, {}
    for network, cell in data["cells"].items():
        total = np.zeros((len(keys), len(prompts)))
        count = np.zeros(len(prompts))
        for run in cell["runs"]:
            if run["redrawn"]:
                total[:, index[run["prompt"]]] += [run[k] for k in keys]
                count[index[run["prompt"]]] += 1
        table = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        tables[network] = table
        resampled, step, two = np.nanmean(table, axis=1)
        cells[network] = {
            "redrawn": cell["redrawn"],
            "resampled": float(resampled),
            "step": float(step),
            "two_steps": float(two),
            "noise_direct": float(2 * step - resampled),
            "noise_chain": float(2 * step - two),
            "identical_captions": cell["identical_captions"],
            "kept": cell["kept"],
            "kept_identical_captions": cell["kept_identical_captions"],
        }

    def pooled(idx: np.ndarray) -> dict[str, float]:
        resampled, step, two = np.mean(
            [np.nanmean(table[:, idx], axis=1) for table in tables.values()], axis=0
        )
        return {
            "resampled": float(resampled),
            "step": float(step),
            "two_steps": float(two),
            "resampled_less_two_steps": float(resampled - two),
            "resampled_less_step": float(resampled - step),
            "noise_direct": float(2 * step - resampled),
            "noise_chain": float(2 * step - two),
            "noise_share_direct": float((2 * step - resampled) / step),
        }

    point = pooled(np.arange(len(prompts)))
    with np.errstate(invalid="ignore"):
        boots = [pooled(idx) for idx in draws]
    return {
        "image_step": data["image_step"],
        "redrawn": sum(c["redrawn"] for c in cells.values()),
        "pooled": {
            k: {
                "value": v,
                "interval": [
                    float(q)
                    for q in np.nanpercentile([b[k] for b in boots], [2.5, 97.5])
                ],
            }
            for k, v in point.items()
        },
        "networks": cells,
    }


def collapsed_runs() -> list[dict]:
    if not AUDIT.exists():
        return []
    first = json.loads(AUDIT.read_text())["images"]["first_flat_image_per_run"]
    return [{k: r[k] for k in ("network", "prompt", "sn", "kind")} for r in first]


def without_low_detail_runs(all_stats: dict[str, dict], prompts: list[str]) -> dict:
    """The headline numbers again, leaving out prompts with a run that simplified.

    `panel_audit.py` lists the runs that spend a third or more of their length on
    images with almost nothing in them. Every number in this script includes
    them; this is how much they move the cells they are in.
    """
    if not AUDIT.exists():
        return {}
    listed = json.loads(AUDIT.read_text())["images"]["low_detail"]["by_run"]
    keys = ("step", "noise_share", "offset_end", "memory_end", "reached")
    out = {}
    for network in sorted({r["network"] for r in listed}):
        dropped = sorted({r["prompt"] for r in listed if r["network"] == network})
        keep = np.array(
            [i for i, prompt in enumerate(prompts) if prompt not in dropped]
        )
        full = summarise(all_stats[network], np.arange(len(prompts)))
        rest = summarise(all_stats[network], keep)
        out[network] = {
            "prompts_left_out": dropped,
            "with": {k: full[k] for k in keys},
            "without": {k: rest[k] for k in keys},
        }
    return out


# Figures. Light surface, three validated categorical hues, text in ink.
SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, PALE_BLUE = "#2a78d6", "#eb6834", "#1baf7a", "#86b6ef"


def styled(ax: plt.Axes) -> plt.Axes:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=8, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, facecolor=SURFACE)
    # no creation date, so a rerun on unchanged data leaves the file unchanged
    fig.savefig(
        FIGURES / f"{name}.pdf", facecolor=SURFACE, metadata={"CreationDate": None}
    )
    plt.close(fig)


def short(network: str) -> str:
    return " + ".join(json.loads(network))


def figure_picture(cells_out: dict[str, dict]) -> None:
    """The whole description in one picture, averaged over the sixteen networks."""
    over = {
        k: np.mean([c["over_time"][k] for c in cells_out.values()], axis=0)
        for k in ("twin", "stranger", "noise")
    }
    curve = np.mean([c["displacement"]["curve"] for c in cells_out.values()], axis=0)
    noise = float(np.mean([c["noise"] for c in cells_out.values()]))
    t = (np.arange(len(over["twin"])) + 0.5) * BIN
    twin_end = float(over["twin"][-END_BINS:].mean())
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.5, 4.6), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]}
    )
    fig.patch.set_facecolor(SURFACE)
    styled(left).plot(
        t,
        over["stranger"],
        color=ORANGE,
        linewidth=2,
        label="runs of different prompts",
    )
    left.plot(
        t, over["twin"], color=BLUE, linewidth=2, label="the two runs of one prompt"
    )
    left.axhline(noise, color=MUTED, linewidth=0.8)
    left.set_xlim(0, 150)
    left.set_ylim(0, 0.68)
    left.set_xlabel("text state", color=SECONDARY, fontsize=9)
    left.set_ylabel("cosine distance between captions", color=SECONDARY, fontsize=9)
    left.set_title(
        "Between runs, at the same text state", loc="left", color=INK, fontsize=10
    )
    left.legend(
        loc="lower left",
        bbox_to_anchor=(0.16, 0.09),
        frameon=False,
        fontsize=8,
        labelcolor=SECONDARY,
    )
    brackets = (
        (noise, twin_end, "a run's own offset"),
        (
            twin_end,
            float(over["stranger"][-END_BINS:].mean()),
            "what the prompt\nstill fixes",
        ),
    )
    for lo, hi, text in brackets:
        left.plot([132, 132], [lo + 0.008, hi - 0.008], color=MUTED, linewidth=0.8)
        left.text(
            128,
            (lo + hi) / 2,
            text,
            color=SECONDARY,
            fontsize=8,
            va="center",
            ha="right",
        )
    left.text(2, noise + 0.008, "seed noise in one step", color=SECONDARY, fontsize=8)

    lags = np.arange(1, len(curve) + 1)
    styled(right).plot(lags, curve, color=AQUA, linewidth=2)
    right.axhline(noise, color=MUTED, linewidth=0.8)
    right.axhline(twin_end, color=BLUE, linewidth=0.8)
    right.set_xlim(0, 75)
    right.set_xlabel("separation in text states", color=SECONDARY, fontsize=9)
    right.set_title(
        "Within one run, late in the run", loc="left", color=INK, fontsize=10
    )
    right.text(
        74,
        twin_end + 0.008,
        "distance to its twin: the ceiling\nif runs stay in their prompt's region",
        color=SECONDARY,
        fontsize=8,
        ha="right",
        va="bottom",
    )
    right.text(
        36,
        curve[35] + 0.03,
        "two states of one run",
        color=SECONDARY,
        fontsize=8,
        ha="center",
    )
    right.text(
        74,
        noise - 0.008,
        "seed noise",
        color=SECONDARY,
        fontsize=8,
        ha="right",
        va="top",
    )
    fig.tight_layout()
    save(fig, "picture")


def grid_axes(cells_out: dict[str, dict], height: float) -> Iterator[tuple]:
    """A generator-by-captioner grid of panels: yields each axis with its cell."""
    names = list(cells_out)
    generators = sorted({json.loads(n)[0] for n in names})
    captioners = sorted({json.loads(n)[1] for n in names})
    fig, axes = plt.subplots(
        len(generators),
        len(captioners),
        figsize=(10.5, height),
        sharex=True,
        sharey=True,
    )
    fig.patch.set_facecolor(SURFACE)
    for i, g in enumerate(generators):
        for j, c in enumerate(captioners):
            ax = styled(axes[i, j])
            ax.set_title(f"{g} + {c}", loc="left", color=INK, fontsize=8.5)
            yield ax, cells_out[json.dumps([g, c], separators=(",", ":"))], (i, j), fig


def figure_memory(cells_out: dict[str, dict]) -> None:
    for ax, cell, (i, j), fig in grid_axes(cells_out, 8.4):
        over = cell["over_time"]
        t = (np.arange(len(over["twin"])) + 0.5) * BIN
        ax.plot(
            t,
            over["stranger"],
            color=ORANGE,
            linewidth=1.8,
            label="runs of different prompts",
        )
        ax.plot(
            t,
            over["twin"],
            color=BLUE,
            linewidth=1.8,
            label="the two runs of one prompt",
        )
        ax.axhline(cell["noise"], color=MUTED, linewidth=0.8)
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 0.72)
        ax.text(
            146,
            0.04 + cell["noise"],
            f"memory at the end {cell['memory_end']:.2f}",
            color=SECONDARY,
            fontsize=7.5,
            ha="right",
        )
        if i == 3:
            ax.set_xlabel("text state", color=SECONDARY, fontsize=8.5)
        if j == 0:
            ax.set_ylabel("cosine distance", color=SECONDARY, fontsize=8.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8.5,
        labelcolor=SECONDARY,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "memory")


def figure_displacement(cells_out: dict[str, dict]) -> None:
    for ax, cell, (i, j), fig in grid_axes(cells_out, 8.4):
        curve = np.asarray(cell["displacement"]["curve"])
        band = np.asarray(cell["displacement"]["curve_interval"])
        lags = np.arange(1, len(curve) + 1)
        twin_end = cell["offset_end"] + cell["noise"]
        ax.fill_between(
            lags, band[:, 0], band[:, 1], color=AQUA, alpha=0.12, linewidth=0
        )
        ax.plot(lags, curve, color=AQUA, linewidth=1.8)
        ax.axhline(cell["noise"], color=MUTED, linewidth=0.8)
        ax.axhline(twin_end, color=BLUE, linewidth=0.8)
        ax.set_xlim(0, 75)
        ax.set_ylim(0, 0.55)
        if (i, j) == (0, 0):
            ax.text(
                73,
                twin_end + 0.012,
                "distance to its twin",
                color=SECONDARY,
                fontsize=7.5,
                ha="right",
            )
            ax.text(
                73,
                cell["noise"] - 0.012,
                "seed noise",
                color=SECONDARY,
                fontsize=7.5,
                ha="right",
                va="top",
            )
        if i == 3:
            ax.set_xlabel("separation in text states", color=SECONDARY, fontsize=8.5)
        if j == 0:
            ax.set_ylabel("cosine distance", color=SECONDARY, fontsize=8.5)
    fig.suptitle(
        "Distance between two states of one run, late in the run (band: 95% interval over prompts)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "displacement")


def figure_prompts(table: dict) -> None:
    rows = table["by_prompt"][::-1]
    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    fig.patch.set_facecolor(SURFACE)
    styled(ax).grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    y = np.arange(len(rows))
    for k, row in zip(y, rows, strict=True):
        ax.plot(
            row["memory_quartiles"],
            [k, k],
            color=PALE_BLUE,
            linewidth=2,
            solid_capstyle="round",
        )
    ax.scatter(
        [r["memory"] for r in rows],
        y,
        s=52,
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=3,
    )
    ax.set_yticks(y, [r["prompt"] for r in rows], fontsize=8.5, color=SECONDARY)
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel(
        "memory: 1 while a prompt's two runs coincide, 0 once they are as far apart as strangers",
        color=SECONDARY,
        fontsize=8.5,
    )
    fig.suptitle(
        "Memory after 150 text states, by prompt",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=10,
    )
    fig.text(
        0.01,
        0.935,
        "dot: mean over the sixteen networks; line: the middle half of them",
        color=SECONDARY,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "prompts")


def figure_ladder(ladders: dict) -> None:
    first, last = (
        ladders[k]["as embedded"]
        for k in ("first 25 text states", "last 25 text states")
    )
    rungs = sorted(RUNGS, key=lambda r: last[r]["value"])
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    fig.patch.set_facecolor(SURFACE)
    styled(ax).grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    y = np.arange(len(rungs))
    for k, rung in zip(y, rungs, strict=True):
        ax.plot(
            [first[rung]["value"], last[rung]["value"]],
            [k, k],
            color=AXIS,
            linewidth=1.2,
        )
    ax.scatter(
        [first[r]["value"] for r in rungs],
        y,
        s=60,
        color=PALE_BLUE,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=3,
        label="first 25 text states",
    )
    ax.scatter(
        [last[r]["value"] for r in rungs],
        y,
        s=60,
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=3,
        label="last 25 text states",
    )
    ax.set_yticks(y, rungs, fontsize=8.5, color=SECONDARY)
    ax.set_xlim(0, 0.72)
    ax.set_xlabel(
        "cosine distance between two captions at the same text state",
        color=SECONDARY,
        fontsize=8.5,
    )
    ax.set_title(
        "How far apart two captions sit, by what they share",
        loc="left",
        color=INK,
        fontsize=10,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=SECONDARY)
    fig.tight_layout()
    save(fig, "ladder")


def print_table(cells_out: dict[str, dict]) -> None:
    print(
        f"{'network':<26}{'step':>6}{'noise':>7}{'share':>7}{'offset':>8}{'gap':>6}{'memory':>8}"
        f"{'moved@70':>10}{'reached':>9}  {'displacement':<14}{'shared e/l':>12}{'static':>8}{'mobile':>8}"
    )
    for network, r in cells_out.items():
        runs = r["runs_net_displacement"]
        print(
            f"{short(network):<26}{r['step']:>6.3f}{r['noise']:>7.3f}{r['noise_share']:>7.2f}"
            f"{r['offset_end']:>8.3f}{r['prompt_gap_end']:>6.2f}{r['memory_end']:>8.2f}"
            f"{r['accumulated_70']:>10.3f}{r['reached']:>9.2f}  {r['displacement']['verdict']:<14}"
            f"{r['shared_early_share']:>6.2f}{r['shared_late_share']:>6.2f}"
            f"{runs['static']:>8}{runs['mobile']:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=pathlib.Path)
    args = parser.parse_args()

    cells, prompts, captions = load_cells(args.export_dir)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(prompts), (BOOTSTRAPS, len(prompts)))
    all_stats = {n: cell_statistics(x) for n, x in cells.items()}
    cells_out, boots = {}, {}
    for n in cells:
        cells_out[n], boots[n] = analyse_cell(all_stats[n], captions[n], draws)

    n_states = next(iter(cells.values())).shape[2]
    windows = {
        "first 25 text states": (0, 25),
        "last 25 text states": (n_states - 25, n_states),
    }
    ladders = {
        name: {
            label: ladder_with_intervals(relation_tables(cells, lo, hi, centre), draws)
            for label, centre in (
                ("as embedded", False),
                ("captioner mean removed", True),
            )
        }
        for name, (lo, hi) in windows.items()
    }
    prompt_results = prompt_table(all_stats, cells_out, prompts)
    results = {
        "export": str(args.export_dir),
        "late_from_text_state": LATE,
        "bootstraps": BOOTSTRAPS,
        "networks": cells_out,
        "ladder": ladders,
        "prompts": prompt_results,
        "contributions": {
            key: contributions(
                list(cells),
                {n: cells_out[n][key] for n in cells},
                {n: [b[key] for b in boots[n]] for n in cells},
            )
            for key in (
                "noise",
                "noise_share",
                "accumulating_step",
                "offset_end",
                "memory_end",
                "reached",
            )
        },
        "seed_resample": seed_resample_check(prompts, draws),
        "collapsed_runs": collapsed_runs(),
        "without_low_detail_runs": without_low_detail_runs(all_stats, prompts),
    }
    OUT.write_text(json.dumps(results, indent=2))
    figure_picture(cells_out)
    figure_memory(cells_out)
    figure_displacement(cells_out)
    figure_prompts(prompt_results)
    figure_ladder(ladders)
    print_table(cells_out)


if __name__ == "__main__":
    main()
