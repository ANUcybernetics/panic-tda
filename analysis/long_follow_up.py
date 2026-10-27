#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars>=1.0",
#   "numpy>=2.0",
#   "matplotlib>=3.9",
# ]
# ///
"""Is a prompt ever forgotten? The long follow-up read as the panel was (TASK-105)

The follow-up (experiment 01a10613) is six networks, five prompts, eight runs
of each and 1,000 text states per run. It is read with the quantities
`drift_memory.py` defined on the panel, over the whole horizon:

- twin distance (runs of one prompt at the same text state), stranger distance
  (runs of different prompts) and the noise in a step, in bins of ten states;
  memory is `(stranger - twin) / (stranger - noise)`
- displacement within a run, `D(k)`, from state LATE on and out to lag MAX_LAG,
  against the distance from one run at t to another run of its prompt at t + k,
  `C(k)`, over the same pairs of states. `C(k)` is the distance between twins
  at the same separation, so `(D(k) - noise) / (C(k) - noise)` is 1 once a
  run's own past says no more about where it is than its twin's does: the run
  has covered its prompt's region
- each prompt's centre, the mean of its eight runs, the spread of the runs
  around it, and whether the eight sit in one region or several. The test is
  SigClust's (Liu et al., JASA 2008): the best split of the eight run means into
  two groups, against the best split of eight draws from one Gaussian with the
  cell's pooled within-prompt covariance
- the panel's own numbers at 150 text states, set beside the follow-up's first
  150 for the six networks and five prompts the two share. Each is tested by
  shuffling each prompt's ten runs (the panel's two, the follow-up's eight)
  between the two experiments, which is what "the same process" predicts

Intervals resample runs within each prompt. The five prompts were chosen, not
drawn, so the intervals are conditional on them.

    ./analysis/long_follow_up.py 01a10613_parquet 01a09e21_parquet

Results -> analysis/long_follow_up.json, figures in analysis/long_follow_up/,
summary to stdout.
"""

import argparse
import itertools
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import drift_memory as panel
from drift_memory import (
    AQUA,
    BLUE,
    INK,
    MUTED,
    ORANGE,
    PALE_BLUE,
    SECONDARY,
    SURFACE,
    load_cells,
    short,
    styled,
)

HERE = pathlib.Path(__file__).parent
OUT = pathlib.Path(__file__).with_suffix(".json")
FIGURES = HERE / "long_follow_up"
AUDIT = HERE / "follow_up_audit.json"
PANEL_RESULTS = HERE / "drift_memory.json"

BIN = 10
# reporting blocks for the tables; curves keep the ten-state bins
BLOCK = 100
# Memory and twin distance have followed the logarithm of time (the design
# doc's "logarithmic course"), so their rates are per doubling of the run's
# length. The verdict sets the last span against the one before it.
DOUBLING_SPANS = {"early": (20, 150), "mid": (150, 400), "late": (400, 1000)}
# Movement slows over the panel's whole length (its `ageing`), so displacement
# here is taken from past the panel's horizon. Lag 500 is half the run.
LATE = 150
MAX_LAG = 500
REPORT_LAGS = (1, 2, 10, 25, 50, 70, 100, 150, 200, 300, 400, 500)
COVERED = 0.9
FIT_LAGS = (100, MAX_LAG + 1)
AGE_WINDOWS = tuple((lo, lo + BLOCK) for lo in range(0, 1000, BLOCK))
AGE_LAGS = (1, 2, 50, 100)
# a prompt's centre and spread are read from run means over windows of this size
CENTRE_WINDOW = 100
NULL_DRAWS = 2000
# Two run means closer than this are closer than two consecutive captions of
# one run (a step is 0.03-0.06), so a group inside it has gone to one place.
CORE = 0.05
PANEL_STATES = 150
PANEL_DRAWS = 1000
BOOTSTRAPS = 1000
SEED = 0


def statistics(x: np.ndarray) -> dict[str, np.ndarray]:
    """Everything the cell's numbers are built from, per run or per pair of runs."""
    n_prompts, n_runs, n_states, _ = x.shape
    bins = n_states // BIN
    x32 = x.astype(np.float32)
    same_time = np.empty((n_prompts, n_runs, n_runs, bins))
    within = np.empty((n_prompts, n_runs, MAX_LAG + 1))
    cross = np.empty((n_prompts, n_runs, n_runs, MAX_LAG + 1))
    for p in range(n_prompts):
        flat = x32[p].reshape(n_runs * n_states, -1)
        gram = (flat @ flat.T).reshape(n_runs, n_states, n_runs, n_states)
        t = np.arange(n_states)
        d = 1.0 - gram[:, t, :, t].astype(np.float64)  # (t, r, s)
        same_time[p] = (
            d.reshape(bins, BIN, n_runs, n_runs).mean(axis=1).transpose(1, 2, 0)
        )
        for k in range(1, MAX_LAG + 1):
            t = np.arange(LATE, n_states - k)
            d = 1.0 - gram[:, t, :, t + k].astype(np.float64).mean(axis=0)
            # both orders: run r at t against run s at t + k, and the reverse
            pair = (d + d.T) / 2
            within[p, :, k] = np.diag(pair)
            cross[p, :, :, k] = pair
    # strangers at the same text state, per pair of runs
    stranger = 1.0 - np.einsum("prtd,qstd->pqrst", x32, x32, optimize=True).astype(
        np.float64
    ).reshape(n_prompts, n_prompts, n_runs, n_runs, bins, BIN).mean(axis=5)
    # one- and two-step distances per run and bin, for the noise in each bin
    one = 1.0 - np.einsum("prtd,prtd->prt", x[:, :, :-1], x[:, :, 1:])
    two = 1.0 - np.einsum("prtd,prtd->prt", x[:, :, :-2], x[:, :, 2:])
    one = np.pad(one, ((0, 0), (0, 0), (0, 1)), constant_values=np.nan)
    two = np.pad(two, ((0, 0), (0, 0), (0, 2)), constant_values=np.nan)
    floor = 2 * np.nanmean(one.reshape(n_prompts, n_runs, bins, BIN), axis=3)
    floor -= np.nanmean(two.reshape(n_prompts, n_runs, bins, BIN), axis=3)
    return {
        "same_time": same_time,
        "stranger": stranger,
        "floor": floor,
        "within": within,
        "cross": cross,
    }


def counts_for(draws: np.ndarray, n_runs: int) -> np.ndarray:
    """(draw, prompt, run): how often each run is drawn."""
    return (draws[..., None] == np.arange(n_runs)).sum(axis=2).astype(np.float64)


def summarise(stats: dict[str, np.ndarray], counts: np.ndarray) -> dict:
    """A cell's curves from run weights; counts is (prompt, run)."""
    n_prompts, n_runs = counts.shape
    pair_w = counts[:, :, None] * counts[:, None, :]
    pair_w[:, np.arange(n_runs), np.arange(n_runs)] = 0  # a run is not its own twin
    run_w = counts / counts.sum(axis=1, keepdims=True)

    twin_p = (
        np.einsum("prs,prsn->pn", pair_w, stats["same_time"])
        / pair_w.sum(axis=(1, 2))[:, None]
    )
    cross_w = np.einsum("pr,qs->pqrs", counts, counts)
    stranger_pq = (
        np.einsum("pqrs,pqrsn->pqn", cross_w, stats["stranger"])
        / cross_w.sum(axis=(2, 3))[:, :, None]
    )
    off = ~np.eye(n_prompts, dtype=bool)
    stranger_p = np.stack(
        [stranger_pq[p, off[p]].mean(axis=0) for p in range(n_prompts)]
    )
    floor_p = np.einsum("pr,prn->pn", run_w, stats["floor"])
    within_p = np.einsum("pr,prk->pk", run_w, stats["within"])
    cross_p = (
        np.einsum("prs,prsk->pk", pair_w, stats["cross"])
        / pair_w.sum(axis=(1, 2))[:, None]
    )

    twin, floor = twin_p.mean(axis=0), floor_p.mean(axis=0)
    stranger = stranger_pq[off].mean(axis=0)
    within, cross = within_p.mean(axis=0), cross_p.mean(axis=0)
    noise = 2 * within[1] - within[2]
    return {
        "twin": twin,
        "stranger": stranger,
        "floor": floor,
        "memory": (stranger - twin) / (stranger - floor),
        "memory_by_prompt": (stranger_p - twin_p) / (stranger_p - floor_p),
        "twin_by_prompt": twin_p,
        "displacement": within,
        "twin_at_lag": cross,
        "noise": noise,
        "reached": (within - noise) / (cross - noise),
    }


def blocks(series: np.ndarray) -> np.ndarray:
    return series.reshape(*series.shape[:-1], -1, BLOCK // BIN).mean(axis=-1)


def per_doubling(series: np.ndarray, lo: int, hi: int) -> float:
    """Change in a ten-state series per doubling of the run's length, between states lo and hi."""
    centres = (np.arange(len(series)) + 0.5) * BIN
    keep = (centres >= lo) & (centres < hi)
    return float(np.polyfit(np.log2(centres[keep]), series[keep], 1)[0])


def scalars(s: dict) -> dict[str, float]:
    """The numbers the verdicts rest on, from one (re)sample."""
    reached = s["reached"]
    over = np.nonzero(reached[1:] >= COVERED)[0]
    # reached against the doubling of the separation, over lags FIT_LAGS
    lags = np.arange(*FIT_LAGS)
    rate, at = np.polyfit(np.log2(lags / MAX_LAG), reached[lags], 1)
    out = {
        "memory_start": float(s["memory"][0]),
        "memory_150": float(s["memory"][13:15].mean()),
        "memory_end": float(s["memory"][-BLOCK // BIN :].mean()),
        "stranger_change": float(
            s["stranger"][-BLOCK // BIN :].mean() / s["stranger"][:2].mean()
        ),
        "noise": float(s["noise"]),
        "noise_share": float(s["noise"] / s["displacement"][1]),
        **{f"reached_{k}": float(reached[k]) for k in (70, 150, 300, 500)},
        # first lag at which a run is COVERED of the way to its twin; MAX_LAG + 1 if never
        "covered_lag": float(over[0] + 1 if over.size else MAX_LAG + 1),
        "reached_per_doubling": float(rate),
        # where the logarithmic course of `reached` would cross COVERED
        "covered_lag_projected": float(MAX_LAG * 2 ** ((COVERED - at) / rate))
        if rate > 0
        else float("inf"),
        # accumulated displacement at lags 400-500 over lags 200-300: 1 is a plateau
        "late_growth": float(
            (s["displacement"][400:501] - s["noise"]).mean()
            / (s["displacement"][200:301] - s["noise"]).mean()
        ),
    }
    for name, (lo, hi) in DOUBLING_SPANS.items():
        out[f"memory_per_doubling_{name}"] = per_doubling(s["memory"], lo, hi)
        out[f"twin_per_doubling_{name}"] = per_doubling(s["twin"], lo, hi)
    out["memory_slowing"] = (
        out["memory_per_doubling_late"] - out["memory_per_doubling_mid"]
    )
    return out


def interval(samples) -> list[float]:
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])]


def memory_verdict(point: dict, iv: dict) -> tuple[str, str]:
    """Forgotten, still falling, levelled off, or not yet told apart, from the late rate."""
    late, slowing = iv["memory_per_doubling_late"], iv["memory_slowing"]
    rate = point["memory_per_doubling_late"]
    span = "{}-{}".format(*DOUBLING_SPANS["late"])
    if iv["memory_end"][0] <= 0:
        return (
            "forgotten",
            f"memory over the last {BLOCK} states has an interval reaching zero",
        )
    if late[1] < 0:
        zero = 1000 * 2 ** (point["memory_end"] / -rate)
        return (
            "keeps falling",
            (
                f"memory falls {-rate:.3f} per doubling over states {span} (interval "
                f"{late[0]:.3f} to {late[1]:.3f}); on that course it reaches zero "
                f"near state {zero:,.0f}"
            ),
        )
    if slowing[0] > 0:
        return (
            "levels off above zero",
            f"the rate per doubling over states {span} ({late[0]:.3f} to {late[1]:.3f}) includes "
            "none and is slower than the cell's own rate over states "
            "{}-{}".format(*DOUBLING_SPANS["mid"]),
        )
    return (
        "cannot yet be told",
        f"the rate per doubling over states {span} ({late[0]:.3f} to {late[1]:.3f}) includes "
        "none, and also the rate the cell fell at over states "
        "{}-{}".format(*DOUBLING_SPANS["mid"]),
    )


def analyse_cell(stats: dict, draws: np.ndarray) -> tuple[dict, dict]:
    n_prompts, n_runs = stats["floor"].shape[:2]
    point = summarise(stats, np.ones((n_prompts, n_runs)))
    boot_counts = counts_for(draws, n_runs)
    boots = [summarise(stats, c) for c in boot_counts]
    sc = scalars(point)
    boot_sc = [scalars(b) for b in boots]
    iv = {k: interval([b[k] for b in boot_sc]) for k in sc}
    verdict, reason = memory_verdict(sc, iv)
    covered = [b["covered_lag"] for b in boot_sc]
    out = {
        **sc,
        "intervals": iv,
        "memory_verdict": verdict,
        "memory_reason": reason,
        "covered": {
            "rule": f"first lag at which displacement less the noise is {COVERED:.0%} of the "
            "distance to a twin at the same lag, less the noise",
            "lag": None if sc["covered_lag"] > MAX_LAG else int(sc["covered_lag"]),
            "lower_bound": int(sc["covered_lag"])
            if sc["covered_lag"] > MAX_LAG
            else None,
            "share_of_resamples_covered_by_max_lag": float(
                np.mean(np.array(covered) <= MAX_LAG)
            ),
            "interval": interval(covered),
        },
        "over_time": {
            "bin_text_states": BIN,
            **{
                k: [float(v) for v in point[k]]
                for k in ("twin", "stranger", "floor", "memory")
            },
            "memory_interval": [
                interval([b["memory"][i] for b in boots])
                for i in range(len(point["memory"]))
            ],
            "twin_interval": [
                interval([b["twin"][i] for b in boots])
                for i in range(len(point["twin"]))
            ],
            "stranger_interval": [
                interval([b["stranger"][i] for b in boots])
                for i in range(len(point["stranger"]))
            ],
        },
        "by_block": {
            "block_text_states": BLOCK,
            **{
                k: {
                    "value": [float(v) for v in blocks(point[k])],
                    "interval": [
                        interval(col)
                        for col in np.array([blocks(b[k]) for b in boots]).T
                    ],
                }
                for k in ("twin", "stranger", "floor", "memory")
            },
        },
        "displacement": {
            "late_from_text_state": LATE,
            "by_lag": {
                str(k): {
                    "within": float(point["displacement"][k]),
                    "twin_at_lag": float(point["twin_at_lag"][k]),
                    "reached": float(point["reached"][k]),
                    "reached_interval": interval([b["reached"][k] for b in boots]),
                }
                for k in REPORT_LAGS
            },
            "within": [float(v) for v in point["displacement"]],
            "twin_at_lag": [float(v) for v in point["twin_at_lag"]],
            "reached": [float(v) for v in point["reached"]],
            "reached_interval": [
                interval([b["reached"][k] for b in boots]) for k in range(MAX_LAG + 1)
            ],
        },
    }
    return out, {"point": point, "boots": boots}


def ageing(x: np.ndarray) -> dict:
    out = {"windows": [f"{lo}-{hi - 1}" for lo, hi in AGE_WINDOWS]}
    for lag in AGE_LAGS:
        out[f"lag_{lag}"] = [
            float(panel.started(x, lag, lo, min(hi, x.shape[2] - lag)).mean())
            if hi - lag > lo
            else None
            for lo, hi in AGE_WINDOWS
        ]
    return out


def splits(n: int) -> np.ndarray:
    """Every way of putting n points into two groups of at least two, as 0/1 rows."""
    rows = [
        np.isin(np.arange(n), a)
        for size in range(2, n // 2 + 1)
        for a in itertools.combinations(range(n), size)
    ]
    return np.array(rows, dtype=np.float64)


def cluster_index(
    points: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Two-means cluster index, within SS over total SS, at the best split of each set.

    points is (set, n, d); returns the index and the best split's row in masks.
    """
    centred = points - points.mean(axis=1, keepdims=True)
    gram = centred @ centred.transpose(0, 2, 1)
    total = np.einsum("sii->s", gram)
    within = np.zeros((len(points), len(masks)))
    for m in (masks, 1 - masks):
        # SS of a group about its own mean: trace less the sum over its size
        size = m.sum(axis=1)
        within += np.einsum("ki,sii->sk", m, gram)
        within -= np.einsum("ki,sij,kj->sk", m, gram, m) / size
    best = within.argmin(axis=1)
    return within[np.arange(len(points)), best] / total, best


def one_region_test(
    m: np.ndarray, pooled: np.ndarray, rng: np.random.Generator
) -> dict:
    """SigClust: is the best two-group split of these points better than one Gaussian gives?

    The Gaussian's covariance is the cell's pooled within-prompt covariance of
    run means (`pooled`, one row per run, each centred on its prompt). Eight
    points cannot estimate a 256-dimensional covariance of their own, and
    their sample spectrum makes the test conservative to the point of uselessness.
    A prompt whose runs do split into regions inflates the pooled covariance
    along that split, which errs the same way.
    """
    masks = splits(len(m))
    (index,), (best,) = cluster_index(m[None], masks)
    dof = len(pooled) - pooled.shape[0] // len(m)
    draws = (
        rng.standard_normal((NULL_DRAWS, len(m), len(pooled))) @ pooled / np.sqrt(dof)
    )
    null, _ = cluster_index(draws, masks)
    group = masks[best].astype(bool)
    d = 0.5 * ((m[:, None] - m[None, :]) ** 2).sum(axis=2)

    def within(g: np.ndarray) -> float:
        sub = d[np.ix_(g, g)]
        return float(sub[~np.eye(len(sub), dtype=bool)].mean())

    p = float((null <= index).mean())
    inside = [within(group), within(~group)]
    between = float(d[np.ix_(group, ~group)].mean())
    tight = [g for g, w in zip((group, ~group), inside, strict=True) if w < CORE]
    if p >= 0.05:
        structure = "one region"
    elif between < CORE:
        structure = "one tight place"
    elif tight:
        structure = f"a core of {int(tight[0].sum())} and the rest"
    else:
        structure = "two regions"
    return {
        "cluster_index": float(index),
        "p": p,
        "structure": structure,
        "groups": [np.nonzero(group)[0].tolist(), np.nonzero(~group)[0].tolist()],
        "within_group_distance": inside,
        "between_group_distance": between,
    }


def centres(
    x: np.ndarray, captions: np.ndarray, prompts: list[str], rng: np.random.Generator
) -> dict:
    """Each prompt's centre, the spread of its runs about it, and whether they form one region."""
    n_prompts, n_runs, n_states, _ = x.shape
    windows = [(lo, lo + CENTRE_WINDOW) for lo in range(0, n_states, CENTRE_WINDOW)]
    middle = len(windows) // 2 - 1
    # run means over each window: (window, prompt, run, d)
    means = np.stack([x[:, :, lo:hi].mean(axis=2) for lo, hi in windows])
    centre = means.mean(axis=2)  # (window, prompt, d)
    # half squared Euclidean, the scale cosine distance is on
    spread = 0.5 * ((means - centre[:, :, None]) ** 2).sum(
        axis=3
    )  # (window, prompt, run)
    between = 0.5 * ((centre[:, :, None] - centre[:, None, :]) ** 2).sum(axis=3)
    pooled = (means - centre[:, :, None]).reshape(len(windows), n_prompts * n_runs, -1)
    off = ~np.eye(n_prompts, dtype=bool)
    out = {}
    for p, prompt in enumerate(prompts):
        last = x[p, :, -CENTRE_WINDOW:].reshape(-1, x.shape[3])
        nearest = int(np.argmax(last @ centre[-1, p]))
        r, t = divmod(nearest, CENTRE_WINDOW)
        # how far the centre moved from the middle window to the last, and how far
        # it would seem to move from sampling eight runs alone
        step = means[-1, p] - means[middle, p]
        moved = 0.5 * float((step.mean(axis=0) ** 2).sum())
        floor = (
            0.5
            * float(((step - step.mean(axis=0)) ** 2).sum())
            / (n_runs * (n_runs - 1))
        )
        out[prompt] = {
            "windows": [f"{lo}-{hi - 1}" for lo, hi in windows],
            "spread": [float(v) for v in spread[:, p].mean(axis=1)],
            "spread_by_run_end": [float(v) for v in spread[-1, p]],
            "centre_length": [float(v) for v in np.linalg.norm(centre[:, p], axis=1)],
            "centre_moved_from_first_window": [
                float(0.5 * ((c - centre[0, p]) ** 2).sum()) for c in centre[:, p]
            ],
            "centre_moved_middle_to_end": moved,
            "centre_moved_middle_to_end_sampling_alone": floor,
            "nearest_other_centre_end": float(between[-1, p][off[p]].min()),
            "caption_nearest_centre_end": {
                "run": int(r),
                "text_state": int(n_states - CENTRE_WINDOW + t),
                "caption": str(captions[p, r, n_states - CENTRE_WINDOW + t]),
            },
            "one_region": {
                "windows": [list(windows[middle]), list(windows[-1])],
                "middle": one_region_test(means[middle, p], pooled[middle], rng),
                "end": one_region_test(means[-1, p], pooled[-1], rng),
            },
        }
    return out


def panel_comparison(
    follow: dict[str, np.ndarray],
    panel_cells: dict[str, np.ndarray],
    follow_prompts: list[str],
    panel_prompts: list[str],
    published: dict,
    rng: np.random.Generator,
) -> dict:
    """The follow-up's first 150 text states beside the panel's, on the shared networks and prompts."""
    keep = [panel_prompts.index(p) for p in follow_prompts]
    keys = (
        "step",
        "noise",
        "offset_end",
        "prompt_gap_end",
        "memory_start",
        "memory_end",
        "reached",
    )
    out = {}
    for network, x in follow.items():
        y = panel_cells[network][keep]
        n_prompts, n_runs = x.shape[:2]
        mine = panel.summarise(
            panel.cell_statistics(x[:, :, :PANEL_STATES]), np.arange(n_prompts)
        )
        theirs = panel.summarise(panel.cell_statistics(y), np.arange(n_prompts))
        # Is the panel's pair of runs per prompt exchangeable with the follow-up's
        # eight? Shuffle each prompt's ten runs between the two and recompute both.
        pool = np.concatenate([x[:, :, :PANEL_STATES], y], axis=1)
        observed = {k: theirs[k] - mine[k] for k in keys}
        shuffled = {k: [] for k in keys}
        for _ in range(PANEL_DRAWS):
            order = np.argsort(rng.random(pool.shape[:2]), axis=1)
            mixed = pool[np.arange(n_prompts)[:, None], order]
            a = panel.summarise(
                panel.cell_statistics(mixed[:, n_runs:]), np.arange(n_prompts)
            )
            b = panel.summarise(
                panel.cell_statistics(mixed[:, :n_runs]), np.arange(n_prompts)
            )
            for k in keys:
                shuffled[k].append(a[k] - b[k])
        # each prompt's twin distance at the end: the panel's one pair against the follow-up's 28
        end = slice(PANEL_STATES - 20, PANEL_STATES)
        pairs = []
        for p, prompt in enumerate(follow_prompts):
            chunk = x[p, :, end]
            d = 1.0 - np.einsum("rtd,std->rst", chunk, chunk).mean(axis=2)
            pair_values = d[np.triu_indices(n_runs, 1)]
            theirs_pair = float(panel.twins(y[p : p + 1], end.start, end.stop)[0])
            pairs.append(
                {
                    "prompt": prompt,
                    "panel": theirs_pair,
                    "follow_up_mean": float(pair_values.mean()),
                    "follow_up_pairs_range": [
                        float(pair_values.min()),
                        float(pair_values.max()),
                    ],
                    "panel_percentile": float((pair_values < theirs_pair).mean()),
                }
            )
        out[network] = {
            "panel": {k: float(theirs[k]) for k in keys},
            "panel_published_memory_end": published[network]["memory_end"],
            "follow_up": {k: float(mine[k]) for k in keys},
            "panel_less_follow_up": observed,
            "permutation_p": {
                k: float(np.mean(np.abs(shuffled[k]) >= abs(observed[k]))) for k in keys
            },
            "twin_end_by_prompt": pairs,
        }
    return out


def flat_runs(prompts: list[str], run_ids: dict) -> dict[str, list[tuple[int, int]]]:
    """(prompt, run) of every run the audit found reaching a flat field, by network."""
    if not AUDIT.exists():
        return {}
    first = json.loads(AUDIT.read_text())["images"]["first_flat_image_per_run"]
    out: dict[str, list] = {}
    for r in first:
        out.setdefault(r["network"], []).append(run_ids[r["run_id"]])
    return out


def run_index(
    export_dir: pathlib.Path, prompts: list[str]
) -> dict[str, tuple[int, int]]:
    import polars as pl

    runs = pl.read_parquet(export_dir / "runs.parquet").sort(
        "network", "initial_prompt", "run_number"
    )
    out = {}
    for (_,), cell in runs.group_by(["network"], maintain_order=True):
        for p, prompt in enumerate(prompts):
            ids = cell.filter(pl.col("initial_prompt") == prompt)["id"].to_list()
            out.update({rid: (p, i) for i, rid in enumerate(ids)})
    return out


def without_flat(
    stats: dict, flat: list[tuple[int, int]], shape: tuple[int, int]
) -> dict:
    counts = np.ones(shape)
    for p, r in flat:
        counts[p, r] = 0
    point = summarise(stats, np.ones(shape))
    rest = summarise(stats, counts)
    keys = ("memory_end", "reached_500", "memory_per_doubling_late", "noise")
    return {
        "runs_left_out": len(flat),
        "with": {k: scalars(point)[k] for k in keys},
        "without": {k: scalars(rest)[k] for k in keys},
    }


# Figures


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, facecolor=SURFACE)
    fig.savefig(
        FIGURES / f"{name}.pdf", facecolor=SURFACE, metadata={"CreationDate": None}
    )
    plt.close(fig)


def grid(cells_out: dict, height: float):
    generators = sorted({json.loads(n)[0] for n in cells_out})
    captioners = sorted({json.loads(n)[1] for n in cells_out})
    fig, axes = plt.subplots(
        len(generators),
        len(captioners),
        figsize=(9.6, height),
        sharex=True,
        sharey=True,
    )
    fig.patch.set_facecolor(SURFACE)
    for i, g in enumerate(generators):
        for j, c in enumerate(captioners):
            ax = styled(axes[i, j])
            ax.set_title(f"{g} + {c}", loc="left", color=INK, fontsize=8.5)
            yield (
                ax,
                json.dumps([g, c], separators=(",", ":")),
                i == len(generators) - 1,
                j == 0,
                fig,
            )


def figure_memory(cells_out: dict, comparison: dict) -> None:
    for ax, name, bottom, left, fig in grid(cells_out, 8.2):
        cell = cells_out[name]["over_time"]
        t = (np.arange(len(cell["twin"])) + 0.5) * BIN
        for key, colour, label in (
            ("stranger", ORANGE, "runs of different prompts"),
            ("twin", BLUE, "runs of one prompt"),
        ):
            band = np.array(cell[f"{key}_interval"])
            ax.fill_between(
                t, band[:, 0], band[:, 1], color=colour, alpha=0.15, linewidth=0
            )
            ax.plot(t, cell[key], color=colour, linewidth=1.6, label=label)
        ax.plot(t, cell["floor"], color=MUTED, linewidth=0.8)
        ax.axvline(PANEL_STATES, color=PALE_BLUE, linewidth=0.8, linestyle=":")
        panel_end = comparison[name]["panel"]
        twin_panel = panel_end["offset_end"]
        ax.scatter(
            [PANEL_STATES - 10],
            [twin_panel + cells_out[name]["noise"]],
            s=14,
            color=BLUE,
            marker="x",
            zorder=3,
            label="the panel's two runs at 150",
        )
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 0.72)
        ax.text(
            990,
            0.09,
            f"memory at the end {cells_out[name]['memory_end']:.2f}",
            color=SECONDARY,
            fontsize=7.5,
            ha="right",
        )
        if bottom:
            ax.set_xlabel("text state", color=SECONDARY, fontsize=8.5)
        if left:
            ax.set_ylabel("cosine distance", color=SECONDARY, fontsize=8.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=SECONDARY,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "memory")


def figure_displacement(cells_out: dict) -> None:
    for ax, name, bottom, left, fig in grid(cells_out, 8.2):
        d = cells_out[name]["displacement"]
        lags = np.arange(1, MAX_LAG + 1)
        ax.plot(
            lags,
            d["twin_at_lag"][1:],
            color=BLUE,
            linewidth=1.2,
            label="to a twin, same separation",
        )
        ax.plot(
            lags,
            d["within"][1:],
            color=AQUA,
            linewidth=1.8,
            label="two states of one run",
        )
        ax.axhline(cells_out[name]["noise"], color=MUTED, linewidth=0.8)
        ax.set_xlim(0, MAX_LAG)
        ax.set_ylim(0, 0.6)
        ax.text(
            MAX_LAG - 10,
            0.03,
            f"reached at 500: {cells_out[name]['reached_500']:.0%}",
            color=SECONDARY,
            fontsize=7.5,
            ha="right",
        )
        if bottom:
            ax.set_xlabel("separation in text states", color=SECONDARY, fontsize=8.5)
        if left:
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
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "displacement")


def figure_prompts(cells_out: dict, prompts: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    fig.patch.set_facecolor(SURFACE)
    styled(ax)
    colours = (BLUE, ORANGE, AQUA, MUTED, PALE_BLUE)
    for p, prompt in enumerate(prompts):
        series = np.mean(
            [c["memory_by_prompt_curve"][p] for c in cells_out.values()], axis=0
        )
        t = (np.arange(len(series)) + 0.5) * BIN
        ax.plot(t, series, color=colours[p], linewidth=1.6, label=prompt)
    ax.axhline(0, color=AXIS_COLOUR, linewidth=0.8)
    ax.set_xlim(0, 1000)
    ax.set_xlabel("text state", color=SECONDARY, fontsize=8.5)
    ax.set_ylabel("memory, mean over the six networks", color=SECONDARY, fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY, loc="upper right")
    fig.tight_layout()
    save(fig, "prompts")


AXIS_COLOUR = panel.AXIS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=pathlib.Path)
    parser.add_argument("panel_dir", type=pathlib.Path)
    args = parser.parse_args()

    cells, prompts, captions = load_cells(args.export_dir)
    rng = np.random.default_rng(SEED)
    n_prompts, n_runs = next(iter(cells.values())).shape[:2]
    draws = rng.integers(0, n_runs, (BOOTSTRAPS, n_prompts, n_runs))
    ids = run_index(args.export_dir, prompts)
    flat = flat_runs(prompts, ids)

    cells_out, kept_by = {}, {}
    for network, x in cells.items():
        stats = statistics(x)
        out, kept = analyse_cell(stats, draws)
        point = kept["point"]
        out["memory_by_prompt"] = {
            prompt: {
                "start": float(point["memory_by_prompt"][p, 0]),
                "at_150": float(point["memory_by_prompt"][p, 13:15].mean()),
                "end": float(point["memory_by_prompt"][p, -BLOCK // BIN :].mean()),
                "end_interval": interval(
                    [
                        b["memory_by_prompt"][p, -BLOCK // BIN :].mean()
                        for b in kept["boots"]
                    ]
                ),
                "twin_end": float(point["twin_by_prompt"][p, -BLOCK // BIN :].mean()),
            }
            for p, prompt in enumerate(prompts)
        }
        out["memory_by_prompt_curve"] = point["memory_by_prompt"].tolist()
        out["ageing"] = ageing(x)
        out["frozen"] = panel.frozen(captions[network])
        out["centres"] = centres(x, captions[network], prompts, rng)
        if network in flat:
            out["without_flat_runs"] = without_flat(
                stats, flat[network], (n_prompts, n_runs)
            )
        cells_out[network] = out
        kept_by[network] = kept
        print(f"analysed {short(network)}", file=sys.stderr)

    average = {
        k: {
            "value": [
                float(v)
                for v in np.mean(
                    [blocks(kept_by[n]["point"][k]) for n in cells], axis=0
                )
            ],
            "interval": [
                interval(col)
                for col in np.mean(
                    [[blocks(b[k]) for b in kept_by[n]["boots"]] for n in cells], axis=0
                ).T
            ],
        }
        for k in ("twin", "stranger", "floor", "memory")
    }

    panel_cells, panel_prompts, _ = load_cells(args.panel_dir)
    published = json.loads(PANEL_RESULTS.read_text())["networks"]
    comparison = panel_comparison(
        cells, panel_cells, prompts, panel_prompts, published, rng
    )

    results = {
        "export": str(args.export_dir),
        "panel_export": str(args.panel_dir),
        "prompts": prompts,
        "bootstraps": BOOTSTRAPS,
        "resampled": "runs within each prompt",
        "displacement_late_from_text_state": LATE,
        "networks": cells_out,
        "six_network_average": {"block_text_states": BLOCK, **average},
        "panel_comparison": comparison,
    }
    OUT.write_text(json.dumps(results, indent=2))
    figure_memory(cells_out, comparison)
    figure_displacement(cells_out)
    figure_prompts(cells_out, prompts)

    print(
        f"{'network':<26}{'mem0':>6}{'m150':>6}{'mEnd':>6}  {'interval':<13}{'slope':>7}"
        f"  {'verdict':<22}{'r150':>6}{'r500':>6}{'cover':>7}{'strΔ':>6}"
    )
    for network, r in cells_out.items():
        iv = r["intervals"]["memory_end"]
        cover = r["covered"]["lag"] or f">{r['covered']['lower_bound'] - 1}"
        print(
            f"{short(network):<26}{r['memory_start']:>6.2f}{r['memory_150']:>6.2f}{r['memory_end']:>6.2f}"
            f"  {iv[0]:>5.2f}-{iv[1]:<6.2f}{r['memory_per_doubling_late']:>7.3f}  {r['memory_verdict']:<22}"
            f"{r['reached_150']:>6.2f}{r['reached_500']:>6.2f}{cover!s:>7}{r['stranger_change']:>6.2f}"
        )


if __name__ == "__main__":
    main()
