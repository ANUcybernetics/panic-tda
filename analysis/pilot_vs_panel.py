#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["polars>=1.0", "numpy>=2.0", "scipy>=1.14"]
# ///
"""Did mid-sentence caption truncation change the dynamics, or only the captions?

Compares the caption-length pilot (Flux2Klein+Gemma3n, captioner ceiling 1024)
against the same network arm of `balanced_panel_5x5`, which ran at the old
hardcoded 128-token ceiling. Same 20 prompts, 4 runs each, 50 steps, Qwen3Embed.

    ./analysis/pilot_vs_panel.py [panel_parquet_dir] [pilot_parquet_dir]

Four families of comparison (TASK-85 AC #3):

1. captions      -- word length, terminal-punctuation share, by step
2. (FTLE was dropped from the analysis in TASK-73 and is not reported)
3. clusters      -- occupancy and transition structure over a shared alphabet
4. drift         -- cosine distance from the t_0 prompt, and step to step

The cluster alphabet needs care. EVoC clustering is global and was last run
before the pilot existed, so the pilot has no labels of its own and reclustering
would relabel both conditions at once. Instead both conditions are assigned to
the *panel's* existing medoids by nearest cosine, which keeps the symbol
alphabet fixed and makes occupancy directly comparable. Medoid vectors come from
the dev database because a medoid may live in an experiment outside either dump.
The script reports how often that assignment reproduces EVoC's own label on the
panel rows, as a check on the method.

Results -> analysis/pilot_vs_panel.json, summary table to stdout.
"""

import json
import pathlib
import re
import sqlite3
import sys

import numpy as np
import polars as pl
from scipy import stats

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
DB = ROOT / "priv" / "panic_tda_dev.db"
OUT = HERE / "pilot_vs_panel.json"

PANEL_DIR = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else ROOT / "019f3645_parquet"
)
PILOT_DIR = pathlib.Path(
    sys.argv[2] if len(sys.argv) > 2 else ROOT / "01a060b4_parquet"
)

# `network` is stored as a JSON string in the runs table
NETWORK = '["Flux2Klein","Gemma3n"]'
TERMINAL = re.compile(r'[.!?]["\')\]]*\s*$')

report: dict = {}


# --------------------------------------------------------------------------
# loading


def load_arm(export_dir: pathlib.Path, network: str | None) -> pl.DataFrame:
    """One row per text embedding in the arm, with run, step, caption, vector."""
    runs = pl.read_parquet(export_dir / "runs.parquet").select(
        "id", "experiment_id", "network", "initial_prompt", "run_number"
    )
    if network is not None:
        runs = runs.filter(pl.col("network") == network)
    inv = pl.read_parquet(
        export_dir / "invocations.parquet",
        columns=["id", "run_id", "sequence_number", "type", "model", "output_text"],
    ).filter(pl.col("type") == "text")
    emb = pl.read_parquet(
        export_dir / "embeddings.parquet", columns=["id", "invocation_id", "vector"]
    ).rename({"id": "embedding_id"})

    return (
        emb.join(inv, left_on="invocation_id", right_on="id")
        .join(runs, left_on="run_id", right_on="id")
        .sort(["run_id", "sequence_number"])
    )


def caption_stats(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        words=pl.col("output_text").str.extract_all(r"\S+").list.len(),
        chars=pl.col("output_text").str.len_chars(),
        complete=pl.col("output_text").str.strip_chars().str.contains(TERMINAL.pattern),
    )


def unit(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def vectors(frame: pl.DataFrame) -> np.ndarray:
    return unit(np.asarray(frame["vector"].to_list(), dtype=np.float32))


# --------------------------------------------------------------------------
# 1. captions

panel = caption_stats(load_arm(PANEL_DIR, NETWORK))
pilot = caption_stats(load_arm(PILOT_DIR, None))
# drop the synthetic t_0 prompt rows from the caption stats; keep them for drift
panel_c = panel.filter(pl.col("sequence_number") >= 0)
pilot_c = pilot.filter(pl.col("sequence_number") >= 0)

print(f"panel arm {panel_c.height} captions over {panel_c['run_id'].n_unique()} runs")
print(f"pilot     {pilot_c.height} captions over {pilot_c['run_id'].n_unique()} runs")


def summarise(frame: pl.DataFrame, label: str) -> dict:
    w = frame["words"].to_numpy()
    return {
        "condition": label,
        "n": int(frame.height),
        "words_median": float(np.median(w)),
        "words_p10": float(np.percentile(w, 10)),
        "words_p90": float(np.percentile(w, 90)),
        "words_max": int(w.max()),
        "complete_share": float(frame["complete"].mean()),
    }


report["captions"] = {
    "panel": summarise(panel_c, "balanced_panel_5x5 (ceiling 128)"),
    "pilot": summarise(pilot_c, "caption pilot (ceiling 1024)"),
    "mann_whitney_p": float(
        stats.mannwhitneyu(
            panel_c["words"].to_numpy(), pilot_c["words"].to_numpy()
        ).pvalue
    ),
}

# --------------------------------------------------------------------------
# 2. FTLE, paired over the 20 prompts



# --------------------------------------------------------------------------
# 3. cluster occupancy and transitions over the panel's medoid alphabet

con = sqlite3.connect(DB)
layers = con.execute(
    "select id, layer from clustering_results where embedding_model='Qwen3Embed' order by layer"
).fetchall()
cluster_report = {}

panel_labels_evoc = (
    pl.read_parquet(PANEL_DIR / "embedding_clusters.parquet")
    .join(
        pl.read_parquet(PANEL_DIR / "clustering_results.parquet").select("id", "layer"),
        left_on="clustering_result_id",
        right_on="id",
    )
    .select("embedding_id", "layer", "medoid_embedding_id")
)

for layer_id, layer in layers:
    medoid_ids = [
        r[0]
        for r in con.execute(
            "select distinct medoid_embedding_id from embedding_clusters "
            "where clustering_result_id=? and medoid_embedding_id is not null",
            (layer_id,),
        )
    ]
    rows = con.execute(
        f"select id, vector from embeddings where id in ({','.join('?' * len(medoid_ids))})",
        medoid_ids,
    ).fetchall()
    ids = [r[0] for r in rows]
    med = unit(np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows]))
    index = {mid: i for i, mid in enumerate(ids)}

    def assign(frame: pl.DataFrame) -> np.ndarray:
        return (vectors(frame) @ med.T).argmax(axis=1)

    panel_sym = assign(panel_c)
    pilot_sym = assign(pilot_c)

    # sanity: how often does nearest-medoid reproduce EVoC's own panel label?
    truth = (
        panel_c.select("embedding_id")
        .join(
            panel_labels_evoc.filter(pl.col("layer") == layer).select(
                "embedding_id", "medoid_embedding_id"
            ),
            on="embedding_id",
            how="left",
        )["medoid_embedding_id"]
        .to_list()
    )
    labelled = [(i, t) for i, t in enumerate(truth) if t is not None and t in index]
    agreement = (
        float(np.mean([panel_sym[i] == index[t] for i, t in labelled]))
        if labelled
        else None
    )

    def structure(frame: pl.DataFrame, sym: np.ndarray) -> dict:
        k = med.shape[0]
        counts = np.bincount(sym, minlength=k).astype(float)
        occ = counts / counts.sum()
        run_ids = frame["run_id"].to_numpy()
        trans = np.zeros((k, k))
        distinct, self_trans, steps = [], 0, 0
        for rid in np.unique(run_ids):
            seq = sym[run_ids == rid]
            distinct.append(len(np.unique(seq)))
            for x, y in zip(seq[:-1], seq[1:]):
                trans[x, y] += 1
                self_trans += x == y
                steps += 1
        nz = occ[occ > 0]
        return {
            "occupancy": occ,
            "occupied_clusters": int((counts > 0).sum()),
            "occupancy_entropy_bits": float(-(nz * np.log2(nz)).sum()),
            "distinct_per_run_mean": float(np.mean(distinct)),
            "self_transition_rate": float(self_trans / steps),
            "transitions": trans,
        }

    p_struct, q_struct = structure(panel_c, panel_sym), structure(pilot_c, pilot_sym)

    def js(p: np.ndarray, q: np.ndarray) -> float:
        m = (p + q) / 2

        def kl(x: np.ndarray, y: np.ndarray) -> float:
            nz = x > 0  # m is positive wherever either p or q is, so y > 0 there
            return float((x[nz] * np.log2(x[nz] / y[nz])).sum())

        return (kl(p, m) + kl(q, m)) / 2

    tp = p_struct["transitions"] / max(p_struct["transitions"].sum(), 1)
    tq = q_struct["transitions"] / max(q_struct["transitions"].sum(), 1)
    cluster_report[f"layer_{layer}"] = {
        "n_clusters": int(med.shape[0]),
        "nearest_medoid_agrees_with_evoc": agreement,
        "panel": {k: v for k, v in p_struct.items() if not isinstance(v, np.ndarray)},
        "pilot": {k: v for k, v in q_struct.items() if not isinstance(v, np.ndarray)},
        "occupancy_js_bits": js(p_struct["occupancy"], q_struct["occupancy"]),
        "transition_js_bits": js(tp.ravel(), tq.ravel()),
    }

report["clusters"] = cluster_report

# --------------------------------------------------------------------------
# 4. drift from t_0 and step to step


def drift(frame_all: pl.DataFrame) -> dict:
    from_prompt, step_to_step = {}, {}
    for rid, group in frame_all.group_by("run_id"):
        group = group.sort("sequence_number")
        vecs = vectors(group)
        seqs = group["sequence_number"].to_numpy()
        if seqs[0] == -1:
            origin, vecs, seqs = vecs[0], vecs[1:], seqs[1:]
            for s, cos in zip(seqs, vecs @ origin):
                from_prompt.setdefault(int(s), []).append(1.0 - float(cos))
        for s, cos in zip(seqs[1:], (vecs[:-1] * vecs[1:]).sum(axis=1)):
            step_to_step.setdefault(int(s), []).append(1.0 - float(cos))
    curve = lambda d: {s: float(np.mean(v)) for s, v in sorted(d.items())}
    flat = lambda d: (
        np.concatenate([np.asarray(v) for v in d.values()]) if d else np.array([])
    )
    return {
        "from_prompt_curve": curve(from_prompt),
        "step_to_step_curve": curve(step_to_step),
        "from_prompt_mean": float(flat(from_prompt).mean()) if from_prompt else None,
        "step_to_step_mean": float(flat(step_to_step).mean()),
        "step_to_step_late_mean": float(
            np.mean([np.mean(v) for s, v in step_to_step.items() if s >= 25])
        ),
    }


report["drift"] = {"panel": drift(panel), "pilot": drift(pilot)}

# --------------------------------------------------------------------------

OUT.write_text(json.dumps(report, indent=2, default=float))

c = report["captions"]
print("\n== captions")
for key in ("panel", "pilot"):
    s = c[key]
    print(
        f"  {s['condition']:35}  median {s['words_median']:6.1f} w  "
        f"p10-p90 {s['words_p10']:.0f}-{s['words_p90']:.0f}  max {s['words_max']:4d}  "
        f"complete {s['complete_share']:.1%}"
    )
print("\n== clusters (panel medoid alphabet)")
for name, layer in report["clusters"].items():
    print(
        f"  {name}: k={layer['n_clusters']:4d}  agree={layer['nearest_medoid_agrees_with_evoc']}  "
        f"occupancy JS {layer['occupancy_js_bits']:.4f} bits  "
        f"transition JS {layer['transition_js_bits']:.4f} bits"
    )
    for key in ("panel", "pilot"):
        s = layer[key]
        print(
            f"      {key:6} occupied {s['occupied_clusters']:4d}  H {s['occupancy_entropy_bits']:.2f} bits  "
            f"distinct/run {s['distinct_per_run_mean']:.1f}  self-trans {s['self_transition_rate']:.1%}"
        )
print("\n== drift")
for key in ("panel", "pilot"):
    d = report["drift"][key]
    print(
        f"  {key:6} from t_0 {d['from_prompt_mean']}  step-to-step {d['step_to_step_mean']:.4f}  "
        f"late (>=25) {d['step_to_step_late_mean']:.4f}"
    )
print(f"\nwrote {OUT}")
