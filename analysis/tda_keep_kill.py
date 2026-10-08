#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars>=1.0",
#   "numpy>=2.0",
#   "scikit-learn>=1.5",
#   "giotto-ph>=0.2.4",
# ]
# ///
"""Does a run's persistence diagram say anything the distances do not? (TASK-77)

The pipeline gives every run one Vietoris-Rips diagram, to dimension 2, over
its caption embeddings (`lib/panic_tda/models/tda.ex`). TASK-103 describes the
same runs with plain distances: step size, seed noise, how far a run spreads
and moves, and how far it sits from its twin. TDA earns a place in the paper
only if the diagram carries something those numbers do not. Three tests:

- redundancy: how much of each diagram feature the distance numbers predict,
  cross-validated with whole prompts held out
- added value: whether diagram features improve on the distance numbers at
  telling the four generators, and the four captioners, apart
- a null: each run's diagram against the diagram of a Gaussian cloud with the
  run's own mean and covariance. H1 and H2 bars that a cloud with no structure
  also produces are geometry of the noise, not loops in the run.

A Rips diagram is computed on the set of states, so it cannot see their order:
shuffling a run in time leaves it unchanged. Recurrence in time is a separate
question (sliding-window persistence), not tested here.

The embeddings are unit vectors and the diagrams use Euclidean distance, which
is a monotone function of cosine distance, so the filtration is the cosine one
reparametrised. Distance numbers below are cosine distances.

    ./analysis/tda_keep_kill.py long_horizon_panel_01a09e21_parquet

Results -> analysis/tda_keep_kill.json, summary to stdout.
"""

import argparse
import json
import pathlib

import numpy as np
import polars as pl
from gph import ripser_parallel
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = pathlib.Path(__file__).with_suffix(".json")
LATE = 75
FOLDS = 5
NULL_DRAWS = 3
BOOTSTRAPS = 1000
SEED = 0

BASELINE = (
    "step", "noise", "spread", "spread_late", "displacement",
    "twin_late", "unique_share", "repeat_share", "nearest", "nearest_late",
    "participation", "twonn",
)  # fmt: skip


def load(export_dir: pathlib.Path) -> pl.DataFrame:
    """One row per run: its labels, embeddings (150 x 256) and captions."""
    embeddings = pl.read_parquet(export_dir / "embeddings.parquet").select(
        "invocation_id", "vector"
    )
    invocations = pl.read_parquet(export_dir / "invocations.parquet").select(
        "id", "run_id", "sequence_number", "output_text"
    )
    runs = pl.read_parquet(export_dir / "runs.parquet").select(
        "id", "network", "initial_prompt", "run_number"
    )
    diagrams = pl.read_parquet(export_dir / "persistence_diagrams.parquet").select(
        "run_id", "diagram_data"
    )
    steps = (
        embeddings.join(invocations, left_on="invocation_id", right_on="id")
        .filter(pl.col("output_text").is_not_null())
        .sort("run_id", "sequence_number")
        .group_by("run_id", maintain_order=True)
        .agg("vector", "output_text")
    )
    return (
        runs.join(steps, left_on="id", right_on="run_id")
        .join(diagrams, left_on="id", right_on="run_id")
        .with_columns(
            pl.col("network").str.json_decode(pl.List(pl.String)).alias("pair")
        )
        .with_columns(
            pl.col("pair").list.get(0).alias("generator"),
            pl.col("pair").list.get(1).alias("captioner"),
        )
        .sort("network", "initial_prompt", "run_number")
    )


def bars(dgms: list) -> list[np.ndarray]:
    """Finite bars per dimension, as (birth, death) arrays."""
    out = []
    for d in dgms:
        a = np.asarray(d, dtype=np.float64).reshape(-1, 2)
        out.append(a[np.isfinite(a[:, 1])])
    return out


def diagram_features(dgms: list) -> dict[str, float]:
    """Counts, total and longest persistence, and entropy per dimension.

    Zero-length bars (from repeated captions) are dropped before anything is
    counted, so H0's count is the number of distinct points less one.
    """
    feats = {}
    for dim, a in enumerate(bars(dgms)):
        life = a[:, 1] - a[:, 0]
        life = life[life > 1e-9]
        n = len(life)
        total = float(life.sum())
        p = life / total if total > 0 else life
        entropy = float(-(p * np.log(p)).sum()) if n else 0.0
        feats |= {
            f"h{dim}_count": float(n),
            f"h{dim}_total": total,
            f"h{dim}_max": float(life.max()) if n else 0.0,
            f"h{dim}_entropy": entropy,
            f"h{dim}_entropy_norm": entropy / np.log(n) if n > 1 else 0.0,
        }
    return feats


def rips(x: np.ndarray) -> list:
    return ripser_parallel(x, maxdim=2, n_threads=4)["dgms"]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - (a * b).sum(axis=-1)


def participation(x: np.ndarray) -> float:
    """How many directions the run's variance is spread over: (sum l)^2 / sum l^2."""
    lam = np.linalg.svd(x - x.mean(axis=0), compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam**2).sum())


def twonn(x: np.ndarray) -> float:
    """Local intrinsic dimension from the ratio of each point's second to first
    nearest-neighbour distance (Facco et al. 2017). Repeated captions, which
    have a first neighbour at distance zero, are left out."""
    _, keep = np.unique(x.round(6), axis=0, return_index=True)
    y = x[np.sort(keep)]
    d = np.sqrt(np.maximum(2.0 * (1.0 - y @ y.T), 0.0))
    np.fill_diagonal(d, np.inf)
    d.sort(axis=1)
    mu = d[:, 1] / d[:, 0]
    mu = mu[np.isfinite(mu) & (mu > 1)]
    return float(len(mu) / np.log(mu).sum())


def baseline_features(x: np.ndarray, twin: np.ndarray, captions: list) -> dict:
    """TASK-103's vocabulary, per run. x and twin are (150, 256) unit vectors."""
    d1 = cosine(x[1:], x[:-1]).mean()
    d2 = cosine(x[2:], x[:-2]).mean()
    gram = 1.0 - x @ x.T
    iu = np.triu_indices(len(x), 1)
    late = gram[LATE:, LATE:][np.triu_indices(len(x) - LATE, 1)]
    caps = np.asarray(captions, dtype=object)
    np.fill_diagonal(gram, np.inf)
    nearest = gram.min(axis=1)
    return {
        "participation": participation(x),
        "twonn": twonn(x),
        "nearest": float(nearest.mean()),
        "nearest_late": float(nearest[LATE:].mean()),
        "step": float(d1),
        "noise": float(2 * d1 - d2),
        "spread": float(gram[iu].mean()),
        "spread_late": float(late.mean()),
        "displacement": float((1.0 - x[:10] @ x[-10:].T).mean()),
        "twin_late": float(cosine(x[LATE:], twin[LATE:]).mean()),
        "unique_share": len(set(captions)) / len(captions),
        "repeat_share": float((caps[1:] == caps[:-1]).mean()),
    }


def null_features(x: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Mean diagram features of Gaussian clouds with x's mean and covariance,
    projected back to the sphere as the real embeddings are."""
    mu = x.mean(axis=0)
    centred = x - mu
    draws = []
    for _ in range(NULL_DRAWS):
        # x's covariance, sampled through its own centred rows: rank <= n-1 as x's
        z = rng.standard_normal((len(x), len(x))) / np.sqrt(len(x) - 1)
        y = mu + z @ centred
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        draws.append(diagram_features(rips(y)))
    return {k: float(np.mean([d[k] for d in draws])) for k in draws[0]}


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def redundancy(table: pl.DataFrame, tda: list[str]) -> dict[str, float]:
    """Cross-validated R^2 of each diagram feature from the distance numbers,
    whole prompts held out."""
    X = table.select(BASELINE).to_numpy()
    groups = table["initial_prompt"].to_numpy()
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    out = {}
    for f in tda:
        y = table[f].to_numpy()
        if y.std() == 0:
            continue
        pred = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(FOLDS))
        out[f] = r2(y, pred)
    return out


def classify(table: pl.DataFrame, cols: list[str], label: str) -> np.ndarray:
    """Held-out predictions of a label, whole prompts held out."""
    X = table.select(cols).to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    return cross_val_predict(
        model, X, table[label].to_numpy(),
        groups=table["initial_prompt"].to_numpy(), cv=GroupKFold(FOLDS),
    )  # fmt: skip


def added_value(table: pl.DataFrame, tda: list[str], rng) -> dict:
    """Accuracy at naming the generator and the captioner from the distance
    numbers, the diagram, and both; the gain is bootstrapped over prompts.

    H0's bars are the edges of the minimum spanning tree, so they are a
    statement about spacing between captions rather than about shape. The
    second gain asks what H1 and H2 add once H0 is in."""
    h0 = [c for c in tda if c.startswith("h0_")]
    prompts = table["initial_prompt"].to_numpy()
    unique = np.unique(prompts)
    index = {p: np.flatnonzero(prompts == p) for p in unique}
    out = {}
    for label in ("generator", "captioner"):
        truth = table[label].to_numpy()
        hits = {
            name: classify(table, cols, label) == truth
            for name, cols in (
                ("distances", list(BASELINE)),
                ("diagram", tda),
                ("both", list(BASELINE) + tda),
                ("distances_h0", list(BASELINE) + h0),
                ("distances_h0_h1_h2", list(BASELINE) + tda),
            )
        }
        gains, higher = [], []
        for _ in range(BOOTSTRAPS):
            rows = np.concatenate([index[p] for p in rng.choice(unique, len(unique))])
            gains.append(hits["both"][rows].mean() - hits["distances"][rows].mean())
            higher.append(
                hits["distances_h0_h1_h2"][rows].mean() - hits["distances_h0"][rows].mean()
            )
        out[label] = {k: float(v.mean()) for k, v in hits.items()} | {
            "gain": float(np.mean(gains)),
            "gain_interval": [float(q) for q in np.percentile(gains, [2.5, 97.5])],
            "gain_h1_h2_over_h0": float(np.mean(higher)),
            "gain_h1_h2_over_h0_interval": [
                float(q) for q in np.percentile(higher, [2.5, 97.5])
            ],
            "chance": 0.25,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("export", type=pathlib.Path)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    runs = load(args.export)
    xs = [
        (lambda a: a / np.linalg.norm(a, axis=1, keepdims=True))(
            np.asarray(v.to_list(), dtype=np.float64)
        )
        for v in runs["vector"]
    ]
    key = list(zip(runs["network"], runs["initial_prompt"], runs["run_number"]))
    where = {k: i for i, k in enumerate(key)}

    rows, stored_vs_recomputed = [], []
    for i, (network, prompt, run) in enumerate(key):
        twin = xs[where[(network, prompt, 1 - run if run in (0, 1) else run)]]
        stored = json.loads(runs["diagram_data"][i])["dgms"]
        row = {
            "network": network,
            "generator": runs["generator"][i],
            "captioner": runs["captioner"][i],
            "initial_prompt": prompt,
            "run_number": run,
        }
        row |= baseline_features(xs[i], twin, runs["output_text"][i].to_list())
        row |= diagram_features(stored)
        row |= {f"null_{k}": v for k, v in null_features(xs[i], rng).items()}
        if i % 40 == 0:
            again = diagram_features(rips(xs[i]))
            stored_vs_recomputed.append(
                max(abs(again[k] - row[k]) for k in again if k.endswith("_total"))
            )
        rows.append(row)
    table = pl.DataFrame(rows)
    table.write_parquet(OUT.with_suffix(".parquet"))

    tda = [c for c in diagram_features(json.loads(runs["diagram_data"][0])["dgms"])]
    tda = [c for c in tda if table[c].std() > 0]

    null = {}
    for f in tda:
        diff = (table[f] - table[f"null_{f}"]).to_numpy()
        null[f] = {
            "real": float(table[f].mean()),
            "null": float(table[f"null_{f}"].mean()),
            "share_of_runs_above_null": float((diff > 0).mean()),
        }

    result = {
        "export": str(args.export),
        "runs": table.height,
        "late_from_text_state": LATE,
        "stored_diagram_max_total_difference": float(max(stored_vs_recomputed)),
        "redundancy_r2": redundancy(table, tda),
        "added_value": added_value(table, tda, rng),
        "against_null": null,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    print(f"stored vs recomputed diagrams, worst total-persistence gap: "
          f"{result['stored_diagram_max_total_difference']:.2e}")
    print("\nredundancy: R^2 of each diagram feature from the distance numbers")
    for f, v in result["redundancy_r2"].items():
        print(f"  {f:18s} {v:6.3f}")
    print("\nadded value: held-out accuracy (chance 0.25)")
    for label, v in result["added_value"].items():
        lo, hi = v["gain_interval"]
        print(f"  {label:10s} distances {v['distances']:.3f}  diagram {v['diagram']:.3f}"
              f"  both {v['both']:.3f}  gain {v['gain']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lo, hi = v["gain_h1_h2_over_h0_interval"]
        print(f"  {'':10s} distances+H0 {v['distances_h0']:.3f}  +H1,H2 "
              f"{v['distances_h0_h1_h2']:.3f}  gain {v['gain_h1_h2_over_h0']:+.3f}"
              f" [{lo:+.3f}, {hi:+.3f}]")
    print("\nagainst a Gaussian cloud of the run's own shape")
    for f, v in null.items():
        print(f"  {f:18s} real {v['real']:8.3f}  null {v['null']:8.3f}"
              f"  runs above null {v['share_of_runs_above_null']:.2f}")


if __name__ == "__main__":
    main()
