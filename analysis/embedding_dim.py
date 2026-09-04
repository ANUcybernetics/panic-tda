#!/usr/bin/env python
"""Does truncating Qwen3Embed to 256 dimensions change the dynamics? (TASK-89 AC#6)

Every embedding in the pipeline is Matryoshka-truncated from Qwen3Embed's native
2560 dimensions to 256 and renormalised (panic_models._encode_embedding). That
choice was never measured on our own captions, and every step-size, plateau and
clustering result rests on it.

Method: re-embed the caption pilot's trajectories (01a060b4 --- natural-length
captions, the regime the panel will run in) at the native dimension, derive the
256-d vectors by the same truncation, and compare:

  - rank correlation of pairwise distances over a sample of captions
  - the step-size and plateau statistics by step bin, at both dimensions
  - top-k neighbour overlap, which is what clustering actually depends on

Also checks the derived 256-d vectors against the ones already in the database,
which confirms the truncation reproduces the pipeline's own.

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python analysis/embedding_dim.py

Results -> analysis/embedding_dim.json
"""

import base64
import json
import pathlib
import sqlite3
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/home/ben/projects/panic_tda/priv/python")
import panic_models as pm  # noqa: E402

HERE = pathlib.Path(__file__).parent
DB = pathlib.Path("/home/ben/projects/panic_tda/priv/panic_tda_dev.db")
OUT = HERE / "embedding_dim.json"
EXPERIMENT = "01a060b4"
DIM = 256
BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
N_PAIRWISE = 500
TOPK = 10


def load() -> tuple[list[str], list[tuple[str, int]], dict[str, np.ndarray]]:
    con = sqlite3.connect(DB)
    rows = con.execute(
        """
        select i.run_id, i.sequence_number, i.output_text, e.vector
        from invocations i
        join runs r on r.id = i.run_id
        left join embeddings e
          on e.invocation_id = i.id and e.embedding_model = 'Qwen3Embed'
        where r.experiment_id like ? and i.type = 'text'
          and i.output_text is not null
        order by i.run_id, i.sequence_number
        """,
        (f"{EXPERIMENT}%",),
    ).fetchall()
    con.close()
    texts, index, stored = [], [], {}
    for run_id, seq, text, blob in rows:
        key = f"{run_id}|{seq}"
        texts.append(text)
        index.append((run_id, seq))
        if blob is not None:
            v = np.frombuffer(blob, dtype=np.float32)
            stored[key] = v / np.linalg.norm(v)
    return texts, index, stored


def truncate(v: np.ndarray, dim: int) -> np.ndarray:
    w = v[..., :dim]
    return w / np.linalg.norm(w, axis=-1, keepdims=True)


def trajectory_stats(
    index: list[tuple[str, int]], vecs: np.ndarray
) -> dict[str, dict[str, float]]:
    """Median step-to-step distance and distance from the first caption, by bin."""
    runs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, (run_id, seq) in enumerate(index):
        runs[run_id].append((seq, i))
    step: dict[tuple[int, int], list[float]] = defaultdict(list)
    drift: dict[tuple[int, int], list[float]] = defaultdict(list)
    for entries in runs.values():
        entries.sort()
        v0 = vecs[entries[0][1]]
        for (_s1, a), (s2, b) in zip(entries, entries[1:]):
            for lo, hi in BINS:
                if lo <= s2 < hi:
                    step[(lo, hi)].append(1 - float(vecs[a] @ vecs[b]))
                    drift[(lo, hi)].append(1 - float(v0 @ vecs[b]))
    return {
        "median_step_distance": {
            f"{lo}-{hi}": float(np.median(step[(lo, hi)])) for lo, hi in BINS
        },
        "median_distance_from_first": {
            f"{lo}-{hi}": float(np.median(drift[(lo, hi)])) for lo, hi in BINS
        },
        "mean_step_distance": float(np.mean([d for vs in step.values() for d in vs])),
    }


def main() -> None:
    texts, index, stored = load()
    print(f"{len(texts)} captions from {EXPERIMENT}", flush=True)

    pm.setup()
    pm.load_model("Qwen3Embed")
    pm.swap_to_gpu("Qwen3Embed")
    chunks = []
    for i in range(0, len(texts), 64):
        chunks.append(
            pm._models["Qwen3Embed"].encode(
                texts[i : i + 64], convert_to_numpy=True, normalize_embeddings=True
            )
        )
        print(f"  embedded {min(i + 64, len(texts))}/{len(texts)}", flush=True)
    pm.unload_model("Qwen3Embed")
    native = np.concatenate(chunks).astype(np.float32)
    short = truncate(native, DIM)
    print(f"native dimension {native.shape[1]}", flush=True)

    # does the derived truncation reproduce what the pipeline stored?
    matched = [
        float(short[i] @ stored[f"{run_id}|{seq}"])
        for i, (run_id, seq) in enumerate(index)
        if f"{run_id}|{seq}" in stored
    ]

    rng = np.random.default_rng(0)
    idx = rng.choice(len(native), size=min(N_PAIRWISE, len(native)), replace=False)
    a, b = short[idx], native[idx]
    iu = np.triu_indices(len(idx), 1)
    da, db = (1 - a @ a.T)[iu], (1 - b @ b.T)[iu]

    # top-k neighbour overlap, which is what clustering depends on
    sa, sb = short[idx] @ short[idx].T, native[idx] @ native[idx].T
    np.fill_diagonal(sa, -np.inf)
    np.fill_diagonal(sb, -np.inf)
    overlap = [
        len(
            set(np.argpartition(-sa[i], TOPK)[:TOPK])
            & set(np.argpartition(-sb[i], TOPK)[:TOPK])
        )
        / TOPK
        for i in range(len(idx))
    ]

    results = {
        "experiment": EXPERIMENT,
        "n_captions": len(texts),
        "native_dim": int(native.shape[1]),
        "truncated_dim": DIM,
        "reproduces_stored_vectors": {
            "n": len(matched),
            "min_cosine": float(np.min(matched)) if matched else None,
            "mean_cosine": float(np.mean(matched)) if matched else None,
        },
        "pairwise_distance": {
            "n_vectors": int(len(idx)),
            "n_pairs": int(len(da)),
            "spearman": float(spearmanr(da, db).statistic),
            "pearson": float(np.corrcoef(da, db)[0, 1]),
            "mean_256": float(da.mean()),
            "mean_native": float(db.mean()),
            "ratio_256_over_native": float(da.mean() / db.mean()),
        },
        f"top{TOPK}_neighbour_overlap": {
            "mean": float(np.mean(overlap)),
            "p10": float(np.percentile(overlap, 10)),
            "frac_perfect": float(np.mean([o == 1.0 for o in overlap])),
        },
        "dynamics": {
            "256": trajectory_stats(index, short),
            "native": trajectory_stats(index, native),
        },
    }
    OUT.write_text(json.dumps(results, indent=2) + "\n")

    print(
        f"\nderived 256-d vs stored: mean cosine "
        f"{results['reproduces_stored_vectors']['mean_cosine']:.6f} "
        f"(min {results['reproduces_stored_vectors']['min_cosine']:.6f})"
    )
    pw = results["pairwise_distance"]
    print(
        f"pairwise distances over {pw['n_pairs']} pairs: spearman {pw['spearman']:.4f}, "
        f"pearson {pw['pearson']:.4f}, 256-d distances are {pw['ratio_256_over_native']:.2f}x native"
    )
    nb = results[f"top{TOPK}_neighbour_overlap"]
    print(
        f"top-{TOPK} neighbour overlap: mean {nb['mean']:.3f}, "
        f"p10 {nb['p10']:.3f}, perfect for {100 * nb['frac_perfect']:.0f}%"
    )
    print("\nmedian step-to-step distance by step bin")
    header = "".join(f"{lo}-{hi:<8d}" for lo, hi in BINS)
    print(f"{'dim':8s}{header}")
    for label in ("256", "native"):
        row = "".join(
            f"{v:.4f}    "
            for v in results["dynamics"][label]["median_step_distance"].values()
        )
        print(f"{label:8s}{row}")
    print("\nmedian distance from the first caption by step bin")
    print(f"{'dim':8s}{header}")
    for label in ("256", "native"):
        row = "".join(
            f"{v:.4f}    "
            for v in results["dynamics"][label]["median_distance_from_first"].values()
        )
        print(f"{label:8s}{row}")
    print(f"\nwrote {OUT}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
