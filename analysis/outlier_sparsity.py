#!/usr/bin/env python
"""Are EVoC's outliers the sparse part of semantic space? (TASK-75)

The global clustering leaves a large share of embeddings unlabelled, and the
research programme reads that as genuine sparsity: only the dense regions are
clusters (Hartigan's level-set view), and an outlier is a state in transit
between them. The symbolic-dynamics analysis (TASK-76) treats time spent as an
outlier as a real observable, so this makes the reading a measurement.

Four checks, all over every Qwen3Embed embedding in the dev database (the same
pool `mix cluster.recompute` clusters):

1. density   -- cosine distance to the k-th nearest neighbour (k = 5, the
                clusterer's min_samples, and k = 15, its n_neighbors), on the
                unit sphere. Is the distribution bimodal (dense cores plus a
                diffuse sea) or one continuum? And do the stored outliers sit
                in the sparse part of it?
2. sweep     -- refit EVoC across base_min_cluster_size and noise_level and
                report the outlier fraction per layer. Sparse-sea outliers stay
                outliers as the minimum size falls; small dense groups get
                absorbed.
3. length    -- outlier status against caption length and captioner, and a
                logistic fit of outlier status on density, length and captioner
                together, so length cannot masquerade as sparsity.
4. position  -- outlier rate by trajectory position. Transit states should be
                early states.
5. structure -- who an outlier's neighbours are (other outliers, or members of
                a cluster it sits at the edge of), and how outlier stretches
                within a trajectory are arranged: a transit leaves one cluster
                and arrives at another, an excursion returns to the same one,
                and a flicker is one or two outlier steps inside a cluster.
6. coverage  -- how many distinct runs visit each labelled cluster, against how
                many visit each connected region of the outlier set. Points
                along a trajectory are not independent samples: a run that
                dwells somewhere for thirty steps makes that place dense on
                its own. The sparsity that matters for "time in transit" is
                coverage by distinct trajectories, not point density.
7. second pass -- refit EVoC on the outlier set alone. If the mass has
                structure the global fit could not resolve, it appears here;
                if it is a genuine continuum, most of it stays unlabelled.

EVoC's outliers are HDBSCAN noise in a low-dimensional layout of the 15-NN
graph, and noise_level is a repulsion term in that layout rather than an
outlier threshold (see evoc/clustering.py and evoc/node_embedding.py).

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python analysis/outlier_sparsity.py [db_path]

Resumable: each EVoC fit is checkpointed to analysis/outlier_sparsity/sweep.json.
Results -> analysis/outlier_sparsity.json, figures -> analysis/outlier_sparsity/
"""

import itertools
import json
import pathlib
import sqlite3
import sys
import time

import evoc
import matplotlib
import numpy as np
import torch
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "priv/panic_tda_dev.db")
HERE = pathlib.Path(__file__).parent
OUTDIR = HERE / "outlier_sparsity"
OUT = HERE / "outlier_sparsity.json"
SWEEP_CKPT = OUTDIR / "sweep.json"

EMBEDDING_MODEL = "Qwen3Embed"
KS = (5, 15)
PRODUCTION = {"base_min_cluster_size": 147, "noise_level": 0.5}
MIN_SIZES = (5, 15, 50, 147, 300, 600, 1500)
NOISE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
SEQ_BINS = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 200)]
LOG_FLOOR = 1e-4  # exact caption repeats give d = 0

# reference palette (dataviz skill), light surface
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
)
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#c3c2b7",
        "lines.linewidth": 1.5,
        "figure.dpi": 150,
    }
)


# ---------------------------------------------------------------- loading


def load(con: sqlite3.Connection) -> dict:
    t0 = time.time()
    rows = con.execute(
        """
        select em.id, em.vector, i.output_text, i.model, i.sequence_number,
               substr(e.id, 1, 8), e.max_length, r.id, r.network
        from embeddings em
        join invocations i on i.id = em.invocation_id
        join runs r on r.id = i.run_id
        join experiments e on e.id = r.experiment_id
        where em.embedding_model = ?
        order by em.id
        """,
        (EMBEDDING_MODEL,),
    ).fetchall()
    ids = [r[0] for r in rows]
    X = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    words = np.array([len((r[2] or "").split()) for r in rows])
    captioner = np.array([r[3] for r in rows])
    seq = np.array([r[4] for r in rows])
    experiment = np.array([r[5] for r in rows])
    max_length = np.array([r[6] for r in rows])
    run = np.array([r[7] for r in rows])
    network = np.array(["+".join(json.loads(r[8])) for r in rows])

    index = {eid: i for i, eid in enumerate(ids)}
    layer_rows = con.execute(
        """
        select cr.layer, ec.embedding_id, ec.medoid_embedding_id
        from embedding_clusters ec
        join clustering_results cr on cr.id = ec.clustering_result_id
        where cr.embedding_model = ? and cr.algorithm = 'evoc'
        """,
        (EMBEDDING_MODEL,),
    ).fetchall()
    n_layers = max(r[0] for r in layer_rows) + 1
    stored = np.full((n_layers, len(ids)), -1, dtype=np.int64)
    medoid_ids: list[dict[str, int]] = [{} for _ in range(n_layers)]
    for layer, eid, medoid in layer_rows:
        if medoid is None:
            continue
        stored[layer, index[eid]] = medoid_ids[layer].setdefault(
            medoid, len(medoid_ids[layer])
        )
    print(
        f"loaded {len(ids)} embeddings, {n_layers} stored layers in {time.time() - t0:.0f}s"
    )
    return {
        "X": X,
        "words": words,
        "captioner": captioner,
        "seq": seq,
        "experiment": experiment,
        "max_length": max_length,
        "run": run,
        "network": network,
        "stored": stored,
    }


# ---------------------------------------------------------------- density


def knn(X: np.ndarray, kmax: int, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Cosine distance to, and index of, the 1st..kmax-th nearest neighbour."""
    t0 = time.time()
    Xt = torch.from_numpy(X).cuda()
    dist = np.empty((len(X), kmax), dtype=np.float32)
    idx = np.empty((len(X), kmax), dtype=np.int64)
    for s in range(0, len(X), chunk):
        sims = Xt[s : s + chunk] @ Xt.T
        top = sims.topk(kmax + 1, dim=1)  # drop self (or a twin)
        dist[s : s + chunk] = (1.0 - top.values[:, 1:]).clamp_(min=0.0).cpu().numpy()
        idx[s : s + chunk] = top.indices[:, 1:].cpu().numpy()
    del Xt
    torch.cuda.empty_cache()
    print(f"knn (k<={kmax}) in {time.time() - t0:.0f}s")
    return dist, idx


def bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's coefficient; > 0.555 suggests more than one mode."""
    n = len(x)
    g = stats.skew(x)
    k = stats.kurtosis(x)
    return float((g**2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def characterise_density(d: np.ndarray, outlier: np.ndarray) -> dict:
    nonzero = d > 0
    logd = np.log10(np.clip(d, LOG_FLOOR, None))
    x = logd[nonzero].reshape(-1, 1)
    gmm = {}
    for n in (1, 2, 3):
        g = GaussianMixture(n, random_state=0, n_init=3).fit(x)
        gmm[n] = {
            "bic": float(g.bic(x)),
            "means": [float(10**m) for m in g.means_.ravel()],
            "weights": [float(w) for w in g.weights_],
        }
    q = lambda a: {p: float(np.percentile(a, p)) for p in (5, 25, 50, 75, 95)}
    out_d, in_d = d[outlier], d[~outlier]
    mw = stats.mannwhitneyu(out_d, in_d, alternative="greater")
    return {
        "n": len(d),
        "zero_fraction": float(1 - nonzero.mean()),
        "quantiles_all": q(d),
        "quantiles_outlier": q(out_d),
        "quantiles_clustered": q(in_d),
        "bimodality_coefficient_logd": bimodality_coefficient(logd[nonzero]),
        "gmm_logd": gmm,
        # P(random outlier is sparser than random clustered point)
        "auc_outlier_sparser": float(mw.statistic / (len(out_d) * len(in_d))),
        "outliers_denser_than_clustered_median": float(
            (out_d < np.median(in_d)).mean()
        ),
        "clustered_sparser_than_outlier_median": float(
            (in_d > np.median(out_d)).mean()
        ),
    }


def density_figure(dk: dict[int, np.ndarray], outlier: np.ndarray) -> None:
    fig, axes = plt.subplots(1, len(KS), figsize=(7.2, 2.8), sharey=True)
    for ax, k in zip(axes, KS):
        logd = np.log10(np.clip(dk[k], LOG_FLOOR, None))
        bins = np.linspace(logd.min(), logd.max(), 80)
        for mask, colour, name in (
            (~outlier, BLUE, "clustered"),
            (outlier, ORANGE, "outlier"),
        ):
            ax.hist(
                logd[mask],
                bins=bins,
                histtype="step",
                color=colour,
                label=f"{name} (n={mask.sum():,})",
            )
        ax.set_xlabel(f"log10 cosine distance to {k}th neighbour")
        ax.set_title(f"k = {k}", loc="left")
    axes[0].set_ylabel("embeddings")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Local density, by stored layer-0 label", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUTDIR / "density.pdf")
    fig.savefig(OUTDIR / "density.png")
    plt.close(fig)


# ---------------------------------------------------------------- sweep


def sweep_configs() -> list[dict]:
    configs = [dict(PRODUCTION)]
    for m in MIN_SIZES:
        c = {"base_min_cluster_size": m, "noise_level": PRODUCTION["noise_level"]}
        if c not in configs:
            configs.append(c)
    for nl in NOISE_LEVELS:
        c = {
            "base_min_cluster_size": PRODUCTION["base_min_cluster_size"],
            "noise_level": nl,
        }
        if c not in configs:
            configs.append(c)
    return configs


def config_key(c: dict) -> str:
    return f"m{c['base_min_cluster_size']}_n{c['noise_level']}"


def run_sweep(X: np.ndarray, stored0: np.ndarray, dk: np.ndarray) -> dict:
    ckpt = json.loads(SWEEP_CKPT.read_text()) if SWEEP_CKPT.exists() else {}
    stored_outlier = stored0 == -1
    for c in sweep_configs():
        key = config_key(c)
        if key in ckpt:
            continue
        t0 = time.time()
        clusterer = evoc.EVoC(
            noise_level=c["noise_level"],
            base_min_cluster_size=c["base_min_cluster_size"],
            min_samples=5,
            random_state=42,
        ).fit(X)
        layers = clusterer.cluster_layers_
        elapsed = time.time() - t0
        per_layer = []
        for labels in layers:
            out = labels == -1
            per_layer.append(
                {
                    "n_clusters": len(set(labels.tolist()) - {-1}),
                    "outlier_fraction": float(out.mean()),
                    # stored layer-0 outliers this configuration assigns
                    "stored_outliers_absorbed": float(
                        (~out & stored_outlier).sum() / stored_outlier.sum()
                    ),
                    "median_knn_absorbed": float(np.median(dk[~out & stored_outlier]))
                    if (~out & stored_outlier).any()
                    else None,
                    "median_knn_outlier": float(np.median(dk[out]))
                    if out.any()
                    else None,
                    "median_knn_clustered": float(np.median(dk[~out]))
                    if (~out).any()
                    else None,
                }
            )
        ckpt[key] = {
            **c,
            "seconds": elapsed,
            "n_layers": len(layers),
            "ari_layer0_vs_stored": float(adjusted_rand_score(stored0, layers[0])),
            "layers": per_layer,
        }
        np.save(OUTDIR / f"labels_{key}.npy", np.stack(layers).astype(np.int32))
        SWEEP_CKPT.write_text(json.dumps(ckpt, indent=1))
        print(
            f"evoc {key}: {len(layers)} layers, layer-0 outliers "
            f"{per_layer[0]['outlier_fraction']:.1%}, {elapsed:.0f}s",
            flush=True,
        )
    return ckpt


def sweep_figure(sweep: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    nl = PRODUCTION["noise_level"]
    m0 = PRODUCTION["base_min_cluster_size"]

    def series(rows, xkey):
        rows = sorted(rows, key=lambda r: r[xkey])
        xs = [r[xkey] for r in rows]
        finest = [r["layers"][0]["outlier_fraction"] for r in rows]
        coarsest = [r["layers"][-1]["outlier_fraction"] for r in rows]
        return xs, finest, coarsest

    by_m = [r for r in sweep.values() if r["noise_level"] == nl]
    xs, finest, coarsest = series(by_m, "base_min_cluster_size")
    axes[0].plot(xs, finest, "o-", color=BLUE, label="finest layer", ms=4)
    axes[0].plot(xs, coarsest, "o-", color=ORANGE, label="coarsest layer", ms=4)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(f"base_min_cluster_size (noise_level = {nl})")
    axes[0].axvline(m0, color="#c3c2b7", lw=0.8, ls="--")

    by_n = [r for r in sweep.values() if r["base_min_cluster_size"] == m0]
    xs, finest, coarsest = series(by_n, "noise_level")
    axes[1].plot(xs, finest, "o-", color=BLUE, ms=4)
    axes[1].plot(xs, coarsest, "o-", color=ORANGE, ms=4)
    axes[1].set_xlabel(f"noise_level (base_min_cluster_size = {m0})")
    axes[1].axvline(nl, color="#c3c2b7", lw=0.8, ls="--")

    axes[0].set_ylabel("outlier fraction")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    fig.suptitle("Outlier fraction across EVoC hyperparameters", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUTDIR / "sweep.pdf")
    fig.savefig(OUTDIR / "sweep.png")
    plt.close(fig)


# ---------------------------------------------------------------- length & position


def rate_by_bins(outlier, values, edges) -> list[dict]:
    rows = []
    for lo, hi in itertools.pairwise(edges):
        m = (values >= lo) & (values < hi)
        if m.sum():
            rows.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": int(m.sum()),
                    "outlier_rate": float(outlier[m].mean()),
                }
            )
    return rows


def length_analysis(data: dict, outlier: np.ndarray, logd: np.ndarray) -> dict:
    words, captioner = data["words"], data["captioner"]
    deciles = np.percentile(words, np.linspace(0, 100, 11))
    deciles[-1] += 1
    per_captioner = {}
    for name in sorted(set(captioner.tolist())):
        m = captioner == name
        terciles = np.percentile(words[m], [0, 33.3, 66.7, 100])
        terciles[-1] += 1
        per_captioner[name] = {
            "n": int(m.sum()),
            "mean_words": float(words[m].mean()),
            "outlier_rate": float(outlier[m].mean()),
            "median_log10_knn": float(np.median(logd[m])),
            "by_length_tercile": rate_by_bins(outlier[m], words[m], terciles),
            "by_global_decile": rate_by_bins(outlier[m], words[m], deciles),
        }

    # outlier ~ density + length + captioner: does length survive density?
    names = sorted(set(captioner.tolist()))
    z = lambda a: (a - a.mean()) / a.std()
    cols = [z(logd), z(np.log10(words + 1))]
    cols += [(captioner == n).astype(float) for n in names[1:]]
    A = np.column_stack(cols)
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(A, outlier)
    coef = dict(
        zip(
            ["z_log10_knn", "z_log10_words"] + [f"captioner={n}" for n in names[1:]],
            lr.coef_[0].tolist(),
        )
    )
    lr_density_only = LogisticRegression(C=np.inf, max_iter=1000).fit(A[:, :1], outlier)
    return {
        "spearman_words_vs_log10_knn": float(stats.spearmanr(words, logd).statistic),
        "by_length_decile": rate_by_bins(outlier, words, deciles),
        "per_captioner": per_captioner,
        "logistic": {
            "reference_captioner": names[0],
            "coefficients": coef,
            "intercept": float(lr.intercept_[0]),
            "accuracy": float(lr.score(A, outlier)),
            "accuracy_density_only": float(lr_density_only.score(A[:, :1], outlier)),
            "accuracy_majority": float(max(outlier.mean(), 1 - outlier.mean())),
        },
    }


def rate_by_group(outlier: np.ndarray, groups: np.ndarray) -> dict:
    return {
        str(g): {
            "n": int((groups == g).sum()),
            "outlier_rate": float(outlier[groups == g].mean()),
        }
        for g in sorted(set(groups.tolist()))
    }


def second_pass(X: np.ndarray, data: dict, outlier: np.ndarray) -> dict:
    """EVoC over the outlier set alone, production parameters scaled to its size."""
    n = int(outlier.sum())
    t0 = time.time()
    clusterer = evoc.EVoC(
        noise_level=PRODUCTION["noise_level"],
        base_min_cluster_size=max(5, int(n * 0.001)),
        min_samples=5,
        random_state=42,
    ).fit(X[outlier])
    layers = clusterer.cluster_layers_
    run_codes = np.unique(data["run"][outlier], return_inverse=True)[1]
    per_layer = []
    for labels in layers:
        runs_per_cluster = [
            len(set(run_codes[labels == c].tolist()))
            for c in set(labels.tolist()) - {-1}
        ]
        per_layer.append(
            {
                "n_clusters": len(runs_per_cluster),
                "outlier_fraction": float((labels == -1).mean()),
                "median_runs_per_cluster": float(np.median(runs_per_cluster))
                if runs_per_cluster
                else None,
            }
        )
    return {"n": n, "seconds": time.time() - t0, "layers": per_layer}


def position_analysis(data: dict, outlier: np.ndarray) -> dict:
    seq, max_length = data["seq"], data["max_length"]
    edges = [b[0] for b in SEQ_BINS] + [SEQ_BINS[-1][1]]
    by_horizon = {}
    for L in sorted(set(max_length.tolist())):
        m = max_length == L
        by_horizon[str(L)] = rate_by_bins(
            outlier[m], seq[m], [e for e in edges if e <= L]
        )
    return {"all": rate_by_bins(outlier, seq, edges), "by_max_length": by_horizon}


def length_position_figure(length: dict, position: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    ax = axes[0]
    for colour, (name, row) in zip(SERIES, length["per_captioner"].items()):
        pts = [r for r in row["by_global_decile"] if r["n"] >= 200]
        xs = [(r["lo"] + r["hi"]) / 2 for r in pts]
        ax.plot(
            xs, [r["outlier_rate"] for r in pts], "o-", color=colour, ms=3, label=name
        )
    ax.set_xlabel("caption length (words, global decile midpoints)")
    ax.set_ylabel("outlier rate (stored layer 0)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    for colour, (L, rows) in zip(SERIES, position["by_max_length"].items()):
        xs = [(r["lo"] + r["hi"]) / 2 for r in rows]
        ax.plot(
            xs,
            [r["outlier_rate"] for r in rows],
            "o-",
            color=colour,
            ms=3,
            label=f"{L}-step runs",
        )
    ax.set_xscale("log")
    ax.set_xlabel("trajectory step (bin midpoints)")
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Outlier rate by caption length and by trajectory position", x=0.01, ha="left"
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "length_position.pdf")
    fig.savefig(OUTDIR / "length_position.png")
    plt.close(fig)


# ---------------------------------------------------------------- structure


def neighbour_composition(idx: np.ndarray, outlier: np.ndarray, k: int) -> dict:
    """Share of each point's k nearest neighbours that are outliers."""
    share = outlier[idx[:, :k]].mean(axis=1)
    return {
        "k": k,
        "base_rate": float(outlier.mean()),
        "mean_share_around_outliers": float(share[outlier].mean()),
        "mean_share_around_clustered": float(share[~outlier].mean()),
        # outliers whose neighbourhood is (almost) all outliers: a small dense
        # group EVoC did not label, rather than a point at a cluster's edge
        "outliers_with_all_outlier_neighbours": float((share[outlier] == 1).mean()),
        "outliers_with_majority_outlier_neighbours": float(
            (share[outlier] > 0.5).mean()
        ),
    }


def stretch_structure(data: dict, labels: np.ndarray) -> dict:
    """Maximal outlier stretches within each run, by what surrounds them."""
    order = np.lexsort((data["seq"], data["run"]))
    runs, labs = data["run"][order], labels[order]
    kinds = {"transit": [], "excursion": [], "start": [], "end": [], "whole": []}
    starts = np.flatnonzero(np.r_[True, runs[1:] != runs[:-1]])
    ends = np.r_[starts[1:], len(runs)]
    for a, b in zip(starts, ends):
        lab = labs[a:b]
        i = 0
        while i < len(lab):
            if lab[i] != -1:
                i += 1
                continue
            j = i
            while j < len(lab) and lab[j] == -1:
                j += 1
            before = lab[i - 1] if i > 0 else None
            after = lab[j] if j < len(lab) else None
            if before is None and after is None:
                kind = "whole"
            elif before is None:
                kind = "start"
            elif after is None:
                kind = "end"
            elif before == after:
                kind = "excursion"
            else:
                kind = "transit"
            kinds[kind].append(j - i)
            i = j
    n_stretches = sum(len(v) for v in kinds.values())
    n_outlier_steps = sum(sum(v) for v in kinds.values())
    summary = {}
    for kind, lengths in kinds.items():
        arr = np.array(lengths)
        summary[kind] = {
            "n_stretches": len(arr),
            "share_of_stretches": float(len(arr) / n_stretches),
            "share_of_outlier_steps": float(arr.sum() / n_outlier_steps)
            if len(arr)
            else 0.0,
            "median_length": float(np.median(arr)) if len(arr) else None,
            "mean_length": float(arr.mean()) if len(arr) else None,
            "length_1_share": float((arr == 1).mean()) if len(arr) else None,
        }
    # clustered stretches, for comparison: how long does a run stay in a cluster?
    stay = []
    for a, b in zip(starts, ends):
        lab = labs[a:b]
        i = 0
        while i < len(lab):
            if lab[i] == -1:
                i += 1
                continue
            j = i
            while j < len(lab) and lab[j] == lab[i]:
                j += 1
            stay.append(j - i)
            i = j
    stay = np.array(stay)
    return {
        "n_runs": len(starts),
        "outlier_stretches": summary,
        "clustered_stays": {
            "n": len(stay),
            "median_length": float(np.median(stay)),
            "mean_length": float(stay.mean()),
            "length_1_share": float((stay == 1).mean()),
        },
    }


# ---------------------------------------------------------------- coverage


def run_coverage(data: dict, idx: np.ndarray, labels: np.ndarray, k: int) -> dict:
    """Trajectory coverage of labelled clusters against outlier regions."""
    run = data["run"]
    run_codes = np.unique(run, return_inverse=True)[1]
    outlier = labels == -1

    same_run = (run_codes[idx[:, :k]] == run_codes[:, None]).mean(axis=1)

    def coverage_of(groups: np.ndarray, mask: np.ndarray) -> dict:
        rows = []
        for g in np.unique(groups[mask]):
            m = mask & (groups == g)
            counts = np.bincount(run_codes[m])
            rows.append(
                (int(m.sum()), int((counts > 0).sum()), float(counts.max() / m.sum()))
            )
        size = np.array([r[0] for r in rows])
        n_runs = np.array([r[1] for r in rows])
        purity = np.array([r[2] for r in rows])
        w = size / size.sum()
        return {
            "n_groups": len(rows),
            "n_points": int(size.sum()),
            "median_size": float(np.median(size)),
            "median_runs_per_group": float(np.median(n_runs)),
            "point_weighted_mean_runs_per_group": float((w * n_runs).sum()),
            "point_share_in_single_run_groups": float(w[n_runs == 1].sum()),
            "point_share_in_groups_of_5plus_runs": float(w[n_runs >= 5].sum()),
            "point_weighted_dominant_run_share": float((w * purity).sum()),
        }

    # connected components of the symmetrised k-NN graph restricted to outliers
    src = np.repeat(np.arange(len(labels)), k)
    dst = idx[:, :k].ravel()
    keep = outlier[src] & outlier[dst]
    n = len(labels)
    g = coo_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(n, n))
    _, comp = connected_components(g, directed=False)

    return {
        "k": k,
        "same_run_neighbour_share": {
            "outliers_mean": float(same_run[outlier].mean()),
            "clustered_mean": float(same_run[~outlier].mean()),
            "outliers_all_same_run": float((same_run[outlier] == 1).mean()),
            "clustered_all_same_run": float((same_run[~outlier] == 1).mean()),
            "outliers_majority_same_run": float((same_run[outlier] > 0.5).mean()),
            "clustered_majority_same_run": float((same_run[~outlier] > 0.5).mean()),
        },
        "clusters": coverage_of(labels, ~outlier),
        "outlier_components": coverage_of(comp, outlier),
        "_same_run": same_run,
    }


def structure_figure(coverage: dict, stretches: dict, outlier: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    same_run = coverage["_same_run"]
    bins = np.linspace(0, 1, 16)
    for mask, colour, name in (
        (~outlier, BLUE, "clustered"),
        (outlier, ORANGE, "outlier"),
    ):
        ax.hist(
            same_run[mask],
            bins=bins,
            histtype="step",
            color=colour,
            density=True,
            label=name,
        )
    ax.set_xlabel(f"share of {coverage['k']} nearest neighbours from the same run")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper center")

    ax = axes[1]
    kinds = ["transit", "excursion", "start", "end", "whole"]
    shares = [
        stretches["outlier_stretches"][k]["share_of_outlier_steps"] for k in kinds
    ]
    ax.barh(kinds[::-1], shares[::-1], color=ORANGE, height=0.6)
    for y, v in enumerate(shares[::-1]):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=8, color="#52514e")
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("share of outlier steps, by kind of stretch")
    ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Who an outlier's neighbours are, and where outlier time is spent",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "structure.pdf")
    fig.savefig(OUTDIR / "structure.png")
    plt.close(fig)


# ---------------------------------------------------------------- main


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    OUTDIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    data = load(con)
    con.close()
    X, stored = data["X"], data["stored"]
    outlier0 = stored[0] == -1

    stored_summary = [
        {
            "layer": int(layer),
            "n_clusters": int(stored[layer].max() + 1),
            "outlier_fraction": float((stored[layer] == -1).mean()),
        }
        for layer in range(len(stored))
    ]
    print("stored clustering:", stored_summary)

    dist, idx = knn(X, max(KS))
    dk = {k: dist[:, k - 1] for k in KS}
    np.save(OUTDIR / "knn_distances.npy", dist)
    density = {str(k): characterise_density(dk[k], outlier0) for k in KS}
    for k, d in density.items():
        print(
            f"k={k}: median d outlier {d['quantiles_outlier'][50]:.4f} vs clustered "
            f"{d['quantiles_clustered'][50]:.4f}, AUC {d['auc_outlier_sparser']:.3f}, "
            f"BC {d['bimodality_coefficient_logd']:.3f}, "
            f"BIC 1/2/3: {[round(d['gmm_logd'][n]['bic']) for n in (1, 2, 3)]}"
        )
    density_figure(dk, outlier0)

    logd15 = np.log10(np.clip(dk[15], LOG_FLOOR, None))
    length = length_analysis(data, outlier0, logd15)
    position = position_analysis(data, outlier0)
    length_position_figure(length, position)
    print("logistic:", json.dumps(length["logistic"], indent=1))

    structure = {
        "neighbours": {str(k): neighbour_composition(idx, outlier0, k) for k in KS},
        "stretches": stretch_structure(data, stored[0]),
    }
    structure["by_experiment"] = rate_by_group(outlier0, data["experiment"])
    structure["by_network"] = rate_by_group(outlier0, data["network"])
    structure["second_pass"] = second_pass(X, data, outlier0)
    coverage = run_coverage(data, idx, stored[0], 15)
    structure_figure(coverage, structure["stretches"], outlier0)
    structure["coverage"] = {k: v for k, v in coverage.items() if not k.startswith("_")}
    print("structure:", json.dumps(structure, indent=1))

    sweep = run_sweep(X, stored[0], dk[15])
    sweep_figure(sweep)

    OUT.write_text(
        json.dumps(
            {
                "db": str(DB),
                "embedding_model": EMBEDDING_MODEL,
                "n_embeddings": len(X),
                "stored_clustering": stored_summary,
                "density": density,
                "length": length,
                "position": position,
                "structure": structure,
                "sweep": sweep,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
