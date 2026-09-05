#!/usr/bin/env python
"""Do stored vectors reproduce, and pin the scale against a library upgrade.

Two jobs, both for TASK-96.

AC#3 --- a spot check that the re-embedding actually landed: sample stored
embeddings across every experiment, re-embed the same invocation text with the
current code, and report the cosine between stored and fresh. Before the
recompute this sat at 0.383 because the stored vectors were mean-pooled; it
should now be 1.0 to float32 rounding.

AC#6 --- writes `test/fixtures/qwen3embed_reference.json`, a handful of fixed
texts with their vectors, so the GPU test in `real_models_test.exs` can assert
that a future library upgrade has not silently moved the space again. The
geometry test alongside it catches a collapse; this catches any change at all.

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python analysis/embedding_reference.py

Results -> analysis/embedding_reference.json, fixture -> test/fixtures/
"""

import base64
import json
import pathlib
import sqlite3
import sys

import numpy as np

sys.path.insert(0, "/home/ben/projects/panic_tda/priv/python")
import panic_models as pm  # noqa: E402

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
DB = ROOT / "priv/panic_tda_dev.db"
OUT = HERE / "embedding_reference.json"
FIXTURE = ROOT / "test/fixtures/qwen3embed_reference.json"
MODEL = "Qwen3Embed"
PER_EXPERIMENT = 20

# Fixed texts for the regression fixture. Short, stable, and spanning the
# register the loop actually produces, so a change in any of them is a change
# in the embedding path rather than in one odd input.
REFERENCE_TEXTS = [
    "A black bicycle leans against a red brick wall.",
    "The stock market closed lower after a volatile trading session.",
    "A photograph of a cat sitting on a windowsill in afternoon light, "
    "its fur backlit, with a potted plant just visible behind the glass.",
]


def sample_rows() -> list[tuple[str, str, bytes, str]]:
    con = sqlite3.connect(DB)
    rows = []
    for (experiment,) in con.execute("select id from experiments order by id"):
        rows += con.execute(
            """
            select ?, e.id, e.vector, i.output_text
            from embeddings e
            join invocations i on i.id = e.invocation_id
            join runs r on r.id = i.run_id
            where r.experiment_id = ? and e.embedding_model = ?
              and i.output_text is not null
            limit ?
            """,
            (experiment, experiment, MODEL, PER_EXPERIMENT),
        ).fetchall()
    con.close()
    return rows


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    rows = sample_rows()
    print(
        f"{len(rows)} sampled embeddings from "
        f"{len(set(r[0] for r in rows))} experiments",
        flush=True,
    )

    pm.setup()
    pm.load_model(MODEL)
    pm.swap_to_gpu(MODEL)
    fresh = [
        np.frombuffer(base64.b64decode(b), dtype=np.float32)
        for b in pm.embed_text(MODEL, [r[3] for r in rows])
    ]
    reference = [base64.b64decode(b) for b in pm.embed_text(MODEL, REFERENCE_TEXTS)]
    pm.unload_model(MODEL)

    per_experiment: dict[str, list[float]] = {}
    for (experiment, _id, blob, _text), new in zip(rows, fresh):
        stored = np.frombuffer(blob, dtype=np.float32)
        per_experiment.setdefault(experiment, []).append(cos(stored, new))

    summary = {
        "model": MODEL,
        "n_sampled": len(rows),
        "dimension": len(fresh[0]),
        "per_experiment": {
            e: {
                "n": len(v),
                "mean_cos_stored_vs_fresh": float(np.mean(v)),
                "min_cos_stored_vs_fresh": float(min(v)),
            }
            for e, v in sorted(per_experiment.items())
        },
    }
    every = [c for v in per_experiment.values() for c in v]
    summary["mean_cos_stored_vs_fresh"] = float(np.mean(every))
    summary["min_cos_stored_vs_fresh"] = float(min(every))

    for e, s in summary["per_experiment"].items():
        print(
            f"  {e[:8]}  n={s['n']:3d}  mean {s['mean_cos_stored_vs_fresh']:.6f}  "
            f"min {s['min_cos_stored_vs_fresh']:.6f}",
            flush=True,
        )
    print(
        f"\noverall mean {summary['mean_cos_stored_vs_fresh']:.6f}  "
        f"min {summary['min_cos_stored_vs_fresh']:.6f}",
        flush=True,
    )

    OUT.write_text(json.dumps(summary, indent=2) + "\n")

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(
            {
                "model": MODEL,
                "dimension": len(fresh[0]),
                "note": "float32 little-endian, base64. Regenerate with "
                "analysis/embedding_reference.py only when the "
                "embedding path is deliberately changed.",
                "vectors": [
                    {"text": t, "vector_b64": base64.b64encode(v).decode("ascii")}
                    for t, v in zip(REFERENCE_TEXTS, reference)
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT} and {FIXTURE}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
