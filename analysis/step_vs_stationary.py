#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0"]
# ///
"""How much of the stationary step is generator sampling noise? (TASK-89 AC#2/#5)

`step_decomposition.py` measured, per text-to-image model, how far one caption's
successors travel (step), where their centroid sits (drift) and how far they
scatter around it (noise). That was measured at captions taken from pilot
images, not from a trajectory, so the question it cannot answer alone is what
those numbers mean against the motion an actual run shows once it has settled.

This joins the sweep to the two trajectory measurements. All three are now on
the same embedding scale --- TASK-96 re-embedded the database with last-token
pooling, so the comparison is finally like-for-like:

  sweep        analysis/step_decomposition.json   drift/noise per generator
  200-step     analysis/long_horizon_baseline.json  plateau, old lineup
  pilot/panel  analysis/pilot_vs_panel.json       50-step v2-era arms

The ratio that matters is noise / stationary step: the share of a settled run's
step-to-step motion that one draw of the generator's seed would produce on its
own, with no dynamics at all.

Two mismatches keep this indicative rather than exact, and both are why TASK-90
must re-measure the noise floor from its own trajectories.

The noise term travels through a captioner --- it is the spread of the CAPTIONS
of N images, so it depends on which captioner read them. The sweep used the v2
lineup, greedy; the trajectories here used Moondream, Pixtral, Qwen25VL and
Gemma3n at the old truncating ceilings, and caption length moves step size on
its own (TASK-85). Second, the sweep's source captions come from pilot images
rather than from each network's own stationary regime.

So a ratio at or slightly above 100% does not mean noise exceeds the motion,
which is impossible for one system; it means the two are the same size to
within the mismatch. Read the ratios as an order-of-magnitude claim.

    ./analysis/step_vs_stationary.py

Results -> analysis/step_vs_stationary.json, table to stdout.
"""

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
OUT = HERE / "step_vs_stationary.json"
LATE_BIN = "150-200"


def main() -> None:
    sweep = json.loads((HERE / "step_decomposition.json").read_text())["dims"]["256"]
    baseline = json.loads((HERE / "long_horizon_baseline.json").read_text())["plateau"]
    arms = json.loads((HERE / "pilot_vs_panel.json").read_text())["drift"]

    models = sweep["models"]
    ruler = sweep["ruler"]

    # the settled step size each measurement reports
    late = {
        network: s["median_step_distance"][LATE_BIN] for network, s in baseline.items()
    }
    stationary = {
        "200-step baseline (late window)": (min(late.values()), max(late.values())),
        "50-step panel arm": (arms["panel"]["step_to_step_late_mean"],) * 2,
        "50-step pilot": (arms["pilot"]["step_to_step_late_mean"],) * 2,
    }

    noise = {m: v["mean_noise_cosdist"] for m, v in models.items()}
    step = {m: v["mean_step_cosdist"] for m, v in models.items()}

    # the generator each trajectory measurement used, where there is one
    matched = {
        "50-step pilot": "Flux2Klein",
        "50-step panel arm": "Flux2Klein",
    }
    for network in late:
        for m in noise:
            if network.strip('[]"').split('","')[0] == m:
                matched[f"200-step {network}"] = m

    rows = []
    for label, generator in sorted(matched.items()):
        if label.startswith("200-step"):
            network = label[len("200-step ") :]
            settled = late[network]
        else:
            settled = stationary[label][0]
        rows.append(
            {
                "trajectory": label,
                "generator": generator,
                "stationary_step": settled,
                "sweep_step": step[generator],
                "sweep_noise": noise[generator],
                "noise_share_of_stationary_step": noise[generator] / settled,
            }
        )

    print("sweep, per generator (fixed caption, 16 seeds)")
    print(f"{'model':13s} {'step':>7s} {'drift':>7s} {'noise':>7s}  noise share of step")
    for m, v in models.items():
        print(
            f"{m:13s} {v['mean_step_cosdist']:7.4f} {v['mean_drift_cosdist']:7.4f} "
            f"{v['mean_noise_cosdist']:7.4f}  {v['mean_noise_share']:.1%}"
        )

    print("\nagainst the settled step size of an actual trajectory")
    print(f"{'trajectory':44s} {'gen':12s} {'settled':>8s} {'noise':>7s}  share")
    for r in rows:
        print(
            f"{r['trajectory']:44s} {r['generator']:12s} {r['stationary_step']:8.4f} "
            f"{r['sweep_noise']:7.4f}  {r['noise_share_of_stationary_step']:.0%}"
        )

    shares = [r["noise_share_of_stationary_step"] for r in rows]
    summary = {
        "late_bin": LATE_BIN,
        "ruler": ruler,
        "sweep": {
            m: {k: v[k] for k in ("mean_step_cosdist", "mean_drift_cosdist",
                                  "mean_noise_cosdist", "mean_noise_share")}
            for m, v in models.items()
        },
        "rows": rows,
        "noise_share_of_stationary_step": {
            "min": float(min(shares)),
            "max": float(max(shares)),
            "median": float(np.median(shares)),
        },
        "caveat": (
            "noise is measured through the v2 captioners, greedy; these "
            "trajectories used the old lineup at truncating ceilings, and the "
            "sweep's source captions are not from each network's stationary "
            "regime. Ratios near or above 100% mean 'the same size', not "
            "'larger'."
        ),
        "ruler_check": {
            "seed_resample_cosdist": ruler["seed_resample_cosdist_mean"],
            "unrelated_caption_cosdist": ruler["unrelated_caption_cosdist_mean"],
            "stationary_step_range": [min(late.values()), max(late.values())],
        },
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"\nnoise accounts for {min(shares):.0%}-{max(shares):.0%} of the settled step "
        f"(median {np.median(shares):.0%})"
    )
    print(
        f"\nruler: two seed resamples of one caption sit {ruler['seed_resample_cosdist_mean']:.4f} apart, "
        f"captions of unrelated prompts {ruler['unrelated_caption_cosdist_mean']:.4f}."
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
