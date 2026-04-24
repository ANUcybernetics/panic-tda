---
id: TASK-73
title: >-
  Evaluate bounded-space alternatives to classical FTLE for genAI
  trajectory analysis
status: To Do
assignee: []
created_date: '2026-04-24 10:00'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-71. The paraphrase-FTLE analysis (`analysis/penguin_campfire/`,
`analysis/pilot_3cat/`, `analysis/comparison/`) showed that classical
finite-time Lyapunov exponents don't cleanly fit this data: our embeddings
are L2-normalised onto the unit sphere, so pairwise distances are bounded
in [0, 2] and can't grow unboundedly. Key empirical findings:

- Log-linear fits have median R² ≈ 0.1-0.5 — not close to the straight
  lines classical FTLE assumes
- Distant-prompt FTLEs come out *smaller* than identical-prompt FTLEs
  (opposite of chaotic intuition) because the distant pairs start already
  near saturation with no room to grow
- "Paraphrase FTLE ≈ identical FTLE" (the outcome Sungyeon predicted as
  "interesting") holds, but likely trivially — both regimes are
  saturation-limited, not genuinely insensitive to perturbation

The trajectory system looks much more like a **bounded diffusion / Markov
chain on a compact semantic space** than a classical chaotic system with
exponential sensitivity to initial conditions. This task is to evaluate
which bounded-space-appropriate statistic(s) should replace or supplement
FTLE in future work, especially before committing to the full 5-category
controlled-perturbation sweep.

## Candidate statistics to evaluate

Ordered by expected payoff vs implementation cost:

1. **Transition-matrix Markov-chain framework on existing HDBSCAN clusters.**
   Cluster transitions along trajectories → empirical transition matrix per
   (network, embedding) cell. From the matrix: stationary distribution,
   mixing time via spectral gap, detailed-balance violation, absorbing
   classes (= attractor detection). Largely free — the clustering stage
   already runs. Fits Sungyeon's "AI as dynamical system" framing
   naturally. **Most promising.**

2. **Mixing time τ via exponential-approach fit.**
   Fit `d(t) ≈ d_∞ - (d_∞ - d_0) · exp(-t/τ)` to the same divergence
   curves we already compute. τ is the characteristic relaxation time to
   the stationary distribution. Single defensible number per cell, drops
   in as a direct replacement for the FTLE column. Respects the bounded
   space. Cheap — just swap the curve-fitting function in
   `priv/python/penguin_analysis.py`.

3. **Prompt-recoverability / information-decay curves.**
   Instead of "does perturbation grow?", ask "does initial-prompt
   information survive the mixing?" Train a cheap classifier to recover
   the initial prompt from the embedding at step t, plot accuracy vs t.
   The decay timescale is the practically-relevant "sensitivity"
   measurement, and it's well-defined on bounded spaces.

4. **Early-time slope (salvage FTLE).**
   Fit the log-linear slope only for the first N steps before saturation
   kicks in. Gives a defensible "local sensitivity" number. Cheaper than
   reworking the whole framework, but still prompt-pair-dependent and
   doesn't address the deeper issue that distant pairs have no early
   window.

5. **Absorbing-state / attractor detection.**
   Diagnostic rather than headline statistic: measure the fraction of
   trajectories that collapse to fixed points (images that loop, text
   that stabilises), time-to-absorption distributions, attractor-set
   geometry. Tells us which networks have "black holes" in their
   dynamics.

## Scope

This task is **evaluation / decision**, not full implementation. Output
should be:

- A short memo comparing (1)-(5) on the existing `penguin_campfire` +
  `pilot_3cat` data, with figures where cheap
- A recommendation for which one (or combination) to adopt as the
  primary statistic for future experiments
- An updated framing for the 5-category controlled-perturbation
  experiment (which this implicitly blocks)

Probably do (1) and (2) concretely on existing data before deciding;
(3)-(5) described in the memo but not implemented unless the first two
are uninformative.

## Not in scope

- Running any new GPU experiments
- The full 5-category controlled-perturbation sweep (separate follow-up
  task, blocked on this one)
- TDA of the stationary distribution — Sungyeon's domain, see her
  email / PKB 741

## References

- Prior analysis: `analysis/penguin_campfire/`, `analysis/pilot_3cat/`,
  `analysis/comparison/`
- Predecessor task: TASK-71
- Spec: `docs/superpowers/specs/2026-04-21-penguin-campfire-paraphrase-ftle-analysis-design.md`
- Plan: `docs/superpowers/plans/2026-04-22-penguin-campfire-paraphrase-ftle-analysis.md`
- Sungyeon's original framing: PKB note 741
<!-- SECTION:DESCRIPTION:END -->
