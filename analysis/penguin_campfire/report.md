# Lyapunov-style divergence analysis: a penguin sitting by a campfire / a penguin sitting alongside a campfire / a penguin sitting by a fire / a penguin sitting beside a campfire / a penguin sitting next to a campfire

Experiment: `019d2ec7-2b02-7658-a8fe-4af8be31d75e`
Networks: [["SD35Medium", "Moondream"], ["SD35Medium", "Qwen25VL"], ["SD35Medium", "Pixtral"], ["Flux2Klein", "Moondream"], ["Flux2Klein", "Qwen25VL"], ["Flux2Klein", "Pixtral"], ["GLMImage", "Moondream"], ["GLMImage", "Qwen25VL"], ["GLMImage", "Pixtral"]]
Embedding models: ["Nomic", "Qwen3Embed"] (L2-normalised onto the unit sphere)
Num runs per (network, prompt): 8
Max length: 200 invocations (text embeddings ≈ half this)

## Data

TODO: 2-3 sentences on what this experiment contains and how the prompt
set was chosen (paraphrase cluster, distant-topic set, controlled
perturbation category, etc.).

## Method and a caveat on classical FTLE

For each prompt (or prompt pair) we compute the mean pairwise Euclidean
distance between trajectories at each invocation step, take the natural
log, and fit a line. The slope is the finite-time Lyapunov exponent.

Our embeddings are L2-normalised onto the unit sphere, so pairwise
Euclidean distance is bounded in [0, 2]. Classical Lyapunov analysis
assumes unbounded exponential growth of initial separation, so the
slope we measure is really the *escape rate* from the starting
separation before the trajectories saturate against the bound — not
true sensitivity to initial conditions. This matters for interpretation
below.

## Heatmap

[FTLE heatmap — per-network prompt × prompt matrices (PDF)](ftle_heatmap.pdf)

Diagonal cells = within-prompt FTLE (stochastic-noise divergence).
Off-diagonal cells = cross-prompt FTLE (paraphrase or distant-prompt
pairs, depending on the experiment). Matrix is symmetric.

TODO: 2-3 sentences characterising the pattern (are diagonal cells
consistently brighter/darker than off-diagonal? do networks differ?).

## Qualitative divergence curve

Representative cell: **SD35Medium|Pixtral  ·  Qwen3Embed**.

[Divergence curves (PDF)](divergence_curves.pdf)

TODO: 1 paragraph — do the curves show early-time linear growth in
log-space, or do they saturate quickly? How well does the log-linear
fit actually describe the data?

## Fit quality

TODO: report median R² for identical rows and for paraphrase rows. If
R² is consistently low, that's evidence that the log-linear fit is
struggling — likely because distances saturate before exponential
growth can establish itself.

## Interpretation

TODO: 200 words. Given the bounded-space caveat:

- If diagonal (identical-prompt) FTLEs are similar to off-diagonal
  (paraphrase) FTLEs, that could be a genuine invariance OR a trivial
  artefact — both regimes are saturation-limited from near-zero
  separation.
- If distant-prompt FTLEs are *smaller* than identical-prompt FTLEs,
  that's consistent with the distant pairs already starting near
  saturation with no room to grow.
- A classical-FTLE-style "sensitivity-to-initial-conditions" story is
  hard to support directly; the useful signal may be the *shape* of
  the divergence curve (early-time slope, saturation time,
  stationary distance), not the whole-trajectory slope.

## Proposed next moves

TODO: 200-300 words. Options:

1. Keep classical FTLE but report only the early-time slope (first
   10-30 steps, before saturation) — gives a more defensible chaotic
   exponent.
2. Switch to a geometry-first statistic: stationary distance
   distribution, mixing time, or distance-autocorrelation — these
   respect the bounded space.
3. If pushing ahead with the 5-category experiment, decide on the
   statistic first, because classical FTLE on distant prompts
   saturates immediately (as this data shows).
4. Orthogonal: compare across embedding models / dimensions to see
   whether the bounded-space issue is geometry-specific.
