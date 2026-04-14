---
id: TASK-71
title: >-
  Design sensitivity-to-initial-conditions experiment with semantic prompt
  variants
status: To Do
assignee: []
created_date: '2026-04-14 07:28'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design (and discuss) an experiment that uses the existing Lyapunov stage to
probe how sensitive trajectories are to different *kinds* of change in the
initial prompt. Context: email from Sungyeon (see PKB note 741) --- she wants
to frame this project as "study of AI as a dynamical system" (journals like
Chaos, Entropy), with sensitivity-to-initial-conditions as the hook. Results
promised by April 20.

The `penguin_campfire` experiment already covers the "synonyms / paraphrase"
case: 5 near-identical prompts × 9 networks × 8 runs × max_length 200 with
Nomic embeddings, and the FTLE slope is computed per (network, prompt) group
by `LyapunovStage`. This task is about what to run *next*.

## Research question

For a fixed network, how does the Lyapunov exponent depend on the *type* of
variation applied to the initial prompt? Cases to compare:

1. **identical prompt** (num_runs > 1, same seed-less stochasticity) ---
   lower-bound baseline from model noise alone
2. **semantic paraphrase** (already done: penguin_campfire) --- variants that
   a human reader would call synonymous
3. **controlled semantic perturbation** --- Sungyeon's categories:
   - physics violation ("penguin by a fire", "penguin wearing a thick coat")
   - size reversal ("big mouse and a small cat")
   - role inversion ("prey hunting predator")
   - diet violation ("crocodile eating vegetables")
   - imaginary objects ("fish with feet walking on a sand beach")

The question is whether (a) these categories produce systematically different
FTLE values, and (b) the spread *within* a category is smaller than the gap
*between* categories. Either outcome is interesting.

## Design questions to resolve before running

- **Prompt set size.** Do we want a single violating prompt per category and
  many paraphrases of it, or many distinct violating prompts per category and
  few paraphrases? The first gives us a within-category FTLE distribution; the
  second gives a between-violation-instance distribution. Probably we want
  both, but that's 5 categories × K distinct prompts × P paraphrases × R runs
  --- the multiplier gets expensive fast.

- **Control / non-violating counterpart.** Each violating prompt probably
  needs a matched non-violating counterpart ("penguin by a fire" vs "penguin
  by an iceberg") so we can attribute FTLE differences to the violation
  rather than to surface prompt content. Worth deciding whether counterpart
  prompts are in the same experiment (same networks, same max_length) or a
  separate one.

- **Pairing for FTLE.** `LyapunovStage` currently groups runs by
  `(network, initial_prompt)` and computes FTLE within that group. To compare
  *between* prompts (e.g. "penguin by fire" vs "penguin by iceberg") we'd
  need either (a) a new grouping key (category tag? paraphrase cluster?) or
  (b) post-hoc analysis that reads embeddings directly and computes pairwise
  divergence across prompts. Option (b) is probably faster; option (a) needs
  a schema change.

- **Networks to include.** `penguin_campfire` uses 9 networks (3 T2I × 3
  I2T). Do we keep the same grid, or cut it down (e.g. pick the 2-3 fastest
  combinations) so we can afford more prompt variants? At ~6s/invocation for
  SD35+Moondream × 200 steps × many prompts × 8 runs, cost adds up.

- **max_length and num_runs.** 200 steps matches existing data. FTLE is
  fitted as a single slope over the whole divergence curve; if divergence
  saturates early we might only need 50-100 steps. Worth sanity-checking
  against existing penguin_campfire data before committing.

- **Embedding model.** penguin_campfire used Nomic. For text-embedding the
  choice of model matters for FTLE magnitude (different geometries). Sungyeon's
  framing is geometry-agnostic, so either pick one and stick with it, or run
  2-3 embedding models on the same runs for robustness.

- **Reporting.** What's the minimum figure/table we'd need to send Sungyeon
  by April 20? Probably: one plot per network showing FTLE distribution for
  each category, with the identical-prompt and paraphrase baselines overlaid.

## Suggested first cut

Before a full 5-category sweep, do a minimum-viable version with one
category (say physics violation) × one matched control × existing
penguin_campfire network and runs-per-prompt config. If FTLE separates the
two, generalise to the other four categories; if it doesn't, revisit the
design before burning GPU time.

## Not in scope for this task

- Implementation of any new experiment config file or analysis code --- this
  task is design/discussion only
- Changes to `LyapunovStage` or `LyapunovResult` schema
- Decisions about publication venue
<!-- SECTION:DESCRIPTION:END -->
