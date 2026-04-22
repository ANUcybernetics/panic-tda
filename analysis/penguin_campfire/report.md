# Paraphrase-FTLE analysis: a penguin sitting by a campfire / a penguin sitting alongside a campfire / a penguin sitting by a fire / a penguin sitting beside a campfire / a penguin sitting next to a campfire

Experiment: `019d2ec7-2b02-7658-a8fe-4af8be31d75e`
Networks: [["SD35Medium", "Moondream"], ["SD35Medium", "Qwen25VL"], ["SD35Medium", "Pixtral"], ["Flux2Klein", "Moondream"], ["Flux2Klein", "Qwen25VL"], ["Flux2Klein", "Pixtral"], ["GLMImage", "Moondream"], ["GLMImage", "Qwen25VL"], ["GLMImage", "Pixtral"]]
Embedding models: ["Nomic", "Qwen3Embed"]
Num runs per (network, prompt): 8
Max length: 200

## Setup

TODO: 150 words on dynamical-systems framing, config, and FTLE definition.

## Identical-prompt baseline

TODO: characterise distribution (median, spread, outliers).

## Paraphrase FTLEs

TODO: characterise distribution.

## Comparison

[FTLE grid (PDF)](ftle_grid.pdf)

TODO: one sentence per network row on separation between identical and paraphrase FTLE.

## Qualitative divergence

Representative cell: **SD35Medium|Pixtral  ·  Qwen3Embed**.

[Divergence curves (PDF)](divergence_curves.pdf)

TODO: one paragraph on what the two curves show.

## Interpretation

TODO: does within-category spread < between-category gap? 200 words, honest either way.

## Proposed next wave (5-category controlled perturbation)

TODO: 300 words. Minimum viable (physics violation only + matched controls, reuse 9-network grid) vs full sweep. Open questions for Sungyeon:

- Who writes the violating and control prompts?
- Cut the 9-network grid down to 3 to afford more prompts per category?
- Add paraphrases of each violating prompt too (nested design)?
