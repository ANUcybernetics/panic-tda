# Paraphrase-FTLE analysis: A penguin enjoying a wood fire / A big mouse hunting for a small cat / A crocodile having a salad

Experiment: `019cb1be-f312-76ce-97a7-903d879f07b4`
Networks: [["SD35Medium", "Moondream"], ["Flux2Klein", "Qwen25VL"], ["ZImageTurbo", "Gemma3n"], ["QwenImage", "Pixtral"], ["GLMImage", "LLaMA32Vision"], ["SD35Medium", "Pixtral"], ["Flux2Klein", "Moondream"], ["ZImageTurbo", "LLaMA32Vision"], ["QwenImage", "Qwen25VL"], ["GLMImage", "Gemma3n"]]
Embedding models: ["Nomic", "Qwen3Embed"]
Num runs per (network, prompt): 4
Max length: 100

## Setup

TODO: 150 words on dynamical-systems framing, config, and FTLE definition.

## Identical-prompt baseline

TODO: characterise distribution (median, spread, outliers).

## Paraphrase FTLEs

TODO: characterise distribution.

## Comparison

[FTLE heatmap (PDF)](ftle_heatmap.pdf)

TODO: one sentence per network row on separation between identical and paraphrase FTLE.

## Qualitative divergence

Representative cell: **QwenImage|Pixtral  ·  Qwen3Embed**.

[Divergence curves (PDF)](divergence_curves.pdf)

TODO: one paragraph on what the two curves show.

## Interpretation

TODO: does within-category spread < between-category gap? 200 words, honest either way.

## Proposed next wave (5-category controlled perturbation)

TODO: 300 words. Minimum viable (physics violation only + matched controls, reuse 9-network grid) vs full sweep. Open questions for Sungyeon:

- Who writes the violating and control prompts?
- Cut the 9-network grid down to 3 to afford more prompts per category?
- Add paraphrases of each violating prompt too (nested design)?
