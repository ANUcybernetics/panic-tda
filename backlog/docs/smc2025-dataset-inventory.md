# SMC 2025 dataset inventory

What survives of the dataset behind Swift & Hong (2025), "Semantic topologies in
the recursive application of generative AI models," *IEEE SMC 2025*, 664–667
(doi:10.1109/SMC58881.2025.11342470).

Inventoried 2026-08-13 by Sungyeon Hong.

## Location and contents

`projects/panic-tda_SMC2025/` (outside this repository):

- `sungyeon.sqlite` — 5.1 GB
- `persistence_diagrams.parquet` — 24 MB

Database contents:

| Table | Rows |
|---|---|
| `experimentconfig` | 1 |
| `run` | 144 |
| `invocation` | 144,000 |
| `embedding` | 216,000 |
| `persistencediagram` | 432 |
| `clusteringresult` | **0** |
| `embeddingcluster` | **0** |

Experiment configuration:

- Networks: FluxSchnell/BLIP2, FluxSchnell/Moondream, SDXLTurbo/BLIP2,
  SDXLTurbo/Moondream
- Prompts: 9 — an apple, a pear, a banana, a car, a train, a boat, and
  photorealistic portrait photos of a man, a woman and a child
- 4 unseeded repeats per prompt per network, giving 36 runs per network
- `max_length` 1000, so 500 text states per run
- Embedding models: Nomic, STSBRoberta, STSBMpnet
- Ran 2025-04-02 to 2025-04-03, about 15 hours

## This is a subset

The published paper reports **720 runs over 45 prompts**; this database holds
**144 runs over 9 prompts**, roughly a fifth. Missing in particular are the
systematic colour and shape control prompts (for example "a yellow circle on a
red background") that appear in the paper's figures. The paper's headline
numbers — the ~62% outlier rate and the 21-of-180 stationary trajectories —
cannot be reproduced from this subset.

**Open question for Ben Swift: does the full 720-run database still exist?**

## What is usable

1. **All 72,000 text outputs are present** in `invocation.output_text`, so the
   captions can be re-embedded with the current embedding model without
   regenerating any images. See TASK-81.
2. **All 72,000 images are present** in `invocation.output_image_data`, so
   image-side embeddings are also possible.
3. **Full 1000-iteration depth on all four networks** — 500 text states per
   run, against roughly 26 in `balanced_panel_5x5`. This is the deepest
   trajectory data available to the project.
4. **Clustering must be recomputed.** The `clusteringresult` and
   `embeddingcluster` tables are empty.

## Two caveats

**Schema.** This database predates the Elixir/Ash port. Tables are singular
(`run`, `invocation`, `experimentconfig`) and the `type` column uses uppercase
values (`TEXT`, `IMAGE`). Current `mix` tasks will not read it. For analysis,
query it directly with polars or pandas over SQL; migrate into the current
schema only if the Elixir pipeline stages are genuinely needed.

**Models.** FluxSchnell, SDXLTurbo and BLIP2 have all been removed from
`priv/python/panic_models.py`, so these networks cannot currently be re-run.

## Moondream as a cross-era anchor

Moondream is the only model appearing in both this dataset and the current
model panel. Caption lengths measured from both:

| Model | SMC 2025 median (p10–p90) | Current median (p10–p90) |
|---|---|---|
| BLIP2 | 10 (7–12) | not in registry |
| Moondream | 21 (17–26) | 24 (19–29) |

The two Moondream distributions are close enough to suggest the same
behavioural regime. Note the database records the model as `Moondream`; the
paper refers to "Moondream 2", which appears to be the product version of the
same registry entry. **Confirmation of the actual model weights is pending with
Ben Swift.**

If Moondream is unchanged, it is the only cross-era comparison that holds
caption length roughly fixed (21 to 24 words), so holding it fixed as the
captioner and varying only the image generator isolates the image-generator
effect. Every other cross-era pairing confounds model generation with caption
length — FluxSchnell/BLIP2 at 10 words against, say, Flux2Klein/Pixtral at 102
words.

Separately worth recording: captioner verbosity has grown roughly four to five
fold in one model generation, from 10–21 words to 24–106. See
`caption-length-by-i2t-model.md`.
