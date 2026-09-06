# SMC 2025 dataset inventory

What survives of the dataset behind Swift & Hong (2025), "Semantic topologies in
the recursive application of generative AI models," _IEEE SMC 2025_, 664–667
(doi:10.1109/SMC58881.2025.11342470).

First inventoried 2026-08-13 by Sungyeon Hong from a 5.1 GB partial copy
(`sungyeon.sqlite`, 144 runs, no longer on this machine). Re-inventoried
2026-09-07 from the full database, which is in this repository.

## Location and contents

`db/trajectory_data.sqlite` (124 GB; `db/backup/trajectory_data_2025_07_16.sqlite`
is a 140 GB earlier copy of the same database). This is the complete dataset:
the partial copy was experiment `067ed16c` alone.

| Table               | Rows      |
| ------------------- | --------- |
| `experimentconfig`  | 6         |
| `run`               | 3,092     |
| `invocation`        | 3,092,000 |
| `embedding`         | 6,184,000 |
| `persistencediagram`| 12,368    |
| `clusteringresult`  | 3         |
| `embeddingcluster`  | 328,800   |

Every run is a full 1,000 invocations (500 text states), on the four SMC
networks FluxSchnell/BLIP2, FluxSchnell/Moondream, SDXLTurbo/BLIP2 and
SDXLTurbo/Moondream. Embedding models: Nomic, STSBRoberta, STSBMpnet, plus
NomicVision on the images.

| experiment | ran           | prompts | repeats | runs  | prompt set                                      |
| ---------- | ------------- | ------- | ------- | ----- | ----------------------------------------------- |
| `067ed16c` | 2025-04-02/03 | 9       | 4       | 144   | fruit, vehicles, portrait photos                |
| `067ee281` | 2025-04-03/04 | 9       | 4       | 144   | coloured circle on coloured background          |
| `067f8931` | 2025-04-11/12 | 7       | 4       | 112   | single colour words                             |
| `067fcc93` | 2025-04-14/15 | 8       | 8       | 256   | animals                                         |
| `067feecb` | 2025-04-16/23 | 12      | 32      | 1,536 | picture/photo/portrait/painting of a man, woman, child |
| `06826b10` | 2025-05-16, unfinished | 25 | 12   | 900   | re-run of the first three prompt sets           |

The five April experiments are the paper's 45 prompts. The paper reports 720
runs, which is four repeats per prompt per network; the animal and portrait
experiments hold more repeats than that.

A second database, `db/length_5000_experiment.sqlite` (26 GB, experiment
`067efc98`, 2025-04-04), holds 128 runs of 5,000 invocations each: prompts
"yeah" and "nah", 16 repeats, the same four networks. It was never written up
and is the deepest trajectory data the project has, at 2,500 text states per
run.

## What the SMC runs actually were

Checked 2026-09-07 against the code at commit `407c044` and the stored
captions, because the fixes made since (decision-01, decision-02, TASK-96) had
to be ruled in or out for this data.

- **Unseeded.** The paper's methods say four random seeds per prompt; every
  experiment config has `seeds: [-1, ...]`, which meant `generator=None` and no
  seed stored. The repeats are unseeded draws, so no SMC step can be
  regenerated. The loop was a Markov chain with unrecorded noise, the regime
  TASK-93 now records.
- **Not truncated.** The `max_new_tokens` ceilings that cut four of five
  captioners in `balanced_panel_5x5` arrived with the February 2026 port. Over
  128,000 SMC captions, BLIP2 runs to at most 17 words against a 50-token
  ceiling and Moondream (`length="short"`) is never cut. Decision-01 does not
  affect this data; the captions are short because the models were, not
  because they were clipped.
- **Deterministic captioners.** BLIP2 ran beam search without sampling (its
  `do_sample` was tied to a seed being present) and Moondream's caption API
  decodes at temperature zero. Decision-02 does not affect this data.
- **Embedding recipe correct.** The Nomic path used the `clustering:` prefix,
  mean pooling, layer norm and L2 normalisation, which is the documented recipe
  for `nomic-embed-text-v1.5`. The TASK-96 pooling bug was in the Qwen3Embed
  path only. Note the paper names the model as Nomic Text v2; the code loaded
  v1.5.

Caption lengths (words):

| Model     | median | p99 | max |
| --------- | ------ | --- | --- |
| BLIP2     | 10     | 14  | 17  |
| Moondream | 22     | 31  | 181 |

## Using it

The schema predates the Elixir/Ash port: singular table names, uppercase
`type` values (`TEXT`, `IMAGE`), `output_text` and `output_image_data` columns
on `invocation`. Current `mix` tasks will not read it; query it directly with
sqlite3 or polars. The `invocation` table carries every image blob, so filter
by `run_id` (indexed) rather than scanning it.

FluxSchnell, SDXLTurbo and BLIP2 have been removed from
`priv/python/panic_models.py`, so these networks cannot be re-run. Moondream
has since changed weights, length mode and (under TASK-87) architecture, which
is why the cross-era comparison TASK-81 proposed was dropped: nothing spans
both eras. Re-embedding the stored captions with Qwen3Embed remains a small
job if a use appears.

Captioner verbosity has grown four to five fold in one model generation, from
10–22 words here to 45–290 in the v2 lineup (`caption-length-by-i2t-model.md`).
