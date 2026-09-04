---
id: TASK-96
title: >-
  Stored embeddings do not reproduce, and are on a different scale from the
  current code
status: To Do
assignee: []
created_date: '2026-09-04 12:07'
labels:
  - analysis
  - instrument
  - embeddings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found 2026-09-04 while doing TASK-89's AC#6 truncation check, which re-embeds the caption pilot's captions and compares them with what is in the database.

THE OBSERVATION. Re-embedding a caption with the current code does not reproduce the vector stored for that same caption. Over the pilot's 2,000 captions the mean cosine between stored and freshly computed 256-d vectors is 0.383 (min 0.166). For five sampled captions the stored-vs-fresh cosine matrix has no dominant diagonal, so it is not a row misalignment: the two are simply different spaces.

THE SCALES DIFFER BY ROUGHLY FOUR TIMES, and inconsistently between experiments. Median cosine distance between captions of DIFFERENT initial prompts, from the stored vectors:

  019d2ec7 (200-step)  0.0557      019f3645 (balanced_panel_5x5)  0.2178
  01a060b4 (pilot)     0.1462      fresh, current code            0.6478

Within-run step-to-step distance, same data: stored 0.0101 to 0.0203, fresh 0.0430. Both spaces rank things the same way (step < same-prompt < different-prompt), so the stored vectors are not noise --- they are a compressed, anisotropic version of the same ordering. 019d2ec7's 0.056 for unrelated prompts is the extreme case and is hard to read as anything but degenerate.

WHAT IS NOT THE CAUSE. The current path is internally consistent: encoding one text alone and in a batch agree to float32 rounding, and the tokenizer loads with padding_side left, which is what Qwen3-Embedding's last-token pooling needs. Forcing right padding, and attempting mean pooling, did not reproduce the stored space either.

PRIME SUSPECT, UNPROVEN. Commit 659c1d9 (2026-09-03) moved the venv to sentence-transformers 6 and transformers 5.16, and renamed the loader's tokenizer_kwargs to processor_kwargs. Every experiment in the database was embedded before that; everything computed since is on the new scale. That timing fits, but it has not been demonstrated, and it does not explain why the three experiments are compressed by different amounts.

WHY IT MATTERS. Every published number that came from stored embeddings is on the old scale and cannot be compared with anything computed now:

- the plateau result in the research programme, 'median step-to-step distance falls to 0.009-0.02 and stays there' (analysis/long_horizon_baseline.py)
- TASK-85's 28% step-size reduction between the truncated panel and the natural-length pilot, 0.0162 against 0.0116 (analysis/pilot_vs_panel.py), and the cluster occupancy and transition figures alongside it
- the 'unrelated images sit near 0.876 cosine' ruler quoted in backlog/docs/caption-length-by-i2t-model.md and analysis/prompt_tail.py
- every EVoC clustering, since it ran on the stored vectors

Comparisons WITHIN one scale are probably still sound, because both arms used the same embeddings --- the 28% reduction is a ratio between two sets of stored vectors. Absolute values, cross-era comparisons, and anything comparing old numbers with new measurements are not.

It also explains why TASK-89's measured step of 0.062-0.083 looked five times too large against a 'stationary step size of 0.012-0.016'. It was not: the sweep is on the new scale and the baseline was on the old one. Re-measured with the current code the pilot's own step size is 0.043, which is the same order as the sweep.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Determined which embedding path is correct for Qwen3-Embedding-4B --- last-token pooling over a left-padded batch, with the instruction convention the model expects --- with evidence rather than by assuming the newer library is right
- [ ] #2 Cause identified, or explicitly recorded as not identified, including why the three experiments are compressed by different amounts
- [ ] #3 Every experiment re-embedded with the correct path, so the database holds one scale, and a spot check confirms re-embedding a caption now reproduces its stored vector
- [ ] #4 Clustering recomputed after re-embedding, since EVoC ran on the old vectors
- [ ] #5 Every number in the programme, the caption-length doc and the analysis scripts that came from the old scale either recomputed or marked as old-scale, in particular the plateau range and TASK-85's 28%
- [ ] #6 A regression test that re-embeds a known text and asserts it reproduces a stored reference vector, so a library upgrade cannot silently move the scale again
<!-- AC:END -->
