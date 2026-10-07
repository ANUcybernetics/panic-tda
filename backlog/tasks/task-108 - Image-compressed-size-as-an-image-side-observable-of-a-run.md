---
id: TASK-108
title: Image compressed size as an image-side observable of a run
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-05 02:25'
updated_date: '2026-10-07 04:58'
labels:
  - analysis
dependencies: []
references:
  - lib/panic_tda/models/image_converter.ex
  - backlog/docs/drift-and-memory.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every image is stored as AVIF, so its byte length (length(output_image)) is a free measure of how visually complex the image is, and one that does not pass through the captioner. Everything analysed so far is caption embeddings, and image embedding was removed, so this is close to the only image-side observable there is. The question is whether it is worth using, and for what.

WHAT A FIRST LOOK FOUND. On 5 October 2026, over the first 660 steps of the Flux2Klein + Gemma4 cell of experiment 01a10613 (40 runs, 13,200 images, read with ad hoc SQL that is not tracked): images ran 7.8 KB to 326 KB. About 81% of the variance in log byte length was between runs and 19% within a run. Prompt alone accounted for about half: geometric-mean size was 23 KB for the doorway prompt, 29 KB for the apple, 62 KB for firefighters, 72 KB for the train station and 100 KB for the city turning into a forest. Runs sharing a prompt still differed, with city run means spanning 39 to 236 KB. Pooled over runs, log size correlated 0.97 with the next image and 0.79 with the image a hundred later, but that pooling mixes in the between-run difference and says little about memory within a run. Run 2 of the doorway prompt had 307 of its 330 images under 15 KB: it had settled on a suited man in a white corridor.

WHAT IT MIGHT BE GOOD FOR. A cheap detector of a run settling into a visually simple attractor, from the level and windowed variance of size, with no embedding needed. A cross-check on the caption trajectory: whether jumps in embedding space coincide with jumps in size, which would be two unrelated instruments agreeing on where transitions are. A per-run or per-cell signature to set beside the distances in backlog/docs/drift-and-memory.md.

WHAT LIMITS IT. Size depends on the AVIF encoder settings (lib/panic_tda/models/image_converter.ex), so it compares only across images encoded the same way, and the settings in force for each experiment need confirming. It measures texture and detail, not meaning. It is confounded with each text-to-image model's house style, so cells with different generators need separating before they are compared; the first SD35Medium cell of 01a10613 is due about 11 October 2026 and the first ZImageTurbo cell follows it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An analysis/ script computes per-image byte length for a finished cell of 01a10613 and writes its results to the JSON beside it, with its lock committed
- [x] #2 The script reports the between-prompt, between-run and within-run split, and autocorrelation within runs after removing each run's mean
- [x] #3 It reports whether steps in byte length coincide with steps in caption-embedding distance
- [x] #4 The AVIF encoder settings in force for the experiment are confirmed and recorded with the result
- [ ] #5 A short note in backlog/docs says whether the measure is kept, and for what
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
MEASURED 2026-10-07 on the first finished cell of 01a10613 (Flux2Klein + Gemma4, 5 prompts x 8 runs x 1,000 images): analysis/image_size.py, results in analysis/image_size.json, written up in backlog/docs/image-size.md. Variance of log bytes splits 37% between prompts, 30% between runs of a prompt, 33% within a run; the seed is a tenth of the within-run variance and a run is still moving in size after 500 images. Single steps in size do not coincide with single steps between captions (rank correlation 0.03); between blocks of 25 images they do (0.45). A ridge regression from the caption embedding explains 55% of size on held-out runs. Encoder confirmed: libvips heifsave AV1 at quality 50, defaults otherwise, all images 1024 x 1024.

The first look's 81% between runs was at 330 images per run and does not hold over the full run, and the doorway run it called settled left its white corridor in the last quarter.

AC#5 is open on Ben's call: the note recommends keeping it as a cheap second instrument for slow movement and not as a step-level detector. Rerun per cell as the SD35Medium and ZImageTurbo cells finish (the script takes --network and --out).
<!-- SECTION:NOTES:END -->
