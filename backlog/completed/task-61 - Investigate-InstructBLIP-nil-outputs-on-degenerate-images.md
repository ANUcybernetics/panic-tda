---
id: TASK-61
title: Investigate InstructBLIP nil outputs on degenerate images
status: Done
assignee: []
created_date: '2026-02-09 11:00'
updated_date: '2026-02-17 04:10'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In the all_combos experiment (019c381c), 7 out of 16,000 text invocations have nil `output_text`, all from InstructBLIP. 6 of 7 occur with ZImageTurbo inputs, 1 with FluxSchnell. The input images are 0--5 KB AVIF files --- degenerate near-uniform images produced after many recursive iterations.

The nil values caused the embeddings stage to crash (the SentenceTransformer tokeniser received None instead of a string). This was patched by filtering out nil outputs in `EmbeddingsStage.compute_for_invocations`, but the root cause in InstructBLIP's inference path is unaddressed.

### Observations

| Field | Values |
|---|---|
| Affected model | InstructBLIP only (Moondream, Qwen25VL, Gemma3n all handled the same images) |
| Input T2I models | ZImageTurbo (6), FluxSchnell (1) |
| Sequence numbers | 1, 19, 53, 61, 63, 79, 93 --- spread across the run, not concentrated early/late |
| Input image sizes | 0 KB (3), 1 KB (3), 5 KB (1) |

The 0 KB images are suspicious --- these are likely corrupt or fully uniform images that compress to near-zero bytes in AVIF.

### Questions to answer

1. **What does InstructBLIP's Python code return for these images?** Does `batch_decode` produce an empty string that Ash converts to nil, or does the model raise an exception that's silently caught somewhere?
2. **Why does the validation allow nil output_text for text invocations?** The `OutputMatchesType` validation passes when both `output_text` and `output_image` are nil (the fallthrough case). Should this be an error?
3. **Are the 0 KB input images actually valid?** Check whether `ImageConverter.to_avif!` can produce a 0-byte output, or if this indicates an upstream T2I failure.
4. **Do other I2T models handle degenerate inputs gracefully?** Moondream, Qwen25VL, and Gemma3n didn't produce nil outputs, but did they produce meaningful captions for the same degenerate images?

### Suggested investigation steps

1. Reproduce with a 0 KB AVIF image: load one of the 0 KB images from the DB, pass it to InstructBLIP manually via iex, and observe the raw Python output.
2. Check the Ash `:string` attribute behaviour for empty strings --- does `""` get cast to nil?
3. Review `OutputMatchesType` validation: the fallthrough allows both fields nil. Decide whether to reject or handle gracefully.
4. Decide on a fix: either make InstructBLIP return a placeholder string for degenerate inputs, or add error handling in `invoke_batch_real_i2t` that detects nil/empty results and substitutes a fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root cause identified: determine exactly what InstructBLIP returns for degenerate images and how it becomes nil in the DB
- [x] #2 Fix applied so InstructBLIP (and other I2T models) never produce nil output_text
- [x] #3 OutputMatchesType validation updated if appropriate
- [x] #4 Existing tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause identified and fixed. InstructBLIP's batch_decode returns empty string "" for degenerate images (only special tokens generated), which Ash's :string type casts to nil (allow_empty? defaults to false). Fixed by: (1) adding ensure_nonempty_text/1 in GenAI that replaces empty strings with "[empty]" for all I2T model outputs, (2) tightening OutputMatchesType validation to reject nil output_text for text invocations and nil output_image for image invocations, (3) added tests for both nil-output rejection cases. The embeddings stage nil filter was retained as defence-in-depth for historical data.
<!-- SECTION:NOTES:END -->
