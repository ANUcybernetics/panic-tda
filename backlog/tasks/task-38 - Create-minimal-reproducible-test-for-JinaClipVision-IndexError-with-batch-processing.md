---
id: task-38
title: >-
  Create minimal reproducible test for JinaClipVision IndexError with batch
  processing
status: To Do
assignee: []
created_date: '2025-08-03'
labels: []
dependencies: []
---

## Description

Write a failing test case in tests/test_embeddings.py that reproduces the IndexError in JinaClipVision model when processing batches of images. This relates to the HuggingFace transformers issue #26999 where batch processing causes list index out of range errors.

## Error Details

From logs/script_2025-08-01_16-18-45.log, the error occurs when JinaClipVision processes batches of 100 images:

```
IndexError: list index out of range
  File "/home/ben/.cache/huggingface/modules/transformers_modules/jinaai/jina-clip-implementation/39e6a55ae971b59bea6e44675d237c99762e7ee2/modeling_clip.py", line 495, in encode_image
    all_embeddings = [all_embeddings[idx] for idx in _inverse_permutation]
                      ~~~~~~~~~~~~~~^^^^^
```

The error happens in the JinaCLIP implementation's `encode_image` method when processing multiple images at once.

## Test Requirements

1. Create a test that processes multiple images (batch size > 1) with JinaClipVision
2. The test should demonstrate the IndexError occurring in the model's encode_image method
3. Include a workaround test showing batch_size=1 processing works correctly
4. Document the issue and link to HuggingFace issue #26999 in the test comments

## Implementation Notes

- The error occurs at src/panic_tda/embeddings.py:499-501 when calling `self.model.encode_image(images, truncate_dim=EMBEDDING_DIM)`
- NomicVision model processes the same batches successfully, so issue is specific to JinaClipVision
- Current implementation passes a list of PIL Images directly to encode_image
