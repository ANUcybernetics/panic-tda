---
id: task-49
title: add DummyVision embedding models
status: Done
assignee: []
created_date: "2025-08-10 10:25"
labels: []
dependencies: []
---

## Description

The "dummy" embedding models are used for tests, however there are currently
only dummy embedding models for text invocations. Rename Dummy and Dummy2 to
DummyText and DummyText2, and add DummyVision and DummyVision2 models.

Ensure that new tests/test_embeddings.py tests are added to test these new
models. And ensure that the full engine tests (tests/test_engine.py) are updated
so that an experiment can have a dummy vision embedding model, and these tests
pass too.
