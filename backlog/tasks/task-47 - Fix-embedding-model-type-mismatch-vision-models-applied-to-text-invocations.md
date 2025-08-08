---
id: task-47
title: Fix embedding model type mismatch - vision models applied to text invocations
status: Done
assignee: []
created_date: '2025-08-08 06:02'
labels: []
dependencies: []
---

## Description

NomicVision (and likely JinaClipVision) are vision/image embedding models that only work with PIL Image inputs, but they're being applied to TEXT invocations which provide string inputs. This causes ValueError: Expected PIL Image but got <class 'str'>. The engine needs to filter embedding models based on invocation type - only apply text embedding models to TEXT invocations and image embedding models to IMAGE invocations. Currently 1.3M+ NomicVision embeddings for TEXT invocations all have null vectors due to this mismatch.

## Solution Implemented

Added a clean model type system to distinguish between text and image embedding models:

1. **Added `EmbeddingModelType` enum** in `embeddings.py` with TEXT and IMAGE values
2. **Added `model_type` class attribute** to each embedding model class:
   - Text models: Nomic, JinaClip, STSBMpnet, STSBRoberta, STSBDistilRoberta, Dummy, Dummy2
   - Image models: NomicVision, JinaClipVision
3. **Added helper functions**:
   - `get_model_type()`: Get the type of a specific model
   - `list_models_by_type()`: List all models of a given type
4. **Updated `perform_embeddings_stage()` in `engine.py`**:
   - Now filters invocations based on model type
   - TEXT models only process TEXT invocations
   - IMAGE models only process IMAGE invocations

This prevents type mismatches and ensures embeddings are computed correctly for each invocation type.
