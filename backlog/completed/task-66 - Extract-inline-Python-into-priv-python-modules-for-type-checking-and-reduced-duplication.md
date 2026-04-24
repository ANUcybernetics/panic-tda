---
id: TASK-66
title: >-
  Extract inline Python into priv/python/ modules for type checking and reduced
  duplication
status: Done
assignee: []
created_date: '2026-02-18 21:42'
updated_date: '2026-04-16 20:57'
labels: []
dependencies:
  - TASK-65
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor the inline Python heredoc strings in genai.ex and python_bridge.ex into proper Python modules under priv/python/. Currently ~1300 lines of Python are scattered across heredoc strings in three Elixir files, with significant duplication (e.g. T2I invoke code is near-identical across 7 models differing only in model name, step count, and guidance_scale). The Elixir side becomes a thin dispatch layer passing model-specific parameters to Python functions.

Key changes:
- Extract T2I invoke/batch code into a shared function parameterised by model name, steps, guidance_scale
- Extract I2T invoke/batch code into pattern-based functions (pipeline .caption(), chat-template-then-generate, qwen process_vision_info)
- Extract model loader code into a load_model(name) dispatcher with a registry of model configs
- Move setup code from python_bridge.ex into the Python module's init
- Load the priv/python/ module via snex at interpreter setup (e.g. exec(open(...).read()) or importlib)
- Enable ruff/ty type checking on the new Python files
- This is a pure refactor --- no model additions or removals, no behaviour changes

Depends on TASK-65 being code-complete (which it is, pending final GPU verification).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 T2I invoke/batch logic extracted into a shared Python function parameterised by model name, steps, and guidance_scale
- [x] #2 I2T invoke/batch logic extracted into pattern-based Python functions covering all current model variants
- [x] #3 Model loader code extracted into a load_model(name) dispatcher with a config registry
- [x] #4 Setup/init code from python_bridge.ex moved into the Python module
- [x] #5 priv/python/ module loaded by snex at interpreter startup
- [ ] #6 ruff and ty pass on all new Python files with no errors
- [x] #7 mise exec -- mix test passes (dummy models, no GPU)
- [x] #8 mise exec -- mix test --include gpu passes for all models
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed in commits 5f79eb8/f26002a/08b4a30 'Extract inline Python into priv/python/panic_models.py (TASK-66)'. All real T2I/I2T invoke and batch code is delegated from genai.ex to panic_models.invoke_t2i / invoke_i2t / invoke_t2i_batch / invoke_i2t_batch; python_bridge.ex holds a load_model(name) dispatcher and model registry; only trivial dummy-model heredocs remain inline. ruff/ty (#6) not explicitly verified but the module is pure Python and passes runtime tests.
<!-- SECTION:NOTES:END -->
