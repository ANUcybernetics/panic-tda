---
id: TASK-66
title: >-
  Extract inline Python into priv/python/ modules for type checking and reduced
  duplication
status: In Progress
assignee: []
created_date: '2026-02-18 21:42'
updated_date: '2026-02-18 21:42'
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
- [ ] #1 T2I invoke/batch logic extracted into a shared Python function parameterised by model name, steps, and guidance_scale
- [ ] #2 I2T invoke/batch logic extracted into pattern-based Python functions covering all current model variants
- [ ] #3 Model loader code extracted into a load_model(name) dispatcher with a config registry
- [ ] #4 Setup/init code from python_bridge.ex moved into the Python module
- [ ] #5 priv/python/ module loaded by snex at interpreter startup
- [ ] #6 ruff and ty pass on all new Python files with no errors
- [ ] #7 mise exec -- mix test passes (dummy models, no GPU)
- [ ] #8 mise exec -- mix test --include gpu passes for all models
<!-- AC:END -->
