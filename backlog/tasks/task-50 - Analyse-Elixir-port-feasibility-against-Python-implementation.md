---
id: task-50
title: Analyse Elixir port feasibility against Python implementation
status: To Do
assignee: []
created_date: '2026-02-02 07:07'
labels:
  - elixir
  - architecture
  - investigation
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Compare the ELIXIR_PORT_INVESTIGATION.md proposal with the current Python codebase to identify which components are realistically portable and produce revised effort estimates. The investigation document makes optimistic claims about ONNX model availability and Ortex maturity that need verification against actual implementation requirements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Document current Python model execution layer (diffusers, transformers, sentence-transformers) with specific model architectures used
- [ ] #2 Verify ONNX export feasibility for each model (Z-Image-Turbo, Florence-2, SigLIP, Moondream)
- [ ] #3 Compare Python SQLModel schema with proposed Ash resources and identify mapping gaps
- [ ] #4 Map current Ray-based orchestration to proposed Reactor patterns with complexity assessment
- [ ] #5 Assess Rustler NIF requirements for giotto-ph TDA operations
- [ ] #6 Produce revised effort estimates with risk factors for each component
- [ ] #7 Recommend hybrid vs pure-Elixir approach with justification
<!-- AC:END -->
