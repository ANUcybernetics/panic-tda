---
id: TASK-102
title: >-
  Analyse prompt abstraction tier as a declared factor, not as replicate seeds
status: To Do
assignee: []
created_date: '2026-09-16 13:40'
updated_date: '2026-09-16 13:40'
labels:
  - analysis
  - paper
dependencies:
  - TASK-76
  - TASK-90
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sungyeon, 2026-09-16: the long-horizon panel's twenty prompts are not twenty interchangeable seeds. They are five tiers of four, and `config/long_horizon_panel_4x4_300.json` orders them that way: single objects (1-4), multi-object scenes (5-8), people (9-12), events (13-16), abstractions (17-20). `long-horizon-design.md` describes them as "spanning concrete objects, scenes, people and abstractions" and stops there. Nothing in the backlog analyses concreteness as a designed dimension and no task owns it, so as things stand the panel will be analysed as though prompt were a replicate index.

PROPOSAL. Declare abstraction tier a factor of the design and report per-tier observables alongside the per-network ones. Prompt abstraction is a plausible determinant of how many metastable regions a cell has and of how fast a run leaves the region it starts in --- an abstract prompt has no single depiction to settle into, which is a mechanism RQ1 can actually test rather than a decorative covariate. It costs no GPU time: the prompts are already chosen this way and experiment `01a09e21` is already generating the data. The only cost is deciding to look.

WHAT THE BUDGET SUPPORTS. Tier is crossed with cell, not nested in it --- every cell runs all twenty prompts --- so at 2 runs per prompt a (cell, tier) pair holds 8 trajectories, or roughly 600-800 stationary text states after burn-in. That is far too sparse for a transition matrix of its own: at the 60-80 microstates the per-cell budget supports it is 8-13 frames per microstate, against the ~50 that `escape-time-resolvability.md` sets as workable. Pooled across the sixteen cells a tier holds 128 trajectories and roughly 9,600-12,800 stationary frames, which is 120-210 frames per microstate and is workable.

So tier is a pooled observable, not a per-cell one, and the shared frozen partition TASK-76 already specifies is what makes it computable: fit the partition once on all cells' pooled stationary frames, then estimate a transition matrix per tier across cells rather than per (cell, tier). Per-tier estimates inherit the guards `analysis/msm_pipeline.py` already applies --- at least 10 observed crossings for an escape time, 20 complete residences for a dwell verdict --- and anything below them is reported as unresolved in the usual way.

OBSERVABLES per tier: stationary distribution over the shared metastable sets, the number of distinct sets a tier's runs visit, time to first leave the set containing the initial prompt, and escape times between the shared sets where the crossing count allows. The first three need no per-cell transition matrix and are available even where the fourth is unresolved.

WHAT WOULD FALSIFY IT. If between-tier variation in these observables does not exceed within-tier variation across the four prompts of a tier, then the tiers are replicate seeds after all and should be reported as such. That is a result worth having either way, and it is the reason to state the factor before looking rather than after.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Abstraction tier recorded as a declared factor in `long-horizon-design.md`, with the five-tier assignment of the twenty prompts written out and the mechanism it is meant to test stated
- [ ] #2 Per-tier observables computed on the shared frozen partition pooled across cells: stationary distribution over metastable sets, number of distinct sets visited, and time to leave the initial set
- [ ] #3 Per-tier escape times reported where at least 10 crossings are observed and reported as unresolved otherwise, using the existing `msm_pipeline.py` guards rather than new ones
- [ ] #4 Power stated explicitly in the output: stationary frames per (cell, tier) and per pooled tier, and frames per microstate at the chosen partition size
- [ ] #5 Between-tier variation tested against within-tier variation across each tier's four prompts, so the null that tiers are replicate seeds is reported rather than assumed away
- [x] #6 Glossary entry for prompt tier added to `backlog/docs/glossary.md`
<!-- AC:END -->
