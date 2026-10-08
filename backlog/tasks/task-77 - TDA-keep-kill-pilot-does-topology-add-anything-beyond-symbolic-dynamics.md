---
id: TASK-77
title: 'TDA keep/kill pilot: does topology add anything beyond symbolic dynamics?'
status: In Progress
assignee:
  - sungyeon-hong
created_date: '2026-07-10 00:51'
updated_date: '2026-10-08 12:00'
labels:
  - analysis
  - paper
dependencies:
  - TASK-103
references:
  - analysis/tda_keep_kill.py
  - analysis/sliding_window.py
  - backlog/docs/tda-keep-kill.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Decide with data whether TDA earns a place in the next paper or is dropped from the headline claims. Static Rips PH on 25-100 points in 2560-dim is unreliable beyond H0 (curse of dimensionality on persistence diagrams, arXiv:2404.18194) and H0 duplicates hierarchical clustering; persistence entropy is confounded by bar count (duplicate captions). The only candidate value-add is detecting geometric recurrence/limit cycles that cluster-label sequences miss (sliding-window persistence, Perea-Harer). Depends on TASK-76 symbol sequences for comparison; blocked on the running experiment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recurrence statistics on raw embedding distances (return-time distributions / recurrence plots) computed for a sample of runs and compared against symbol-sequence recurrence, establishing whether geometric recurrence exists that symbols miss
- [x] #2 Test of whether (normalised) persistence entropy predicts anything not already predicted by duplicate-caption count and dwell statistics (e.g. partial correlation)
- [ ] #3 If killed: PdStage moved out of the per-experiment hot path (or made opt-in) and docs updated
- [ ] #4 Documented keep/kill decision: sliding-window PH is adopted only if it shows something the symbol sequences miss AND the result is interpretable in terms of the trajectories (a recurrence or cycle that can be pointed to in the captions); otherwise TDA is dropped from headline analyses with the rationale written up for the paper's methods discussion
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Gate for Results III, which exists in the paper only if this passes: sliding-window persistence is retained only if geometric recurrence exists that symbol sequences miss. Otherwise it shrinks to a negative-result paragraph and the TDA rationale moves to discussion. See backlog/docs/research-programme.md.

DEPENDENCY MOVED 2026-10-04. TASK-76 (the Markov state model) was abandoned: the panel has no states that runs both share and cross, so there are no symbol sequences to compare against. The baseline topology has to beat is now TASK-103's description (seed noise, slow wander, prompt memory), and AC#1's question becomes whether geometric recurrence exists that the displacement curve and the twin/stranger distances miss.

RESULT 2026-10-08 (Sungyeon). Written up in backlog/docs/tda-keep-kill.md; numbers in analysis/tda_keep_kill.json and analysis/sliding_window.json. On the panel (01a09e21, 150 text states a run), against a null built from each run's own displacement curve and local intrinsic dimension (sliding_window.Null, TASK-103's description and nothing else):
- AC#1. Sliding-window H1 finds no cycles: 1-3% of runs beat the null at windows 5, 10, 20, against 5% by chance. Recurrence on raw distances does exist: 13% of runs return to within one step's distance of an earlier caption, after moving their typical 20-state distance away, more often than the null (20% for Gemma4 and JoyCaption, chance level for Moondream3 and Qwen25VL). These are near-verbatim returns to a point (an earlier caption), not loops, so a distance count sees them and persistent homology does not. Part of the excess may be the Gaussian null's tails; the doc says how far to trust it.
- AC#2. With the dwell statistics gone with TASK-76, the comparison is against twelve distance numbers in TASK-103's vocabulary plus duplicate-caption share. H0 is predicted by them (R^2 0.98 for total lifetime). H1 and H2 add nothing at naming the generator and +0.017 (interval includes zero) for the captioner once intrinsic dimension is among the baselines. Real runs have less H1/H2 than the null in 79-96% of runs, never more.
- AC#4 is NOT checked: the doc recommends dropping persistent homology from the headline analyses and reporting intrinsic dimension and recurrence directly as distance statistics, but the keep/kill decision is for Sungyeon and Ben.
- AC#3 waits on that decision, and on TASK-104's run (01a10613), since lib/ is not to be touched while it is live.
- Open: repeat both scripts on the long follow-up's 1,000-state runs once it completes (due about 25 Oct).
<!-- SECTION:NOTES:END -->
