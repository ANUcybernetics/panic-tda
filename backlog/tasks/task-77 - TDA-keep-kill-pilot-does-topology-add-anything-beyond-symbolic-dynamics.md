---
id: TASK-77
title: 'TDA keep/kill pilot: does topology add anything beyond symbolic dynamics?'
status: In Progress
assignee:
  - sungyeon-hong
created_date: '2026-07-10 00:51'
updated_date: '2026-11-05 11:30'
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

FOLLOW-UP RESULT 2026-11-04, on 01a10613 (240 runs, six networks, five prompts, eight runs a prompt, 1,000 text states each), run on the GPU server. Written up in backlog/docs/tda-keep-kill.md under "The follow-up at 1,000 states"; numbers in analysis/tda_keep_kill_01a10613.json and analysis/sliding_window_01a10613.json, with the same runs cut to their first 150 states in the files suffixed _states1-150 as a control.
- AC#1 REVERSES ON THE LONGER RUNS. Sliding-window H1 beats the null in 29/31/27% of runs at windows 5/10/20, against 5% by chance. The same runs cut to 150 states sit at chance (4/3/3%), so this is the horizon and not the network set. It is above chance for every one of the five prompts and every one of the six networks. Recurrence on raw distances also rises, to 49% (return after excursion 40%), and separates by captioner as before (Gemma4 0.78, Moondream3 0.18) while sliding-window H1 does not (0.30 against 0.32). Loops and near-verbatim returns are not the same effect.
- AC#2 unchanged and stronger. The diagrams get MORE predictable from the distance numbers at 1,000 states, not less (R^2 for h1_count 0.85 against 0.68 on the panel, h2_count 0.69 against 0.48). Neither the generator nor the captioner gain interval clears zero. H1 and H2 remain below the drift null in 80-97% of runs. One exception: the longest H1 bar beats every draw in 24% of runs against a chance 10%, the only Rips feature to rise above chance, and it moves the same way as the sliding-window result.
- AC#4's second condition TESTED 2026-11-05 and FAILED. See below.
- AC#3's premise is weaker for Rips persistence than for sliding-window persistence, but see below: both end up on the same side.

LOOP ANATOMY 2026-11-05, analysis/loop_anatomy.py -> analysis/loop_anatomy_01a10613.json, written up in backlog/docs/tda-keep-kill.md under "What the loops are made of". AC#4 requires a cycle that can be pointed to in the captions. The 62 runs whose longest H1 bar beats all 19 draws at window 20 were taken apart: ripser's generators give the edge that closes each loop and the chord that fills it, and the rest of the cycle is the shortest path below the birth scale. A vertex is a window, a window is a time, a time has a caption.
- The cycles can be pointed at, and they are NOT circuits. The closing edge joins ADJACENT windows in 41 of 62 runs (median gap 1 state) and is only 1.7 median steps long, so no long-range return shuts the loop. A cycle carries a median 34.5 windows with 33.5 distinct captions but only 7 distinct caption openings, and those seven are seven ways of opening the same sentence about the same image ("this is a studio photograph featuring" / "this is a studio portrait of" / "a full-length studio portrait features a" ...). 60 of 62 cycles revisit an opening they had already left, a median of 8 times, and only 28 return to the one they started from: the sequences shuttle rather than circulate.
- So a loop is a hole in the cloud of paraphrases a captioner produces for one settled scene. The excess bar length over the null is real, but it measures that the paraphrase cloud is not shaped like a Gaussian with the same drift. It is not a trajectory going round anything.
- RECOMMENDATION, for Sungyeon and Ben, unchanged in direction from 2026-10-08 but now for a stated reason rather than for absence of signal: drop persistent homology, Rips and sliding-window both, from the headline analyses. AC#4's first condition is met at 1,000 states and its second is not. If the paraphrase-cloud shape is worth reporting, local intrinsic dimension and clumpiness say it more plainly and are already in the document.
- Method note for whoever reads this next: the first examples pulled out were ranked by span per hop, which selects the degenerate cycles (seven windows all saying "a large tree engulfed in flames") and overstates the case. loop_anatomy.py picks its examples near the median instead. The population numbers, not the examples, are what the conclusion rests on.
<!-- SECTION:NOTES:END -->
