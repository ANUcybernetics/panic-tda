---
id: TASK-104
title: 'Design and run the long follow-up: is a prompt ever forgotten?'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-04 07:19'
updated_date: '2026-10-04 09:40'
labels:
  - experiment
  - paper
  - gpu
dependencies:
  - TASK-103
references:
  - backlog/docs/drift-and-memory.md
  - backlog/docs/research-programme.md
  - analysis/panel_audit.py
  - analysis/panel_regenerate.py
  - backlog/docs/long-follow-up-design.md
  - analysis/follow_up_design.py
  - analysis/pd_cost.py
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-103 found that the panel (experiment 01a09e21, 150 text states per run) is too short to answer the question RQ1 now turns on. Prompt memory falls from 0.71 to 0.35 and has almost stopped falling, and the two runs of a prompt have nearly stopped separating, well short of the distance between runs of different prompts. At the rate of the last fifty states they would meet after about another 570 text states, and that rate has been falling, so telling a plateau from a slow climb needs runs of at least 700 text states (1,400 invocations). The same runs would show how long a run takes to cover its prompt's region: at a separation of seventy states the panel's runs are 23-59% of the way to the distance between twins. The old 5,000-invocation runs (analysis/long_run_drift.py) say displacement does get there, in 200 to more than 1,250 text states, but their prompts had no content, so they say nothing about memory. See backlog/docs/drift-and-memory.md and the horizon section of backlog/docs/research-programme.md.

WHAT IS ALREADY KNOWN ABOUT THE DESIGN. Fewer prompts and more runs per prompt than the panel: with two runs a prompt's centre is never observed, only inferred, and eight runs give it directly. Five prompts by eight runs keeps a cell at 40 runs in lockstep, which is the panel's batch shape; captioner output depends on batch size, so 40 keeps the captioners defined as they were. The prompts should span the range of memory the panel measured (prompts.by_prompt in analysis/drift_memory.json: 0.64 for a red apple down to 0.12 for a city turning into a forest). Cost per cell of 40 runs to 700 text states, from the panel's own step times: about 2.5 days with SD35Medium or ZImageTurbo, 1.5 with Flux2Klein, 15 with Flux2Dev. The GPU is shared at times, which cost the panel two retried steps and nothing else.

LAUNCH TRAPS LEFT BY THE PANEL. logs/long-run.id still holds 01a09e21, and bin/long-run resumes whatever id is in that file, so a new launch would exit at once reporting the old experiment complete. bin/panic-experiment.service and the installed copy in ~/.config/systemd/user both name the panel's config in ExecStart. TASK-90's notes record what else went wrong at its launch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The prompts, runs per prompt, networks and horizon are chosen and written down with the GPU-day cost and what each choice is for, and Ben has agreed the design before any GPU time is spent
- [x] #2 The config is committed and the run is launched detached and resumable, with the experiment id and the expected completion date recorded here
- [ ] #3 The run completes and the audit that passed on the panel passes on it: every image, caption, seed and embedding checked, and a sample of steps regenerated from their stored inputs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. AC#1, done 2026-10-04: the design read off the panel (analysis/follow_up_design.py) and written up with costs and what each choice is for (backlog/docs/long-follow-up-design.md); Ben agreed it before any GPU time was spent.
2. Launch traps, cleared 2026-10-04: bin/long-run keeps its id and log per config, the unit is a template (panic-experiment@<config>), the panel's id and log are moved under its config's name and the old unit is removed; the two end-of-cell calls allow for a long run.
3. AC#2, done 2026-10-04: config committed, one cell smoked at the batch of 40, the new unit killed and seen to restart and resume, launched as experiment 01a10613 with the expected completion recorded in the notes.
4. AC#3, once the run completes (due 25 Oct 2026; the last line of logs/long-run.long_follow_up_3x2_2000.log says so, and the unit exits 0):
   a. mix experiment.export_data 01a10613 --output 01a10613_parquet
   b. ./analysis/panel_audit.py 01a10613 --cache 01a10613_parquet --log logs/long-run.long_follow_up_3x2_2000.log --out <a file of its own>. The script is ready: the low-detail rule follows the run's length, and its image, caption, process and log checks ran clean on the live run's first six steps. Its embeddings check needs at least one embedded cell, so it can first be run when Flux2Klein + Gemma4 finishes (due 7 Oct) to catch anything wrong at 1,000 text states before the other five cells are spent.
   c. analysis/panel_regenerate.py 01a10613 --out <a file of its own>. The script now takes its targets by experiment (TARGETS), so what is left is to add this run's: steps either side of the 20:30 restart on 4 Oct (image step 30 of Flux2Klein + Gemma4) and of any later one, any retried step, and the first flat image of any run that goes flat. --sections encoder_ceiling counts tokens without the GPU and can run while the experiment does; the other sections wait for the run to finish.
   d. The audit's process table has one entry the panel's did not: code under lib/ changed after launch (commit d2143d5, the two reads in the stages that follow a cell, taken up by the restart at step 30; nothing in generation or captioning changed, and RunExecutor's compiled module is still the one from 5 September).
   e. Write the audit up as panel-audit.md was, then disable the unit instance (systemctl --user disable panic-experiment@long_follow_up_3x2_2000).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PRE-LAUNCH FINDINGS 2026-10-04, before any GPU time.

Two calls at the end of a cell would have stopped a long run. (1) The persistence diagram ran under Snex's default five-second timeout. The panel's 150-point diagrams took 0.03 s; at 700 points one takes 7-11 s and at 1,400 up to 140 s (analysis/pd_cost.py, CPU only, in the pipeline's own venv). The run would have failed at the end of its first cell, after 1.5-3 days of GPU, and failed again on every resume. Now an hour's ceiling, with a test that failed with response_timeout before the change. (2) The embedding call had a fixed 60 s for a whole run's captions; Gemma4's took 0.04 s each in the panel, so 700 would have been 27 s plus warm-up, inside the limit but close. Now a second per caption. mix test 118/0 with GPU excluded.

The diagram's cost depends heavily on the cloud. Points with no structure are far slower (7 s at 150 points), but no run is like that: the panel's slowest diagrams are frozen Moondream3 runs at 0.1 s, and clouds drawn from a run's own covariance take under 2 s at 700 points.

Launch traps cleared as the plan records. bin/long-run and the template unit were exercised end to end on CPU with config/experiment.example.json, directly and under systemd; the two dummy experiments were deleted afterwards. Also fixed: the retry line logged the resume's exit status after $(date) had overwritten it, so it always said 0.

One thing left as it is: experiment.run prints an 8-character id, which is all bin/long-run records. Two experiments created within 65 s of each other share those 8 characters (UUIDv7's top 32 bits of the millisecond clock), and the resume loop then never finds its experiment. It took two launches thirteen seconds apart in the CPU exercise to hit it; a real launch cannot unless a first attempt is abandoned and relaunched inside a minute without being deleted.

DESIGN AGREED 2026-10-04 (AC#1). Ben chose the six cells (Moondream3 and Gemma4 with Flux2Klein, SD35Medium and ZImageTurbo), the five prompts the rule picks, and 1,000 text states over the 700 I had recommended: 2,000 invocations a run, 240 runs, 20.2 GPU-days. The longer horizon runs past state 840, where the last fifty states' rate would bring twins to the stranger distance on the twelve affordable cells, and needs no extension step. Written up in backlog/docs/long-follow-up-design.md with what each choice is for and what was priced and not chosen; config/long_follow_up_3x2_2000.json.

PRE-LAUNCH SMOKE 2026-10-04 on the code that runs the experiment (commit cab1d0c, tree clean, model environment last modified 3 September as for the panel). The first cell at the follow-up's batch, Flux2Klein + Gemma4, 5 prompts x 8 runs x 4 steps: completed in 6 min 53 s with no errors or retries. Each step a single batch of 40; image steps 160 s cold and 151 s warm, caption steps 31 s and 21 s, against the panel's medians of 151 and 20; 80 images with 80 distinct recorded seeds; 80 captions, none cut; 80 embeddings at 256 dimensions; 40 persistence diagrams. Deleted afterwards (experiment 01a105fe).

RESTART CHECK 2026-10-04, because the unit and the script both changed. The same cell for 6 steps under panic-experiment@restart_check, with every process in the unit killed (SIGKILL) 40 s into its third step. systemd restarted it 60 s later, bin/long-run resumed the recorded id, and the run completed and exited 0. The interrupted step was redone whole: every step is a single batch of 40, no duplicate (run, step) pairs, every step linked to the one before it, 120 embeddings and 40 diagrams. Deleted afterwards (experiment 01a10607), with its config, id file and log.

LAUNCHED 2026-10-04 19:42 AEDT (AC#2) as experiment 01a10613 (01a10613-d5e1-7393-99af-d59399021508): systemctl --user enable --now panic-experiment@long_follow_up_3x2_2000, enabled so it returns after a reboot, linger on. 240 runs of 2,000 invocations. Id in logs/long-run.long_follow_up_3x2_2000.id, output in logs/long-run.long_follow_up_3x2_2000.log. The first four steps match the cost model: 152 s for 40 Flux2Klein images, 27 s then 18 s for 40 Gemma4 captions, no retries.

EXPECTED COMPLETION 25 Oct 2026, from the panel's step times (analysis/follow_up_design.json, cost.designs). Cell by cell in config order, AEDT: Flux2Klein + Gemma4 by Wed 7 Oct 02:40; SD35Medium + Moondream3 by Sun 11 Oct 04:00; ZImageTurbo + Moondream3 by Thu 15 Oct 03:45; Flux2Klein + Moondream3 by Sun 18 Oct 05:25; SD35Medium + Gemma4 by Wed 21 Oct 14:00; ZImageTurbo + Gemma4 and the experiment by Sun 25 Oct 00:50. The panel's own estimate held to within a day over 19.5 days. Each cell is embedded and given its diagrams when it finishes, which takes 37-56 minutes at this length and is the first time those stages run on 1,000 text states with real captions.

FOUND AFTER LAUNCH 2026-10-04, and fixed with one restart. To see the stages that follow a cell at full length before 7 October, one run of 1,000 text states was stored in the test database as the pipeline stores it (2,000 invocations, 136 MB of images) and put through EmbeddingsStage and PdStage on CPU. The diagram stage failed at once: it read the run's embeddings with each one's invocation loaded, Ash renders that load as one id = ? per embedding in a nested OR, and SQLite refuses an expression more than 1,000 deep (Expression tree is too large). EmbeddingsStage.resume had the same load. So the live run would have finished its first cell at about 02:40 on 7 October, failed in the diagram stage, and then failed in the embeddings resume on every retry, with the GPU idle. 150 states a run, and the 700 I first proposed, are both under the limit; 1,000 is exactly on it. Recompute already pages around the same limit.

Fix in commit d2143d5: the resume reads invocation_id, and the diagram stage orders its embeddings through the run's own invocations. A test of a run of 1,100 text states (embedded, resumed, given its diagram) failed with the SQLite error before and passes after. The full-size rehearsal then passed: stored in 0.6 s, embedded in 0.4 s with the dummy model, diagram stage 42 s. It is kept as test/long_run_rehearsal_test.exs, out of the default suite. No other read in the pipeline passes a list that grows with a run's length.

Ben chose to restart rather than let the run meet the fix at the end of its cell. Killed at 20:30:40 AEDT, 67 s into image step 30 of Flux2Klein + Gemma4, with step 29 fully stored; systemd restarted it at 20:31:42; the resume compiled the two changed files and went on. Step 30 was redone whole from 20:31:46. Checked afterwards: 32 steps each a single batch of 40, no duplicate (run, step) pairs, every step linked to the one before, no retries. About two minutes lost. Only the PdStage and EmbeddingsStage modules were recompiled.

Suite with the fix: three full runs, 119 tests. The second and third passed. The first had one failure, which passed when re-run on its own; an output filter had cut its name, so which test it was is not known.

Also done while the run goes: analysis/panel_regenerate.py takes its targets by experiment and can count tokens without the GPU (run that way on the panel it reproduces the stored section exactly); CLAUDE.md says what to leave alone while a long run is live; TASK-106 holds the eight-character id, which cannot be fixed until the run is over.

The path the resumed run takes at the end of a cell was rehearsed too, since it now runs under experiment.resume: EmbeddingsStage.resume and PdStage.resume from nothing, on one run of 1,000 text states. Both passed (embedding 0.4 s with the dummy model, diagram 42 s). What remains unrehearsed is Qwen3Embed itself on 1,000 real captions a run, which needs the GPU the run is using; from the panel's 0.04 s a caption it should take about 40 s against a limit of 1,000.
<!-- SECTION:NOTES:END -->
