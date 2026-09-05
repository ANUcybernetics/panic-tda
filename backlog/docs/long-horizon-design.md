# Long-horizon experiment design

The design for the uniform factorial that both research questions need
(TASK-90), written 2026-09-05 so the methods section can be lifted from it.
Config: `config/long_horizon_panel_4x5_300.json`. Pilot:
`config/long_horizon_pilot_flux2klein_moondream3.json`, checked by
`analysis/long_horizon_pilot.py`.

## Design

One uniform factorial over the v2 panel: four text-to-image models (SD35Medium,
ZImageTurbo, Flux2Klein, Flux2Dev) crossed with five image-to-text models
(Moondream3, Qwen25VL, Qwen3VL, Gemma4, JoyCaption), twenty prompts spanning
concrete objects, scenes, people and abstractions, 300 steps per run (150
image and 150 caption states), and the same number of independent runs per
prompt in every cell. Every text-to-image invocation draws and records its
own seed; every captioner decodes greedily (decision-02); captions are never
truncated (decision-01); `max_sequence_length` is pinned per generator. Text
states are embedded with Qwen3Embed at 256 dimensions.

## Why 300 steps

The horizon is set by what the Markov state model needs, not by a target
number. The existing 200-step runs (`analysis/long_horizon_baseline.py`)
plateau in step-to-step distance and in distance from the initial caption by
step 100--150, so the chain is in a stationary regime for the second half of a
300-step run. The kinetic result is metastable-region identity and the escape
times between regions; those are resolved by implied timescales converging
with lag time (TASK-76 AC#4), and the longest resolvable timescale scales with
aggregate sampling time across trajectories rather than with the length of any
one (Sinitskiy & Pande 2018). Past the plateau, runs per prompt is therefore
the lever, not horizon. A cell whose slowest implied timescale does not
converge within 300 steps is reported as unresolved rather than extrapolated.

## Step size against the noise floor

TASK-89 measured what one loop step is made of. In the settled part of a
200-step run the generator's own sampling accounts for 89--107% of the step
(Flux2Klein settled step 0.030--0.034 against a noise term of 0.031;
SD35Medium 0.042--0.050 against 0.045), and the settled step is about a
fifteenth of the distance between captions of unrelated prompts. So at any
horizon past the plateau, step-to-step motion is not resolvable as semantic
travel and is not the observable. The horizon is justified by the state-level
kinetics above, and TASK-90 re-measures the noise floor from its own
trajectories, where the lineup and caption lengths match, rather than
carrying TASK-89's indicative figures.

## Runs per prompt and cost

Measured warm per-item times (CLAUDE.md), 150 image and 150 caption steps
per run, 20 prompts, 20 cells. The model predicted 14.9 days for the panel
that took 17, so the second column carries that overhead.

| runs per prompt | trajectories | GPU-days (model) | GPU-days (with overhead) |
| --------------- | ------------ | ---------------- | ------------------------ |
| 1               | 400          | 14.6             | 16.7                     |
| 2               | 800          | 29.3             | 33.4                     |
| 3               | 1,200        | 43.9             | 50.1                     |
| 4               | 1,600        | 58.5             | 66.8                     |

Flux2Dev is 69% of the total at every setting. The config is committed at
two runs per prompt (decided 2026-09-05): 40 trajectories per cell is the
MSM's input, it finishes in five weeks rather than ten, and runs per prompt
is the lever past the plateau, so a second batch can be added if implied
timescales do not converge.

## Launch

`bin/long-run` under the `panic-experiment` systemd user unit
(`bin/panic-experiment.service`), so the run survives crashes and the
fortnightly reboots this machine gets. Cells execute in config order with
Flux2Dev last, and each cell is embedded and given its persistence diagrams
as soon as it finishes, so the fifteen fast cells (31% of the GPU time,
about ten days) are analysable while Flux2Dev runs.

## Pilot

One fast network (Flux2Klein + Moondream3), four prompts, one run each, 300
steps, run before launch to confirm per-step cost against the table and that
nothing degrades along the trajectory. Experiment 01a0708e, run 2026-09-05,
results in `analysis/long_horizon_pilot.json`.

**Cost.** Wall clock 2 h 05 min for four runs. Median per-item time 4.7 s for
Flux2Klein (table: 4.1) and 2.3 s for Moondream3 (table: 2.4), so the per-item
model holds. The other half of the wall clock, 11 s per step (0.91 h in
total), is the swap between the two models on every step. That is a
small-batch artefact: a panel cell runs its 20 prompts times R runs as one
batch, so at R = 4 the swap is 3% of a step and at R = 2 about 5%, within
the overhead the table already carries. Embedding and persistence diagrams
for 600 captions took 42 s.

**Nothing degrades.** By step bin over the four runs:

| steps   | words | exact repeats | step distance | drift from t0 |
| ------- | ----- | ------------- | ------------- | ------------- |
| 0--25   | 44.5  | 0             | 0.038         | 0.12          |
| 25--50  | 47.5  | 0             | 0.051         | 0.17          |
| 50--100 | 47.0  | 0             | 0.044         | 0.20          |
| 100--150 | 46.5 | 0             | 0.038         | 0.21          |
| 150--200 | 44.0 | 0             | 0.044         | 0.22          |
| 200--250 | 48.0 | 0             | 0.032         | 0.28          |
| 250--300 | 48.0 | 0             | 0.038         | 0.34          |

Caption length is flat at Moondream3's natural length, there is not one exact
repeat in 600 captions (the old 23-word Moondream repeated in 38 of 40 runs;
at 46 words it does not, as the length dependence predicted), and the step
distance sits at 0.03--0.05 with no trend, matching the settled step for
Flux2Klein networks in the 200-step baseline. Drift from the initial caption
keeps growing through step 300 while the step size does not, which is the
signature of a stationary chain on a large state space (the Conde et al.
confound in the programme) rather than of continued directed motion; with
four runs it is indicative, and the panel reports both curves.
