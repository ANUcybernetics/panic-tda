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
four runs per prompt, the count the programme carried; two runs halves the
cost and still gives 40 trajectories per cell, which is the MSM's input.

## Pilot

One fast network (Flux2Klein + Moondream3), four prompts, one run each, 300
steps, run before launch to confirm per-step cost against the table and that
nothing degrades along the trajectory: caption length, exact repeats, step
distance and drift from the first caption by 25--50 step bins. Results are
appended below once the pilot completes.
