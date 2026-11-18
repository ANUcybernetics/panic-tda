# Kong et al. (2026): a text-only closed loop that collapses

Written 2026-11-18 by Sungyeon Hong. It records a paper that asks this repo's
question about a different loop, and sets its results beside ours. The
numbers on our side come from `drift-and-memory.md` and
`research-programme.md`.

Weiyi Kong, Shiyang Lai, Jinghua Piao and James Evans, _Multi-LLM Systems
Exhibit Robust Semantic Collapse_, arXiv:2605.17193 (v1 16 May 2026, v3 14 Oct
2026; <https://doi.org/10.48550/arXiv.2605.17193>). Toronto, Chicago (Knowledge
Lab), Tsinghua and the Santa Fe Institute. 94 pages including the
supplementary information.

## What they did

Three LLM agents talk with no topic, no roles and no reward for 200 to 1,000
rounds. An automated referee routes the messages, and each agent has a
short-term dialogue buffer and a retrieval memory of its own history. The
weights are frozen, so like ours it is a loop that runs at inference time.
Unlike ours it is text all the way round, and each agent conditions on a
history rather than on the last output alone.

- Five primary models (GPT-5.6 Terra, Claude Sonnet 5, GPT-4-mini, DeepSeek-V3,
  Phi-4), three 1,000-round runs each. Ten models in all, 73 conditions.
- Each 10-round block is embedded with OpenAI `text-embedding-3-large`. Long
  blocks are split into chunks, the chunk vectors averaged and renormalised.
- Within-run diversity is the cosine distance from a block to the run's first
  block. Cross-run diversity is one minus the mean pairwise cosine similarity
  between runs at the same block. They also report the distance between
  adjacent blocks, Vendi score and cumulative vocabulary.

## What they report

- **Words keep changing and meaning stays put.** Cumulative vocabulary grows
  steadily. The last 500 rounds sit 0.19 from the first ten (95% CI
  0.17--0.22), and runs stay close to one another. Reddit threads matched for
  topic and length move 0.14 further.
- **Thirteen interventions fail.** These were temperature, output budget,
  diversity-seeking decoding, memory ablation, alternative prompts and
  personas, mixing model families, uncensored fine-tunes, activation steering
  against sycophancy, GRPO trained for diversity, population size, AutoGen and
  AgentSociety, and injected noise. None raises semantic diversity
  significantly after Bonferroni correction. Raising temperature from 0 to 2
  multiplies vocabulary 20--36-fold and _lowers_ both diversities.
- **Mechanism.** The tested biases (sycophancy, safety alignment) do not
  explain it. Attention-head diagnostics in Llama-3-8B-Instruct point to
  autoregressive feedback reinforcing historically dominant content. The data
  processing inequality is offered as an analogy for why repeated processing
  forgets its start.
- **Framing.** They distinguish their collapse from training-time model
  collapse (Shumailov et al.): the contraction happens at inference time with
  fixed parameters. They do not cite Hintze et al. (2025) or any loop between
  images and text.

## How it sits beside this repo's data

**The framing is no longer ours alone.** `research-programme.md` opens by
calling the loop an inference-time dynamical system and "explicitly not
training-time model collapse". Kong et al. make the same separation the
headline of their paper. Any write-up of ours has to cite them on that point.

**The headline results point opposite ways.** In the long follow-up
(`01a10613`, six networks, 1,000 text states):

- no prompt is forgotten: memory is 0.25 at state 1,000 and still falling by
  about 0.05 per doubling of run length;
- runs from different prompts converge only a little (the stranger distance
  falls 5--8% in four of six networks);
- no run has covered its prompt's region by a separation of 500 states.

Their loop contracts and stays near where it began. Ours keeps moving away
from its start, slowly, and never loses its prompt entirely.

**"Stays near the start" can mean memory or collapse, and their design cannot
tell which.** Their within-run measure is distance from the first block. A run
that stays close to its start scores as collapsed. In our terms that run
remembers its prompt. Every run of a model also starts from the same
topic-free instruction, so low cross-run diversity is what any loop would
show early on, collapsed or not. The design used here separates the two
readings. Twins (two runs of one prompt) measure a run's own offset. Strangers
(runs of different prompts) measure convergence. Collapse in their sense would
show up as the stranger distance falling to the twin distance, and in our data
it does not.

**Some of their supplementary results agree with ours.** Their clustering of
block embeddings follows the run more than the phase of the interaction (run
AMI 0.20--1.00, phase AMI at most 0.14). The supplement also says that
run-specific initial conditions set runs on different trajectories, which
interaction can preserve. That is the run's own offset, and it is the reason
the Markov state model here had nothing to count (TASK-76): each run keeps to
its own region.

**Averaging a block of turns hides the step.** Here four-fifths of the
distance between consecutive captions is seed noise that the next step takes
back. Averaging each 10-round block, as they do, removes most of that noise
before anything is measured. Their loop has no diffusion seed, but sampling
temperature plays a similar part, and their temperature result suggests that
token-level noise and semantic movement come apart there too.

**The scales do not compare directly.** The embedding models differ, they embed
pooled blocks where we embed single captions, and they take distances from the
first block where we take them from the prompt. Their 0.19 and our memory of
0.25 are not the same quantity. They only point the same or opposite ways.

## The question it raises: is it the modality?

Raised by Sungyeon Hong. The obvious difference between the two loops is that
ours passes through an image and theirs does not. Two things change at that
crossing. A diffusion seed adds fresh noise at every step, while their loop's
only fresh noise is token sampling. And a caption has to describe an image
rather than continue a text, so the captioner cannot condition on the history
of the run, only on the last image. Either difference could be what keeps our
runs moving and keeps our strangers apart.

The data here cannot separate those from the other differences: three agents
against two models, a history against a single message, and a topic-free start
against prompts with content. The direct test is a text-only arm built the way
this repo's experiments are built: a text-to-text paraphrase or continuation
loop, started from the panel's prompts, with twins, and read with the
quantities in `drift-and-memory.md`. The follow-up already says the image
loop does not collapse, so two outcomes matter:

- the text arm collapses, which points to the image step;
- the text arm does not collapse either, which points to their design (the
  shared start, the agents' memory of the history).

None of these has been costed. It is recorded here as an open question, not a
task.
