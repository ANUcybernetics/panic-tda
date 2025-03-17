# Trajectory Tracer

This repo is a collection of tools for analysing the trajectories of
interconnected generative AI models. This is a companion repo to the
[PANIC! art installation](https://cybernetics.anu.edu.au/news/2022/11/22/panic-a-serendipity-engine/)
(first exhibited 2022) to explore the questions raised in a more systematic
manner.

## Requirements

- python 3.12+
- GPU which supports CUDA 12.7 (maybe earlier ok, but untested)

## Why?

At the School of Cybernetics we love thinking about the way that feedback loops
(and the the connections between things) define the behaviour of the systems in
which we live, work and create. That interest sits behind the design of PANIC!
as a tool for making (and breaking!) networks of hosted generative AI models.

Anyone who's played with (or watched others play with) PANIC! has probably had
one of these questions cross their mind at some point.

One goal in building PANIC is to provide answers to these questions which are
both quantifiable and satisfying (i.e. it feels like they represent deeper
truths about the process).

### how did it get _here_ from _that_ initial prompt?

- was it predictable that it would end up here?
- how sensitive is it to the input, i.e. would it still have ended up here with
  a _slightly_ different prompt?
- how sensitive is it to the random seed(s) of the models?

### is it stuck?

- the text/images it's generating now seem to be "semantically stable"; will it
  ever move on to a different thing?
- can we predict in advance which initial prompts lead to a "stuck" trajectory?

### has it done this before?

- how similar is this run's trajectory to previous runs?
- what determines whether they'll be similar? the initial prompt, or something
  else?

### which parts of the system have the biggest impact on what happens?

- does a certain GenAI model "dominate" the behaviour of the network? or is the
  prompt more important? or the random seed? or is it an emergent property of
  the interactions between all models in the network?

## Setup

If you value your sanity you'll use [uv](https://docs.astral.sh/uv/) to manage
your python environment, and then do this:

```
uv pip install -e .
```

## Authors

Ben Swift, Sunyeon Hong

## Licence

MIT
