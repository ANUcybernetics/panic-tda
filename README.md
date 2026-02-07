# PANIC-TDA

An Elixir tool for computing "runs" of text-to-image and image-to-text models
(with outputs fed recursively back in as inputs) and analysing the resulting
text-image-text-image trajectories using
[topological data analysis](https://en.wikipedia.org/wiki/Topological_data_analysis).

If you've got a [sufficiently capable rig](#requirements) you can use this tool
to:

1. specify text-to-image and image-to-text generative AI models in a "network"
   (a cycling list of models)
2. starting from a specified initial prompt, recursively iterate the output of
   one model in as the input of the next to create a "run" of model invocations
3. embed each output into a high-dimensional embedding space using one or more
   embedding models
4. compute persistence diagrams and cluster them to identify topological
   structure in the trajectories

The results of all the above computations are stored in a local SQLite database
for further analysis.

This tool was initially motivated by the
[**PANIC!** art installation](https://cybernetics.anu.edu.au/news/2022/11/22/panic-a-serendipity-engine/)
(first exhibited 2022) --- see [DESIGN.md](./DESIGN.md) for more details.
Watching PANIC! in action, there is clearly some structure to the trajectories
that the genAI model outputs "trace out". This tool is an attempt to quantify
and understand that structure (see [_why?_](#why) below).

## Requirements

- [mise](https://mise.jdx.dev/) for managing Erlang/Elixir versions (see
  `mise.toml`)
- a GPU which supports CUDA (for running the genAI and embedding models)
- SQLite3

## Installation

```bash
# install Erlang & Elixir via mise
mise install

# fetch deps and set up the database
mise exec -- mix setup
```

## Use

Experiments are configured via JSON files and run with Mix tasks. Here's an
example configuration:

```json
{
  "network": ["SDXLTurbo", "Moondream"],
  "prompts": ["a red apple"],
  "embedding_models": ["Nomic"],
  "max_length": 100,
  "num_runs": 4
}
```

Fields:

- **network**: a list of models that cycle (T2I -> I2T -> T2I -> ...)
- **prompts**: initial text inputs; each prompt creates `num_runs` runs
- **embedding_models**: models used in the embeddings stage
- **max_length**: number of model invocations per run
- **num_runs** (optional, default 1): how many runs to create per prompt

Then, to run the experiment:

```bash
# run an experiment
mise exec -- mix experiment.run config/my_experiment.json

# check the status of an experiment (by ID prefix)
mise exec -- mix experiment.status abc123

# list all experiments
mise exec -- mix experiment.list

# resume an interrupted experiment
mise exec -- mix experiment.resume abc123
```

### Available models

| Type | Models |
|---|---|
| text-to-image | `SDXLTurbo`, `FluxDev`, `FluxSchnell` |
| image-to-text | `Moondream`, `BLIP2` |
| text embedding | `STSBMpnet`, `STSBRoberta`, `STSBDistilRoberta`, `Nomic`, `JinaClip` |
| image embedding | `NomicVision`, `JinaClipVision` |
| dummy (testing) | `DummyT2I`, `DummyI2T`, `DummyT2I2`, `DummyI2T2`, `DummyText`, `DummyText2`, `DummyVision`, `DummyVision2` |

## Testing

Tests use ExUnit with dummy models (no GPU required):

    mise exec -- mix test

GPU smoke tests (all real model combinations) are tagged `:gpu` and excluded by
default:

    mise exec -- mix test --include gpu

For further info, see the [design doc](./DESIGN.md).

## Why?

At the [School of Cybernetics](https://cybernetics.anu.edu.au) we love thinking
about the way that feedback loops (and the connections between things) define the
behaviour of the systems in which we live, work and create. That interest sits
behind the design of PANIC! as a tool for making (and breaking!) networks of
hosted generative AI models.

Anyone who's played with (or watched others play with) PANIC! has probably had
one of these questions cross their mind at some point.

One goal in building PANIC is to provide answers to these questions which are
both quantifiable and satisfying (i.e. it feels like they represent deeper truths
about the process).

### how did it get _here_ from _that_ initial prompt?

- was it predictable that it would end up here?
- how sensitive is it to the input, i.e. would it still have ended up here with
  a _slightly_ different prompt?

### is it stuck?

- the text/images it's generating now seem to be "semantically stable"; will it
  ever move on to a different thing?
- can we predict in advance which initial prompts lead to a "stuck" trajectory?

### has it done this before?

- how similar is this run's trajectory to previous runs?
- what determines whether they'll be similar? the initial prompt, or something
  else?

### which parts of the system have the biggest impact on what happens?

- does a certain genAI model "dominate" the behaviour of the network? or is the
  prompt more important? or is it an emergent property of the interactions
  between all models in the network?

## Authors

[Ben Swift](https://benswift.me) wrote the code, and Sunyeon Hong is the
mastermind behind the TDA stuff.

## Licence

MIT
