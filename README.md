# Generative AI Trajectory Tracer

This repo is a command-line tool:

1. connecting up text->image and image->text generative AI models in various
   "network topologies"
2. starting from a specified initial prompt & random seed, recursively passing
   the output of one model in as the input of the next to create a "run" of
   model invocations
3. embedding each output (text or image) into a joint embedding space using a
   multimodal embedding model
4. storing everything (GenAI model outputs & embedding outputs) in a local
   sqlite database
5. using
   [Topological Data Analysis](https://en.wikipedia.org/wiki/Topological_data_analysis)
   to answer questions like [the ones below](#why)

This work design of this tool was initially motivated by the
[PANIC! art installation](https://cybernetics.anu.edu.au/news/2022/11/22/panic-a-serendipity-engine/)
(first exhibited 2022). Watching PANIC! in action, there is clearly some
structure to the trajectories that the model outputs "trace out". This tool is
an attempt to quantify and understand that structure.

## Requirements

- python 3.12+
- sqlite3
- GPU which supports CUDA 12.7 (maybe earlier ok, but untested)

## Installation

Seriously, my recommendation is to just use [uv](https://docs.astral.sh/uv/).

```bash
# install the package
uv pip install -e .

# run the main CLI
uv run trajectory-tracer --help
```

Note: if you're the sort of person who has alternate (strong) opinions on how to
manage python environments, then you can probably figure out how to do it your
preferred way. Godspeed to you.

## Use

The main CLI is `trajectory-tracer`. It has a few subcommands:

- `run-experiment`: Run a trajectory tracer experiment defined in a
  configuration file
- `list-runs`: List all runs stored in the database, with options for detailed
  output
- `list-models`: List all the supported models (both GenAI t2i/i2t and embedding
  models)
- `export-images`: Export images from one or all runs to JPEG files with
  embedded metadata

To run an experiment, you'll need to create a configuration file. Here's an
example:

```json
{
  "networks": [
    ["text-to-image", "image-to-text"],
    ["image-to-text", "text-to-image"]
  ],
  "seeds": [42, 100, 500],
  "prompts": [
    "A red balloon floating in a clear blue sky",
    "The concept of time travel explained through dance"
  ],
  "embedding_models": ["clip-vit-base-patch32"],
  "max_length": 10
}
```

This configuration will run experiments with two different network topologies
(text→image→text and image→text→image), three different random seeds, and two
different prompts. Each run will consist of (max) 10 model invocations.

Example usage:

```bash
# Run an experiment with the above configuration
trajectory-tracer run-experiment my_config.json

# List all runs in the database
trajectory-tracer list-runs

# Export images from a specific run
trajectory-tracer export-images 123e4567-e89b-12d3-a456-426614174000
```

## Why?

At the [School of Cybernetics](https://cybernetics.anu.edu.au) we love thinking
about the way that feedback loops (and the the connections between things)
define the behaviour of the systems in which we live, work and create. That
interest sits behind the design of PANIC! as a tool for making (and breaking!)
networks of hosted generative AI models.

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
