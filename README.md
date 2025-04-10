# Generative AI Trajectory Tracer

This repo is a command-line tool:

1. connecting up text->image and image->text generative AI models in various
   "network topologies"
2. starting from a specified initial prompt & random seed, recursively passing
   the output of one model in as the input of the next to create a "run" of
   model invocations
3. embedding each output (text or image) into a joint embedding space using a
   multimodal embedding model
4. storing everything (genAI model outputs & embedding outputs) in a local
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
- GPU which supports CUDA 12.7 (earlier version maybe earlier ok, but untested)

## Installation

With [uv](https://docs.astral.sh/uv/), it's just:

```bash
# install the package
uv pip install -e .

# run the main CLI
uv run panic-tda --help
```

If you're the sort of person who has alternate (strong) opinions on how to
manage python environments, then you can probably figure out how to do it your
preferred way. Godspeed to you.

## Use

The main CLI is `panic-tda`. It has a few subcommands:

- `perform-experiment`: Run a trajectory tracer experiment defined in a
  configuration file
- `list-runs`: List all runs stored in the database, with options for detailed
  output
- `list-models`: List all the supported models (both genAI t2i/i2t and embedding
  models)
- `export-images`: Export images from one or all runs to JPEG files with
  embedded metadata

To run an experiment, you'll need to create a configuration file. Here's an
example:

```json
{
  "networks": [["FluxDev", "Moondream"]],
  "seeds": [42, 56, 545654, 6545],
  "prompts": ["Test prompt 1"],
  "embedding_models": ["Nomic"],
  "max_length": 100
}
```

The schema for the configuration file is based on `ExperimentConfig` in the
`schema` module (that's the class it needs to parse cleanly into).

Then, to "run" the experiment:

```bash
# Run an experiment with the above configuration
panic-tda perform-experiment my_config.json

# List all runs in the database
panic-tda list-runs

# Export images from a specific run
panic-tda export-images 123e4567-e89b-12d3-a456-426614174000
```

If you're running it on a remote machine and kicking it off via ssh, you'll
probably want to use `nohup` or `tmux` or something to keep it running after you
log out (see `perform-experiment.sh` for an example).

## Repo structure

This repo uses [Pydantic](https://pydantic.dev) for data modelling and
validation and the related [sqlmodel](https://sqlmodel.tiangolo.com) for
persisting data to a sqlite database. The data model is described in the
`schema` module.

The code for performing the experiments is done by the `engine` module. All
other modules do what they say on the tin. For parallelizing the experiments, we
use [ray](https://docs.ray.io/en/latest/).

There are (relatively) comprehensive tests in the `tests` directory. They can be
run with pytest:

    uv run pytest

A few of the tests (anything which involves actual GPU processing) are marked
slow and won't run by default. To run them, use:

    uv run pytest -m slow

If you'd like to add a new genAI or embedding model, have a look in
`genai_models` or `embeddings` respectively, and add a new subclass which
implements the desired interface. Look at the existing models and work from
there - and don't forget to add a new test to `tests/test_genai_models.py` or
`tests/test_embeddings.py`.

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

## Authors

[Ben Swift](https://benswift.me) wrote the code, and Sunyeon Hong is the
mastermind behind the TDA stuff.

## Licence

MIT
