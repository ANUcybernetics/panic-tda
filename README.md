# PANIC-TDA

A python software tool for computing "runs" of text->image and image->text
models (with outputs fed recursively back in as inputs) and analysing the
resulting text-image-text-image trajectories using
[topological data analysis](https://en.wikipedia.org/wiki/Topological_data_analysis).

If you've got a [sufficiently capable rig](#requirements) you can use this tool
to:

1. speficy text->image and image->text generative AI models in various "network
   topologies"
2. starting from a specified initial prompt & random seed, recursively iterate
   the output of one model in as the input of the next to create a "run" of
   model invocations
3. embed each (text) output into a joint embedding space using a semantic
   embedding model

The results of all the above computations will be stored in a local sqlite
database for further analysis (see the `datavis` module for existing
visualizations, or write your own).

This This work design of this tool was initially motivated by the
[**PANIC!** art installation](https://cybernetics.anu.edu.au/news/2022/11/22/panic-a-serendipity-engine/)
(first exhibited 2022). Watching PANIC! in action, there is clearly some
structure to the trajectories that the genAI model outputs "trace out". This
tool is an attempt to quantify and understand that structure (see
[_why?_](#why?) below).

## Requirements

- python 3.12 (the
  [giotto-ph](https://giotto-ai.github.io/giotto-ph/build/html/installation.html)
  dependency doesn't have wheels for 3.13)
- a GPU which supports CUDA 12.7 (earlier version maybe earlier ok, but
  untested)
- sqlite3
- ffmpeg (for generating the output videos)

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

The main CLI is `panic-tda`.

```bash
$ uv run panic-tda --help

 Usage: panic-tda [OPTIONS] COMMAND [ARGS]...

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                             │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.      │
│ --help                        Show this message and exit.                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ perform-experiment   Run a panic-tda experiment defined in CONFIG_FILE.                                     │
│ resume-experiment    Resume a panic-tda experiment by its UUID.                                             │
│ list-experiments     List all experiments stored in the database.                                                   │
│ experiment-status    Get the status of a panic-tda experiment.                                              │
│ delete-experiment    Delete an experiment and all associated data from the database.                                │
│ list-runs            List all runs stored in the database.                                                          │
│ list-models          List all available genAI and embedding models with their output types.                         │
│ export-video         Generate a mosaic video from all runs in one or more specified experiments.                    │
│ doctor               Diagnose and optionally fix issues with an experiment's data.                                  │
│ paper-charts         Generate charts for publication using data from specific experiments.                          │
│ script                                                                                                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

The most useful subcommands are:

- `perform-experiment`: Run a panic-tda experiment defined in a configuration
  file
- `experiment-status`: Get the status of an experiment (% complete, broken down
  by stage: invocation/embedding/persistence diagram)
- `list-experiments`: List all experiments stored in the database, with options
  for detailed output
- `list-models`: List all the supported models (both genAI t2i/i2t and embedding
  models)
- `export-video`: Export a "mosaic" from all the runs in a given experiment

To run an experiment, you'll need to create a configuration file. Here's an
example:

```json
// experiment1.json
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
uv run panic-tda perform-experiment experiment1.json

# check the status of the experiment
uv run panic-tda experiment-status

# List all expeiments in the database
uv run panic-tda list-experiments

# Export images from a specific experiment
uv run panic-tda export-video 123e4567-e89b-12d3-a456-426614174000
```

If you're running it on a remote machine and kicking it off via ssh, you'll
probably want to use `nohup` or `tmux` or something to keep it running after you
log out (look at the `perform-experiment.sh` file for ideas on how to do this).

## Testing

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

## Further technical details

See the [technical report](./technical-report.md).

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

- does a certain genAI model "dominate" the behaviour of the network? or is the
  prompt more important? or the random seed? or is it an emergent property of
  the interactions between all models in the network?

## Authors

[Ben Swift](https://benswift.me) wrote the code, and Sunyeon Hong is the
mastermind behind the TDA stuff.

## Licence

MIT
