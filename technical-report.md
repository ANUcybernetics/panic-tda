---
title: PANICxTDA Technical Report
author: Ben Swift
---

## Background and motivation

The
[**PANIC!** (Playground AI Network for Interactive Creativity) interactive art installation](https://cybernetics.anu.edu.au/news/2022/11/22/panic-a-serendipity-engine/),
first exhibited at the
[ANU School of Cybernetics](https://cybernetics.anu.edu.au/) in 2022, is an
interactive installation that explores the creative potential of connected
generative AI models. Viewers enter text prompts which are transformed through a
"network" of generative AI (genAI) models, producing
[AI images](https://mediarep.org/server/api/core/bitstreams/72370317-e47d-4b4f-9d09-dbc2e12bbc7c/content),
text and audio. Each output becomes the input for the next iteration, creating
an endless cycle of AI-mediated transformation.

The PANIC! installation is a combination of software and hardware:

- a bespoke cloud-hosted webapp
  ([which is open source](https://github.com/anucybernetics/panic)) powering the
  interface and orchestrating the software/hardware co-ordination
- various cloud-hosted genAI models (including
  [replicate](https://replicate.com/), [OpenAI](https://openai.com/) and
  [Google AI Studio](https://ai.google.com/studio))
- a wall of displays with 4x Vestaboards (for text display) and 4x 4K TVs (for
  image & audio display)

Over the past few years, many visitors have observed the way that their initial
text input (prompt) is transformed as the information is recursively transformed
by these genAI models. Motivated by this (and our own curiosity) we set out in
this `panic-tda`repo to create a tool for exploring the patterns in these
information trajectories in a more systematic way.

`panic-tda` has different priorities to the actual PANIC installation: in
particular it's designed to run many (thousands) of experiments in batch mode,
rather than one-at-a-time for an (in-person) audience. The design priorities for
this tool are:

- self-hosted: the PANIC! installation's use of cloud-hosted models is a
  trade-off; it makes it trivial to continue to keep the installation up-to-date
  with the latest text/image/audio genAI models, but with a cost-per-request
  ovehead that becomes provibitive when trying to run batch-mode simulation
  experiments at scale

- open models: while the adjective "open" is
  [contentious](https://www.nature.com/articles/s41586-024-08141-1), because of
  the need to self-host `panic-tda` is set up to use open-weight models, (e.g.
  those from [HuggingFace](https://huggingface.co/models))

- runs on commodity(ish) hardware: as academic researchers we have limited
  resources, and so the tool is designed to work well with genAI models that
  will run on a single high-end gaming GPU (at least, this has influenced the
  initial selection of models implemented in the `genai_models` module; there's
  nothing stoppoing more resource-intensive models being added in future)

- reproducible research: if you're interested in the same questions about the
  way that information flows through these nonlinear, multi-billion parameter
  information processing systems, then do [get in touch](ben.swift@anu.edu.au) -
  but this tool has been designed such that anyone with access to a decent
  gaming rig can reproduce our results (including the analysis and datavis
  parts---also in this repo)

## Domain data model

First, some nomenclature (since many of these terms are overloaded). In the
`panic-tda` tool we use these terms:

- **genai model**: a particular AI model (e.g. _Stable Diffusion_, _GPT4o_)
  which takes text/image/audio input and produces text/image/audio output
  (implementations are in `src/panic_tda/genai_models.py`)
- **network**: a specific network (i.e. cyclic graph) of genai models, designed
  so that the output of one is fed as input to the next
- **invocation**: a specific "inference" event for a single model; includes both
  the input (prompt) an the output (prediction) along with some other metadata
- **run**: a specific sequence of predictions starting from an initial prompt
  and following the models in a network
- **embedding model**: an
  [embedding model](https://huggingface.co/blog/getting-started-with-embeddings)
  (e.g. _RoBERTa_, _Nomic_) which takes text input and returns a vector in a
  high (e.g. 768)-dimensional space such that text inputs that are "semantically
  similar" are close together in this space (implementations are in
  `src/panic_tda/embeddings.py`)
- **experiment**: a specification for a sequence of runs, with different
  prompts, networks, embedding models and random seeds (this abstraction is
  primarily used for organizing and managing experiments, e.g. batch job
  submission)

This repo uses [Pydantic](https://pydantic.dev) for data modelling and
validation and the related [sqlmodel](https://sqlmodel.tiangolo.com) for
persisting data to a sqlite database. Have a look at the `schema` module for the
details.

For info on how to specify and perform an experiment, see the **Use** section of
the [README](./README.md#use).

## Compute workflow

The code for performing the experiments is done by the `src/panic_tda/engine.py`
module. The `perform_experiment` function is the main workhorse.

For a given experiment config, the tool will first enumerate (cartesian product)
all the combinations of networks, prompts and random seeds. Each element of this
enumeration represents a run, which will be performed until the experiment
config's `max_length` is reached, and a `Run` object is created with the
appropriate attributes.

After this, the computation has three distinct stages:

- in the **runs** stage (`perform_runs_stage`) each run is iterated upon for the
  specified network from the initial prompt, producing (eventually) a sequence
  of invocations. The compute graph here is linear (because each invocation
  depends on the previous one for it's input), but each run is independent of
  the others, so multiple runs can happen in parallel.

- In the **embeddings** stage (`perform_embeddings_stage`) each invocation is
  embedded using the specified embedding model, producing a sequence of
  embeddings. The compute graph here is linear (because each embedding depends
  on the previous one for it's input), but each embedding is independent of the
  others, so multiple embeddings can happen in parallel.

- In the **embeddings** stage (`perform_embeddings_stage`) each invocation is
  embedded using all of the embedding models specified in the experiment config.
  These computations are completely independent---this stage is embarrasingly
  parallel.

- In the **persistence diagram** stage (`perform_pd_stage`) each "run of
  embeddings" is processed by the
  [giotto](https://giotto-ai.github.io/giotto-ph/) library to create a
  [persistent homology](https://en.wikipedia.org/wiki/Persistent_homology) for
  that whole run (well, for the whole run under a particular embedding
  model---each run can be embedded with multiple embedding models and will have
  one persistence diagram for each one). This stage is also embarrassingly
  parallel, although at a smaller scale than the embeddings stage (because
  there's one computation per run, not one per invocation).

For parallelizing the experiments, `panic-tda` uses
[ray](https://docs.ray.io/en/latest/). As discused above, the first stage has
some linear dependencies (the implementation handles this using ray's
[dynamic generators](https://docs.ray.io/en/latest/ray-core/tasks/generators.html),
one per run).

The embedding stage uses an
[actor pool](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.ActorPool.html)
to better use the available compute resources. The persistence diagram stage
does not use a GPU, and just uses ray tasks for parallel execution (with careful
management of the ray decorator's `num_gpus` parameter and the giotto library's
`n_threads` parameter to balance time/memory usage).

One other note: each stage (and each ray actor) does need access to the sqlite
database (on the local filesystem) because of the way that ray serialises the
task arguments and return values. While in practice this has not proved to be a
bottleneck (and we use sqlite in WAL-mode for efficiency) this is worth keeping
in mind.

### Current hardware setup

All of our experiments have been done on a single machine with the following
specs:

- AMD Threadripper 24C/48T
- 128 GB RAM
- RTX 6000 ADA 48GB GPU

The ray resource parameters (grep through the codebase for `num_gpus` and
`num_cpus`) are tuned for these specs; if you're running it on your own hardware
you should investigate what works best for you.

In the future we plan to investigate how to scale this up to larger distributed
computing contexts. The ideal configuration for the runs and embeddigs stages
would be relatively "thin" nodes with multiple GPUs attached and actor pools to
manage the genAI and embedding model actors to maximise hardware utilisation.
The persistence diagram stage is slightly different, and could benefit from
"fat" nodes with beefier CPU and RAM resources (especially as the `max_length`
parameter gets up past 10k).

Since this project is fundamentally about looking at statistical patterns in the
genAI & embedding outputs, this tool does not need the high-bandwidth
interconnets typically associated with HPC. The easiest way to take advantage of
a distributed computing context is to run many experiments in parallel, and
combine the data from the multiple generated sqlite databases offline
afterwards.

## How does the TDA fit in?

TODO.
