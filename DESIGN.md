---
title: PANICxTDA Software Design Document
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
by these genAI models.

## This panic-tda repository

Motivated by this (and our own curiosity) we set out in this `panic-tda` repo to
create a tool for exploring the patterns in these information trajectories in a
more systematic way.

`panic-tda` has different priorities to the actual PANIC installation: in
particular it's designed to run many (thousands) of experiments in batch mode,
rather than one-at-a-time for an (in-person) audience. The design priorities for
this tool are:

- self-hosted: the PANIC! installation's use of cloud-hosted models is a
  trade-off; it makes it trivial to continue to keep the installation up-to-date
  with the latest text/image/audio genAI models, but with a cost-per-request
  overhead that becomes prohibitive when trying to run batch-mode simulation
  experiments at scale

- open models: while the adjective "open" is
  [contentious](https://www.nature.com/articles/s41586-024-08141-1), because of
  the need to self-host `panic-tda` is set up to use open-weight models (e.g.
  those from [HuggingFace](https://huggingface.co/models))

- runs on commodity(ish) hardware: as academic researchers we have limited
  resources, and so the tool is designed to work well with genAI models that
  will run on a single high-end gaming GPU (at least, this has influenced the
  initial selection of models in the `GenAI` module; there's nothing stopping
  more resource-intensive models being added in future)

- reproducible research: if you're interested in the same questions about the
  way that information flows through these nonlinear, multi-billion parameter
  information processing systems, then do [get in touch](ben.swift@anu.edu.au)
  --- but this tool has been designed such that anyone with access to a decent
  gaming rig can reproduce our results

## Domain data model

First, some nomenclature (since many of these terms are overloaded). In the
`panic-tda` tool we use these terms:

- **genAI model**: a particular AI model (e.g. _Stable Diffusion_, _Moondream_)
  which takes text or image input and produces text or image output
  (implementations are in `lib/panic_tda/models/genai.ex`)
- **network**: a list of genAI models that cycle --- the output of one is fed as
  input to the next, wrapping around when the list is exhausted (e.g.
  `["SDXLTurbo", "Moondream"]` alternates between text-to-image and
  image-to-text)
- **invocation**: a specific "inference" event for a single model; includes both
  the input and the output along with timestamps and sequence metadata
- **run**: a specific sequence of invocations starting from an initial prompt
  and following the models in a network for `max_length` steps
- **embedding model**: an
  [embedding model](https://huggingface.co/blog/getting-started-with-embeddings)
  (e.g. _RoBERTa_, _Nomic_) which takes text or image input and returns a
  vector in a high (e.g. 768)-dimensional space such that inputs that are
  "semantically similar" are close together in this space (implementations are
  in `lib/panic_tda/models/embeddings.ex`)
- **experiment**: a specification for a batch of runs with a given network,
  set of prompts, embedding models and number of runs per prompt

The data model is implemented using
[Ash](https://hexdocs.pm/ash/get-started.html) resources backed by SQLite (via
[AshSqlite](https://hexdocs.pm/ash_sqlite/)). The resources live in
`lib/panic_tda/resources/` and include custom types for vectors, images and
persistence diagram data. See the **Use** section of the
[README](./README.md#use) for how to specify and perform an experiment.

## Compute workflow

The code for performing experiments is in the `lib/panic_tda/engine/` modules.
The `Engine.perform_experiment/1` function is the main workhorse.

For a given experiment config, the engine first creates all runs --- one for each
combination of prompt and run number (from `0` to `num_runs - 1`). Each run
shares the experiment's network and `max_length`.

The computation then proceeds through four stages:

- In the **runs stage** (`RunExecutor.execute_batch/2`) all runs advance through
  the network in lockstep. At each sequence step, the current model is invoked
  with a batch of inputs (one per run) in a single GPU call, and the outputs are
  recorded as `Invocation` records. This is significantly more efficient than
  running each run independently, since the model only needs to be loaded once
  per step rather than once per run.

- In the **embeddings stage** (`EmbeddingsStage.compute/3`) each run's text
  invocations are embedded using all of the text embedding models specified in
  the experiment config, and image invocations are embedded using image embedding
  models. Each embedding is stored as a float32 binary vector.

- In the **persistence diagram stage** (`PdStage.compute/3`) each run's sequence
  of embeddings (under a particular embedding model) is processed by the
  [giotto-ph](https://giotto-ai.github.io/giotto-ph/) library to compute a
  [persistence diagram](https://en.wikipedia.org/wiki/Persistent_homology).
  There is one persistence diagram per run per embedding model.

- In the **clustering stage** (`ClusteringStage.compute/3`) all persistence
  diagrams for a given embedding model are clustered using
  [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)
  to identify groups of runs with similar topological structure. Medoid
  indices are computed for each cluster.

All stages support resuming --- if an experiment is interrupted, each stage can
detect which work has already been completed and pick up where it left off via
`Engine.resume_experiment/1`.

### Python interop

GPU model invocation, embedding computation, TDA and clustering all happen in
Python. The Elixir side manages orchestration, data persistence and the
experiment lifecycle, while Python handles the numerical heavy lifting.

Python interop is via [Snex](https://hexdocs.pm/snex/), which maintains a
persistent Python interpreter with shared state across calls. This means models
are loaded once into GPU memory and reused across invocations, and the Python
environment (imports, variables, loaded models) persists for the lifetime of an
experiment. The Python code is written inline in the Elixir source files ---
there are no separate `.py` files to maintain.

Model loading is lazy: the `PythonBridge` module ensures each model is loaded at
most once per experiment, and `PythonBridge.unload_all_models/1` frees GPU
memory between stages.

### Current hardware setup

All of our experiments have been done on a single machine with the following
specs:

- AMD Threadripper 24C/48T
- 128 GB RAM
- RTX 6000 ADA 48GB GPU

## How does the TDA fit in?

TODO.
