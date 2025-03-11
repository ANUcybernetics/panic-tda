# TODO

- add working (slow) tests for the nomic embedding modules

- add helper functions for (either to engine, or to db):

  - fetch a run from the db by id, and again by seed/prompt/network
  - write out the images for an invocation (or run) with prompt & other info in
    metadata

- add data analysis & vis code

  - pull a run or series of runs (experiment? or just everything) into a polars
    df
  - do some basic vis with altair
  - do some statistical tests with e.g. scikit.learn

- accelerate the models with CUDA

- work on parallelism (with shared GPU, and with multi-GPU)

- add begin + end timestamps for any calculation (so that we can do timings
  later)
