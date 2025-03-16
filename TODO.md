# TODO

- the model cache sortof works, but needs to be manually cleared. there's an AI
  suggestion, but perhaps it could be done at an engine-level (the "how much gpu
  memory is free", then evict things approach)

- engine: ensure that run is stopped if the same output is seen (and seed isn't
  -1), plus an `is_stuck` method to check (post-hoc) if that was the reason the
  run stopped. plus tests for this!

- see why "main" is required in the run-experiments cli subcommand

- use pytest-benchmark to test the model speeds

- write tests for cli commands

- shell script to run it with nohup

- silence logger.info for VIPS things

- make the logger print overall progress in terms of runs (or embeddings/PDs for
  the end stages)

- make the "test giotto runtime" iterate over pc sizes and log the times to a
  file?

- add documentation to readme at least (and clean up doco for the code)

- ensure that the program can be interrupted (and that it doesn't re-do any work
  it doesn't need to)

- add data analysis & vis code

  - pull a run or series of runs (experiment? or just everything) into a polars
    df
  - do some basic vis with altair
  - do some statistical tests with e.g. scikit.learn
  - wall-clock time plots (so we see what's taking so long)
  - overall (dim-reduced) clustering of the embeddings, colored by the different
    factor levels
  - would love to have some distribution of "length of equilibria" or similar

- work on parallelism (with shared GPU, and with multi-GPU)

- switch to a python package provided sqlite3 (maybe...)

- use the dummy embeddings in the final analysis as a control, perhaps with a
  slightly more sophisticated "random walk" scheme

- idea: have multiple "models" for each model, with (in i2t case) different
  captions, or (in t2i case) different num_steps or similar

- generators == None for "pre-initialised" PDs seems nicer semantically

- generate big image grid (utils, or maybe analysis)... or perhaps move utils
  into analysis
