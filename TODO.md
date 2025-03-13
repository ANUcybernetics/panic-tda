# TODO

- ensure that the program can be interrupted (and that it doesn't re-do any work
  it doesn't need to)

- add tda stuff to engine (but at the end?)

- add data analysis & vis code

  - pull a run or series of runs (experiment? or just everything) into a polars
    df
  - do some basic vis with altair
  - do some statistical tests with e.g. scikit.learn
  - wall-clock time plots (so we see what's taking so long)

- work on parallelism (with shared GPU, and with multi-GPU)

- switch to a python package provided sqlite3 (maybe...)

- use the dummy embeddings in the final analysis as a control, perhaps with a
  slightly more sophisticated "random walk" scheme

- idea: have multiple "models" for each model, with (in i2t case) different
  captions, or (in t2i case) different num_steps or similar

- generators == None for "pre-initialised" PDs seems nicer semantically

- generate big image grid (utils, or maybe analysis)... or perhaps move utils
  into analysis
