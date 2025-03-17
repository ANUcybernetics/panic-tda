# TODO

- silence logger.info for VIPS things

- change run_length to max_run_length

- check that by default both slow and benchmark tests are skipped

- update the README with the new list-models command, and use real models in the
  example config file in the README

- add a note to the readme about how to add new models

- add doco for all the schemas (and point to that in the main readme)

- ensure that the program can be interrupted (and that it doesn't re-do any work
  it doesn't need to)

- switch to fast flux (fp8) using
  [this](https://github.com/aredden/flux-fp8-api)

- add data analysis & vis code

  - pull a run or series of runs (experiment? or just everything) into a polars
    df
  - do some basic vis with altair
  - do some statistical tests with e.g. scikit.learn
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
