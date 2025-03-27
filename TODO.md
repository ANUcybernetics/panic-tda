# TODO

- clustering (with optional dimensionality reduction)

- visualise the time taken for the various different invocations

- do some statistical tests with e.g. scikit.learn

- use the dummy embeddings in the final analysis as a control, perhaps with a
  slightly more sophisticated "random walk" scheme

- idea: have multiple "models" for each model, with (in i2t case) different
  captions, or (in t2i case) different num_steps or similar

- generate big image grid (utils, or maybe analysis)... or perhaps move utils
  into analysis

- run the tests in GitHub actions

- batch the genai models as well (embedding ones already done)

- store the loop_length in the invocation (maybe)

- store the persistence entropy in the PD object (maybe)

- load dfs directly from the databases (because polars can do that, but the
  calculated properties like loop_length become a PITA)

- write an orphans (or some other validation that the run is all there)
  property/method for Run
