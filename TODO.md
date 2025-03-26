# TODO

- use ray actors for each genai/embedding model (for better parallelism)

- store the persistence entropy in the PD object

- switch to fast flux (fp8) using
  [this](https://github.com/aredden/flux-fp8-api)

- add data analysis & vis code

  - plot loop length for all "stopped because of loop" cases
  - plot persistence diagrams and entropy
  - store the persistence entropy in the PD object
  - visualise the time taken for the various different invocations
  - do some statistical tests with e.g. scikit.learn
  - overall (dim-reduced) clustering of the embeddings, colored by the different
    factor levels
  - would love to have some distribution of "length of equilibria" or similar

- use the dummy embeddings in the final analysis as a control, perhaps with a
  slightly more sophisticated "random walk" scheme

- idea: have multiple "models" for each model, with (in i2t case) different
  captions, or (in t2i case) different num_steps or similar

- generate big image grid (utils, or maybe analysis)... or perhaps move utils
  into analysis

- run the tests in GitHub actions

- batch those suckers, all of them. might get 5x just like that.
