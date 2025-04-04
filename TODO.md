# TODO

- nicer alignment of the prompts in the factored charts

- build out mixed modelling testing

- make list.embeddings a dict, keyed by embedding model

- maybe unplug the screen for weddle? probably not necessary (cybersonic, otoh)

- check that ExperimentConfig deletion cascades to all runs (and therefore all
  invocations & embeddings)

- run a script to find orphaned runs and invocations in the db (not attached to
  an experiment config)

- statistical tests on PE across different networks, prompts, seeds, and
  embedding models (look into contrast tests)

- look into indexes for the db

- tSNE chart would be cool/helpful (to see whether the different runs get
  clustered together)

- ensure persistence diagrams and mosaic videos use the same layout algo (for
  ease of comparison between the two)

- chart ideas:

  - [this one](https://altair-viz.github.io/gallery/select_detail.html) with PE
    on left, and PD on the right
  - add [strips](https://altair-viz.github.io/gallery/dot_dash_plot.html) to the
    new PD plots
  - maybe use a
    [minimap](https://altair-viz.github.io/gallery/scatter_with_minimap.html)
  - [wrapped facets](https://altair-viz.github.io/gallery/us_population_over_time_facet.html)
  - plot the
    [images in a tooltip](https://altair-viz.github.io/case_studies/numpy-tooltip-images.html)

- add florence2 or blip3 or some other (more modern) captioning model

- [try this approach](https://gist.github.com/sayakpaul/e1f28e86d0756d587c0b898c73822c47)
  to getting flux running on cybersonic, or perhaps onnx

- experiment with actor pools for the run stage (because e.g. SDXLTurbo can
  certainly fit a few copies at once)

- use the dummy embeddings in the final analysis as a control, perhaps with a
  slightly more sophisticated "random walk" scheme

- visualise the time taken for the various different invocations

- run the tests in GitHub actions

- try batching the genai models (in a test in the first case)

- store the loop_length in the invocation (maybe)

- write an orphans (or some other validation that the run is all there)
  property/method for Run. Or maybe just a cleanup function

- embeddings: are they normalised? should they be?

- create similarity matrices for runs
