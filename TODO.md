# TODO

- prompt ideas: apple/pear/banana, man/woman/person, car/train/boat, rainbow
  colours

- statistical tests on PE across different networks, prompts, seeds, and
  embedding models

- tSNE chart would be cool/helpful (to see whether the different runs get
  clustered together)

- ensure persistence diagrams and mosaic videos use the same layout algo (for
  ease of comparison between the two)

- use tqdm for the video export progress (and other long-running but
  non-distributed things)

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

- experiment with actor pools for the run stage (because e.g. SDXLTurbo can
  certainly fit a few copies at once)

- switch to onnx for the inference workflows

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
