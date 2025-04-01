# TODO

- ensure persistence diagrams and mosaic videos use the same layout algo (for
  ease of comparison between the two)

- faceted persistence entropy plot

- look at change-point detection with ruptures library, and visualise the run
  lengths

- statistical tests on PE across different networks, prompts, seeds, and
  embedding models

- experiment with actor pools for the run stage (because e.g. SDXLTurbo can
  certainly fit a few copies at once)

- (related to previous) change engine run generation phase to only load the
  models necessary for a particular run (since a single run pegs the GPU anyway)
  to allow for multiple runs to be queued up without fear of OOM

- use the dummy embeddings in the final analysis as a control, perhaps with a
  slightly more sophisticated "random walk" scheme

- visualise the time taken for the various different invocations

- run the tests in GitHub actions

- batch the genai models as well (embedding ones already done)

- store the loop_length in the invocation (maybe)

- write an orphans (or some other validation that the run is all there)
  property/method for Run. Or maybe just a cleanup function

- embeddings: are they normalised? should they be?

- create similarity matrices for runs
