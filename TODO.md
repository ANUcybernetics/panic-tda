# TODO

- calculate semantic dispersion (or drift)

- write a test for the persistence diagram charting function (because it doesn't
  work yet)

- look at change-point detection with ruptures library, and visualise the run
  lengths

- statistical tests on PE across different networks, prompts, seeds, and
  embedding models

- add the progress bar and initial prompt display for mosaic videos

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
