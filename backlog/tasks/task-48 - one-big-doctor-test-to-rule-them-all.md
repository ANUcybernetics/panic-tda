---
id: task-48
title: one big doctor test to rule them all
status: To Do
assignee: []
created_date: "2025-08-09 02:28"
labels: []
dependencies: []
---

## Description

Move the test_experiment_doctor_with_fix test from @tests/test_engine.py to
@tests/test_doctor.py, and update it so that it:

- creates and performs a full experiment which uses all the "Dummy" genai models
  and emebdding models (DummyI2T, DummyI2T2, DummyT2I, DummyT2I2, DummyText,
  DummyText2, DummyVision, DummyVision2) and a max length of 100 (much longer
  than some of the other tests)

- runs a few basic checks to see that the experiment was performed correctly
  (but it doesn't have to be exhaustive---the point of this test is to check the
  doctor functionality)

- then, strategically deletes and/or corrupts the data in the db in a systematic
  way (such that the doctor checks would fail)

- runs the doctor checks and asserts that the correct checks pass and fail

- then, runs doctor with `--fix`

- finally, runs doctor again to assert that all the checks pass

This test can be marked as `slow` and so will be excluded by default. And it
doesn't have to test every single permutation of possible missing/corrupted
data. But it's important that it checks that every part of the doctor
functionality in @src/panic_tda/doctor.py works, both in the "can detect the
problem" and "can fix it" sense, for all types of invocations and all types of
embedding models (text and image).
