---
id: task-35
title: identify slow tests and see if they can be sped up
status: To Do
assignee: []
created_date: "2025-07-29"
labels: []
dependencies: []
---

## Description

pytest has a way to run all tests and print the N slowest ones. Run the full
test suite to identify the slowest 10 tests, and identify any easy ways to speed
them up. If the slowness is unavoidable without complicating the testing setup,
then we'll just let them be slow.
