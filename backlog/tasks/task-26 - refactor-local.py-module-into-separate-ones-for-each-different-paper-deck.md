---
id: task-26
title: refactor local.py module into separate ones for each different paper/deck
status: To Do
assignee: []
created_date: "2025-07-27"
labels: []
dependencies: []
---

## Description

The @local.py module now contains code for producing visuals and analysis for
two separate documents. There's

- ieee_smc_charts
- artificial_futures_slides_charts

and all the code they depend on

Finally, there's the paper_charts function at the bottom.

Refactor (potentially adding one more level of module nesting via a `local/`
subfolder) into separate modules for each of those (and potentially more to come
in the future). Ensure that `export charts` (in @main.py) still works (and I'm
fine having the specific module-du-jour hardcoded in paper_charts, because that
doesn't need to change often).

This will help with keeping the claude context relevant. Also, it'd be nice if
the "warn on unused imports" linter thing was turned off for these files (since
they change so much).
