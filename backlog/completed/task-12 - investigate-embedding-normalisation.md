---
id: TASK-12
title: investigate embedding normalisation
status: Done
assignee: []
created_date: '2025-07-21'
updated_date: '2026-02-09 06:35'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The embedding models support (I think) auto-normalisation, which would make some
of the "normalise so that euclidean == cosine" login in
@src/panic_tda/clustering.py redundant (esp. the "un-normlise medoid to get the
label back" part).

I need to investigate whether this is a better option that the current approach.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed: embedding normalisation is now implemented
<!-- SECTION:NOTES:END -->
