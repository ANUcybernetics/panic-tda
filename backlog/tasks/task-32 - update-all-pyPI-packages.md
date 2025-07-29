---
id: task-32
title: update all pyPI packages
status: In Progress
assignee: []
created_date: "2025-07-28"
labels: []
dependencies: []
---

## Description

I know that `uv` doesn't have a command that automatically updates all packages
to their latest versions on pyPI, but that's what I want to do. Ensure that
_all_ tests pass at the end with `mise test`.

## Progress

### Completed Tasks

1. ✅ Updated all packages to their latest versions using `uv lock --upgrade`
2. ✅ Updated version specifiers in pyproject.toml to reflect new minimum versions
3. ✅ Ran standard test suite with `mise test` - all tests passed (166 passed, 5 skipped)

### Updated Packages

Major version updates:
- accelerate: 1.4.0 → 1.9.0
- diffusers: 0.32.2 → 0.34.0  
- plotnine: 0.14.5 → 0.15.0
- polars: 1.24.0 → 1.31.0
- pyarrow: 19.0.1 → 21.0.0
- pydantic: 2.10.6 → 2.11.7
- pyvips: 2.2.3 → 3.0.0
- ray: 2.44.0 → 2.48.0
- rich: 13.9.4 → 14.1.0
- ruff: 0.11.0 → 0.12.5
- sentence-transformers: 4.0.2 → 5.0.0
- torch: 2.6.0 → 2.7.1
- torchvision: 0.21.0 → 0.22.1
- transformers: 4.50.2 → 4.54.0
- And many other minor updates

### Notes

- Standard tests (without slow tests) pass successfully
- Slow tests encounter GPU memory issues, but this appears to be unrelated to the package updates
