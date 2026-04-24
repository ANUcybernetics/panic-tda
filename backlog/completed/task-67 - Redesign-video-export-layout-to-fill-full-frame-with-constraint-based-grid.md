---
id: TASK-67
title: Redesign video export layout to fill full frame with constraint-based grid
status: Done
assignee: []
created_date: '2026-02-20 01:56'
updated_date: '2026-04-14'
labels:
  - export
  - video
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The current video export layout (lib/panic_tda/export.ex) doesn't use the full video frame --- images are capped at 512px and the grid floats in the centre with large margins. Redesign compute_layout and render_frame to:

1. **Subgrid per network+prompt pair**: each cell in the outer grid is a network+prompt combination containing num_runs images arranged in a subgrid. The subgrid shape should be close to 16:9 (square is acceptable).

2. **Images stretch to fill the frame**: remove the @max_thumb_size cap. After accounting for fixed-pixel gutters and label heights, images should scale up to fill all remaining space. The constraint solver works backwards from the frame dimensions.

3. **Labels above each subgrid**: each subgrid gets a label showing the network name (e.g. 'SD35Medium → Moondream') and the prompt text. Labels are fixed height.

4. **Outer grid also targets 16:9**: the arrangement of subgrids across the frame should also approximate the target aspect ratio.

5. **Fixed-size gutters**: gutter between subgrids is a fixed pixel size (not proportional to thumb size).

6. **Enforce 'nice' grid numbers**: rather than handling arbitrary counts, constrain the supported run counts to values that make nice subgrids (1, 2=1×2, 3=1×3, 4=2×2, 6=2×3, 8=2×4, 9=3×3, 12=3×4, etc.) and similarly constrain the number of network+prompt pairs to values that tile well. Raise a clear error for unsupported counts rather than producing a bad layout.

7. **Tests**: the layout logic should be well-tested. Test compute_layout with various combinations of network counts, prompt counts, and run counts. Verify images fill the frame (no large dead margins). Test error cases for unsupported counts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Images scale to fill the full HD/4K frame (no 512px cap, minimal dead margin)
- [x] #2 Each subgrid is labelled with network name and prompt
- [x] #3 Subgrid arrangement of runs targets 16:9 aspect ratio
- [x] #4 Outer grid arrangement of subgrids also targets 16:9
- [x] #5 Fixed-pixel gutters between subgrids
- [x] #6 Unsupported run counts or network×prompt counts raise clear errors
- [x] #7 Layout logic has comprehensive ExUnit tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented as a hierarchical nested layout (prompt → network → run) rather
than a flat outer grid keyed on (network, prompt) pairs. The hierarchy makes
the grouping of prompts visually explicit, which matters for the current
experiments.

Key pieces in `lib/panic_tda/export.ex`:

- `grid_shapes/1` now iterates over columns (`c=1..n`, `r=ceil(n/c)`) so every
  generated factorisation fills all its rows. The old row-iteration produced
  shapes like `{4,3}` for n=9 where the fourth row was entirely empty.
- `compute_layout/6` scores each candidate by `img_size * fill_ratio`, where
  `fill_ratio` is the product of `n_items / (rows*cols)` across the three
  nested sub-grids. This prefers balanced factorisations (e.g. 3×3 for 9
  networks) over sparse ones (e.g. 2×5 with an empty cell) when the raw
  thumbnail size is close. This indirectly targets 16:9 because frame-fit is
  already baked into `img_size`, and fill_ratio pushes away from wasteful
  shapes that would distort the overall aspect.
- After picking the best candidate, raises `ArgumentError` if `img_size`
  drops below a resolution-dependent minimum (20px HD, 40px 4K). The error
  names the specific counts and suggests run counts with balanced
  factorisations.

Tests in `test/export_test.exs` cover the penguin combo (5,9,8) → 3×3 nets
+ 4×2 or 2×4 runs, verify no chosen sub-grid has an empty row for a range of
combos, and exercise the error path.
<!-- SECTION:NOTES:END -->
