---
id: TASK-67
title: Redesign video export layout to fill full frame with constraint-based grid
status: To Do
assignee: []
created_date: '2026-02-20 01:56'
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
- [ ] #1 Images scale to fill the full HD/4K frame (no 512px cap, minimal dead margin)
- [ ] #2 Each subgrid is labelled with network name and prompt
- [ ] #3 Subgrid arrangement of runs targets 16:9 aspect ratio
- [ ] #4 Outer grid arrangement of subgrids also targets 16:9
- [ ] #5 Fixed-pixel gutters between subgrids
- [ ] #6 Unsupported run counts or network×prompt counts raise clear errors
- [ ] #7 Layout logic has comprehensive ExUnit tests
<!-- AC:END -->
