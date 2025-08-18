---
id: task-44
title: Make doctor command output more detailed with specific issues
status: Done
assignee: []
created_date: '2025-08-07 06:31'
updated_date: '2025-08-15 01:01'
labels: []
dependencies: []
---

## Description

Instead of just showing counts like 'run invocation issues', 'embedding issues',
etc., the doctor command should list the specific issues found. The table can
have more rows to accommodate the detailed information.

## Requirements

1. Update the doctor command output format to show specific issues rather than
   just summary counts
2. For each category of issues (runs, embeddings, persistence diagrams, etc.),
   list:
   - The specific entity ID (run ID, invocation ID, etc.)
   - The exact nature of the problem (e.g., "missing invocation at sequence 5",
     "no embedding for model X", etc.)
3. The table format should be adjusted to accommodate more detailed information
4. Keep the summary statistics but add detailed breakdowns below each category

## Technical Details

- The doctor command is implemented in `src/panic_tda/doctor.py`
- It's called from `src/panic_tda/main.py` in the `doctor_command` function
- Currently outputs a summary table with issue counts
- Should maintain both text and JSON output formats

## Acceptance Criteria

- [x] Doctor command shows specific issue details for each problem found
- [x] Output clearly identifies which runs/invocations/embeddings have issues
- [x] Table format remains readable despite additional rows
- [x] Both text and JSON output formats work correctly
- [x] Tests pass without failures

## Implementation Notes

Updated the doctor.py output formatting to show:
- First 10 issues of each type with specific details
- Run/invocation/experiment IDs (truncated to 8 chars for readability)
- Exact nature of each issue (missing sequences, duplicate embeddings, etc.)
- Summary count if more than 10 issues exist
- Maintains both text and JSON output formats
