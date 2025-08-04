# task-39 - Clean up failed invocations and enforce data integrity

**Status:** Pending  
**Created:** 2025-08-05 09:38:39  
**Updated:** 2025-08-05 09:38:39  
**Tags:** data-integrity, database, cleanup  

## Description

Investigate and clean up 24 failed SDXLTurbo invocations with NULL outputs, and implement database constraints to prevent incomplete data.

## Context

During the process of ensuring 100% NomicVision embedding coverage for IMAGE invocations, we discovered 24 IMAGE invocations that have NULL outputs. These represent failed model executions that cannot be embedded.

### Failed Invocations Analysis

**Summary:**
- Total failed invocations: 24
- All failures are from model: SDXLTurbo
- All have network configuration: ['SDXLTurbo', 'Moondream']
- All have seed: -1
- All have NULL outputs

**Sequence Number Distribution:**
```
[80, 100, 210, 530, 532, 534, 536, 540, 542, 546, 548, 550, 552, 562, 564, 574, 582, 584, 584, 584, 584, 584, 584, 584]
```

Key observations:
- 8 failures (33%) at sequence number 584
- Most failures occur at high sequence numbers (500+)
- Suggests failures happen late in long runs, possibly due to:
  - Memory issues after many iterations
  - Timeout limits being reached
  - Systematic issues manifesting after many invocations

## Goals

1. **Zero incomplete/failed data in the database**
2. **Database constraints to enforce data integrity**
3. **Clean handling of retry scenarios**

## Proposed Investigation & Solutions

### 1. Check for Retry Patterns

Investigate whether these failed invocations were retried:
- Check if there's another Invocation in the same run with the same `input_invocation_id`
- Verify no "next" invocation uses the failed one as its `input_invocation_id`
- If confirmed as retried and orphaned, these can be safely deleted

### 2. Database Constraints to Consider

**Option A: NOT NULL constraint on output**
- Pros: Prevents NULL outputs from being saved
- Cons: May break retry logic if failures need to be recorded

**Option B: Check constraint**
- Add constraint ensuring if `type = IMAGE` then `output IS NOT NULL`
- More flexible, could allow temporary NULLs during processing

**Option C: Trigger-based validation**
- Before insert/update trigger to validate data completeness
- Could auto-delete or flag incomplete records

### 3. Application-level Solutions

- Modify the engine to:
  - Never save invocations with NULL outputs
  - Always retry failed invocations before moving on
  - Mark failed invocations with a specific status instead of NULL output

### 4. Cleanup Script

Create a script to:
1. Identify all invocations with NULL outputs
2. Check if they were successfully retried
3. Delete orphaned failed invocations
4. Report any that can't be safely deleted

## Implementation Steps

1. [ ] Write analysis query to identify retry patterns for the 24 failed invocations
2. [ ] Document findings about whether these were retried
3. [ ] Design appropriate database constraints
4. [ ] Implement cleanup script
5. [ ] Test constraints don't break existing functionality
6. [ ] Deploy constraints to prevent future occurrences

## Example Analysis Query

```sql
-- Find failed invocations that were likely retried
WITH failed_invocations AS (
    SELECT id, run_id, input_invocation_id, sequence_number
    FROM invocation 
    WHERE type = 'IMAGE' AND output IS NULL
),
potential_retries AS (
    SELECT f.id as failed_id, 
           r.id as retry_id,
           f.sequence_number as failed_seq,
           r.sequence_number as retry_seq
    FROM failed_invocations f
    JOIN invocation r ON f.run_id = r.run_id 
                     AND f.input_invocation_id = r.input_invocation_id
                     AND r.id != f.id
                     AND r.output IS NOT NULL
)
SELECT * FROM potential_retries;
```

## Success Criteria

- All 24 failed invocations are either cleaned up or documented as necessary
- Database constraints prevent future NULL output invocations for IMAGE type
- No valid data is lost in the cleanup process
- Retry logic continues to work properly