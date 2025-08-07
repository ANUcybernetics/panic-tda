# Task: Clean up incomplete runs from experiment 06826b10

**Status:** Todo  
**Created:** 2025-08-07  
**Priority:** High  
**Tags:** cleanup, data-integrity, doctor-command  

## Problem Statement

The doctor command is reporting 300 runs with incomplete invocation sequences in experiment `06826b10-ddc2-79cb-9cc5-0879a974a6be`. These runs are stopping around 584-586 invocations instead of the expected 1000, all with `stop_reason: unknown`.

### Key Findings:
- **300 incomplete runs** all from the same experiment
- Most common stopping point: **584 invocations** (37 runs)
- All have `stop_reason: unknown` indicating abnormal termination
- Pattern suggests systematic failure, not random errors
- Primarily affects `['SDXLTurbo', 'Moondream']` network configuration

## Objective

Clean up these incomplete runs and all associated data so the doctor command passes for this experiment, accepting the small data loss.

## Implementation Plan

### Step 1: Identify Incomplete Runs
```python
# Find all runs in experiment with < 1000 invocations
experiment_id = UUID("06826b10-ddc2-79cb-9cc5-0879a974a6be")
incomplete_runs = session.exec(
    select(Run)
    .where(Run.experiment_id == experiment_id)
    .where(Run.stop_reason == "unknown")
).all()

# Alternative: check by invocation count
for run in experiment_runs:
    count = session.exec(
        select(func.count(Invocation.id))
        .where(Invocation.run_id == run.id)
    ).first()
    if count < 1000:
        incomplete_runs.append(run)
```

### Step 2: Count Associated Data
Before deletion, count:
- Number of incomplete runs
- Total invocations to be deleted
- Total embeddings to be deleted  
- Total persistence diagrams to be deleted

### Step 3: Delete with Cascading
The Run model has cascade delete configured:
```python
invocations: List[Invocation] = Relationship(
    back_populates="run",
    sa_relationship_kwargs={"cascade": "all, delete-orphan"}
)
```

This should automatically delete:
- All invocations for the run
- All embeddings (via invocation cascade)
- All persistence diagrams for the run

### Step 4: Deletion Implementation
```python
def delete_incomplete_runs(session, experiment_id):
    """Delete all incomplete runs from the specified experiment."""
    
    # Find incomplete runs
    incomplete_runs = []
    all_runs = session.exec(
        select(Run).where(Run.experiment_id == experiment_id)
    ).all()
    
    for run in all_runs:
        inv_count = session.exec(
            select(func.count(Invocation.id))
            .where(Invocation.run_id == run.id)
        ).first()
        
        if inv_count < 1000:
            incomplete_runs.append(run)
    
    print(f"Found {len(incomplete_runs)} incomplete runs to delete")
    
    # Delete each run (cascades to invocations, embeddings, PDs)
    for run in incomplete_runs:
        session.delete(run)
    
    session.commit()
    return len(incomplete_runs)
```

### Step 5: Verification
After deletion:
1. Run doctor command on the experiment
2. Verify no integrity issues remain
3. Check that complete runs (1000 invocations) are intact

## Expected Outcome

- 300 incomplete runs deleted
- ~175,000 invocations deleted (300 runs × ~584 avg invocations)
- Associated embeddings and persistence diagrams deleted
- Doctor command passes with 0 issues for this experiment

## Notes

- This is a one-time cleanup for a systematic error
- Root cause appears to be a timeout/resource issue at ~584 iterations
- Future runs should be monitored to prevent recurrence
- Consider implementing a "resume incomplete runs" feature

## Implementation

Implemented cleanup script in `src/panic_tda/main.py:script()` function. The script:

1. Finds all runs in experiment `06826b10-ddc2-79cb-9cc5-0879a974a6be`
2. Identifies runs with < 1000 invocations as incomplete
3. Shows distribution of invocation counts for incomplete runs
4. Counts total data to be deleted (runs, invocations, embeddings, persistence diagrams)
5. Requires user confirmation (type "DELETE") before proceeding
6. Deletes incomplete runs using SQLModel's cascade delete
7. Verifies deletion by counting remaining runs

To run: `uv run panic-tda script`

The cascade delete configured on the Run model ensures all related data is properly cleaned up:
- Invocations (cascade via Run.invocations relationship)
- Embeddings (cascade via Invocation.embeddings relationship)  
- Persistence Diagrams (cascade via Run.persistence_diagrams relationship)