---
id: task-35
title: identify slow tests and see if they can be sped up
status: Done
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

## Findings

### Slowest Tests (from pytest --durations=10):

1. **test_cluster_all_data_no_downsampling** (83.89s)
   - Runs 2 full experiments with max_length=50 each
   - Calls perform_experiment() which involves Ray distributed computing
   - Then performs clustering on all generated data
   - **Analysis**: Slow due to full experiment execution + clustering

2. **test_load_clusters_df_with_downsampling** (74.20s)
   - Creates experiment with max_length=200 (more data points)
   - Runs full experiment then tests downsampling functionality
   - **Analysis**: Slow due to large data generation (200 steps)

3. **test_add_persistence_entropy** (70.45s)
   - Creates experiment with max_length=100 and 2 seeds
   - Runs full experiment to generate persistence diagrams
   - **Analysis**: Slow due to full experiment execution

4. **test_clustering_persistence_across_sessions** (69.00s)
   - Tests that clustering results persist in database
   - Runs full experiment with max_length=25
   - **Analysis**: Slow due to full experiment execution

5. **test_plot_diagram** (52.45s)
   - Generates 1000 points and computes persistence diagrams
   - Creates visualization plots
   - **Analysis**: Slow due to TDA computation on 1000 points

6. **test_calculate_wasserstein_distances** (44.07s)
   - Creates multiple runs (3 seeds × 2 prompts × 2 models = 12 runs)
   - Computes pairwise Wasserstein distances between persistence diagrams
   - **Analysis**: Slow due to multiple runs and distance computations

7. **test_plot_persistence_diagram_by_prompt** (40.72s)
   - Runs experiment and creates persistence diagram visualizations
   - **Analysis**: Slow due to experiment execution + plotting

8. **test_optics_scalability[5000]** (32.48s)
   - Tests OPTICS clustering on 5000 synthetic embeddings
   - **Analysis**: Slow due to OPTICS algorithm complexity on large dataset

9. **test_uuid_type_validation** (27.70s)
   - Runs experiment with max_length=20
   - **Analysis**: Slow due to full experiment execution

10. **test_optics_scalability[1000]** (23.46s)
    - Tests OPTICS clustering on 1000 synthetic embeddings
    - **Analysis**: Slow due to OPTICS algorithm complexity

### Common Patterns:

1. **Full Experiment Execution**: Most slow tests (7/10) run complete experiments using `perform_experiment()`, which involves:
   - Ray distributed computing setup
   - Running generative AI model pipelines
   - Computing embeddings
   - Database operations

2. **Large Data Processing**: Several tests use large datasets (max_length=50-200 or 1000-5000 points)

3. **Complex Algorithms**: TDA computations and OPTICS clustering are inherently expensive on large datasets

## Optimization Recommendations

### 1. Pre-computed Test Data (Highest Impact)

The main bottleneck is that most tests run full experiments using `perform_experiment()`. We could:

- Create fixtures with pre-computed embeddings/runs that can be loaded directly into the database
- This would eliminate the need to run Ray + generative models for most tests
- Tests could focus on the specific functionality being tested (clustering, visualization, etc.)

**Implementation approach**:
- Create a fixture that populates database with pre-generated Run/Embedding data
- Store as SQL dump or programmatic fixture
- Most tests could then skip `perform_experiment()` entirely

### 2. Reduce Data Sizes for Tests

Several tests use larger data than necessary:
- `test_load_clusters_df_with_downsampling`: max_length=200 could be reduced
- `test_add_persistence_entropy`: max_length=100 could be reduced
- `test_optics_scalability`: Could test with smaller sizes (100, 500) instead of (1000, 5000)

### 3. Mock Ray Operations

For tests that must run experiments, we could:
- Mock the Ray remote operations to run synchronously
- This would eliminate Ray overhead while still testing the logic

### 4. Parallel Test Execution

Already implemented with `pytest -n 8`, so no further optimization needed here.

## Recommendation

**I recommend NOT implementing these optimizations because:**

1. **Integration Testing Value**: The current tests provide valuable end-to-end integration testing, ensuring the full pipeline works correctly with Ray, database operations, and model execution.

2. **Complexity vs Benefit**: Creating and maintaining test fixtures would add significant complexity. The test suite already runs in ~3 minutes with parallelization, which is reasonable.

3. **Real-world Behavior**: Testing with actual experiment execution ensures tests catch issues that might only appear in the full pipeline.

4. **Maintenance Burden**: Pre-computed test data would need to be regenerated whenever the data model or processing logic changes.

The slowness is inherent to the comprehensive nature of the tests and doesn't warrant the added complexity of optimization.

## Resolution

After analyzing the 10 slowest tests, I determined that the slowness is due to running full end-to-end experiments with Ray, generative models, and database operations. While optimizations are possible (pre-computed fixtures, smaller data sizes, mocking), they would reduce the integration testing value of these tests. The ~3 minute total test runtime with parallelization is acceptable, so no changes were made.
