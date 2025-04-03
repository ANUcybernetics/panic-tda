from trajectory_tracer.analysis import load_runs_df
from trajectory_tracer.art import analyze_with_r
from trajectory_tracer.engine import perform_experiment
from trajectory_tracer.schemas import ExperimentConfig
import polars as pl

def test_analyze_with_r(db_session):
    """Test that analyze_with_r correctly processes TDA data using the R function."""

    # Create a simple test configuration with multiple models and prompts
    config = ExperimentConfig(
        networks=[["DummyT2I", "DummyI2T"], ["DummyT2I2", "DummyI2T2"]],
        seeds=[-1, -1],
        prompts=["test prompt 1", "test prompt 2"],
        embedding_models=["Dummy"],
        max_length=10,  # Short runs for testing
    )

    # Save config to database to get an ID
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)

    # Run the experiment to populate database
    db_url = str(db_session.get_bind().engine.url)
    perform_experiment(str(config.id), db_url)

    # Load runs dataframe
    runs_df = load_runs_df(db_session, use_cache=False)

    # Filter for just this experiment
    exp_df = runs_df.filter(pl.col("experiment_id") == str(config.id))

    # Make sure we have data to analyze
    assert exp_df.height > 0

    # Filter for homology dimension 0
    dim0_df = exp_df.filter(pl.col("homology_dimension") == 0)
    assert dim0_df.height > 0

    # Rename columns to match what the R function expects
    analysis_df = dim0_df.rename({
        "text_model": "caption_model",
        "initial_prompt": "prompt"
    })

    # Make sure we have all the columns needed for R analysis
    required_columns = ["image_model", "caption_model", "prompt", "homology_dimension", "entropy"]
    for col in required_columns:
        assert col in analysis_df.columns

    # Call the function under test
    results = analyze_with_r(analysis_df)

    # Verify the structure of the results
    assert isinstance(results, dict)
    assert "models" in results
    assert "anova" in results
    assert "contrasts" in results

    # Check that models contains the expected models
    assert "full" in results["models"]
    assert "control" in results["models"]
    assert "selected" in results["models"]

    # Check that contrasts contains the expected contrast types
    contrast_types = ["image_model", "caption_model", "prompt", "interaction", "diff_of_diff"]
    for contrast_type in contrast_types:
        assert contrast_type in results["contrasts"]
