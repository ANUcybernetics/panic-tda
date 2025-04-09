import itertools
import os
import random

import polars as pl
import pytest

from trajectory_tracer.datavis import (
    plot_persistence_diagram,
    plot_persistence_diagram_by_run,
    plot_persistence_diagram_faceted,
    plot_persistence_entropy,
    plot_semantic_drift,
)


@pytest.fixture
def mock_runs_df():
    # Define real-world model names
    text_models = ["DummyI2T", "DummyI2T2"]
    image_models = ["DummyT2I", "DummyT2I2"]
    embedding_models = ["Dummy", "Dummy2"]
    prompts = ["one fish", "two fish", "red fish", "blue fish"]
    seeds = [-1] * 16

    # Generate combinations without embedding_model
    base_combinations = list(
        itertools.product(text_models, image_models, prompts, seeds)
    )

    # Create run IDs for unique combinations excluding embedding_model
    run_id_map = {combo: run_id for run_id, combo in enumerate(base_combinations, 1)}

    # Prepare data
    data = []
    for text_model, image_model, embedding_model, prompt, seed in itertools.product(
        text_models, image_models, embedding_models, prompts, seeds
    ):
        # Use the same run_id for different embedding models with same other factors
        base_combo = (text_model, image_model, prompt, seed)
        run_id = run_id_map[base_combo]

        # Create multiple homology dimensions per run
        for dim in [0, 1, 2]:
            # Generate multiple persistence pairs for each dimension
            num_points = random.randint(3, 10)
            for _ in range(num_points):
                birth = random.uniform(0, 0.5)
                death = random.uniform(birth + 0.1, 1.0)

                data.append({
                    "run_id": run_id,
                    "text_model": text_model,
                    "image_model": image_model,
                    "embedding_model": embedding_model,
                    "initial_prompt": prompt,
                    "seed": seed,
                    "homology_dimension": dim,
                    "birth": birth,
                    "death": death,
                    "persistence": death - birth,
                    "entropy": random.uniform(0.1, 2.0),
                    "experiment_id": 1,  # Dummy experiment ID
                })

    return pl.DataFrame(data)


@pytest.fixture
def mock_embeddings_df():
    # Define real-world model names
    text_models = ["DummyI2T", "DummyI2T2"]
    image_models = ["DummyT2I", "DummyT2I2"]
    embedding_models = ["Dummy", "Dummy2"]
    prompts = ["one fish", "two fish", "red fish", "blue fish"]
    seeds = [-1] * 2

    # Generate combinations without embedding_model
    base_combinations = list(
        itertools.product(text_models, image_models, prompts, seeds)
    )

    # Create run IDs for unique combinations excluding embedding_model
    run_id_map = {combo: run_id for run_id, combo in enumerate(base_combinations, 1)}

    # Prepare data
    data = []
    for text_model, image_model, embedding_model, prompt, seed in itertools.product(
        text_models, image_models, embedding_models, prompts, seeds
    ):
        # Use the same run_id for different embedding models with same other factors
        base_combo = (text_model, image_model, prompt, seed)
        run_id = run_id_map[base_combo]

        # Create sequence points for each run
        for seq_num in range(100):
            data.append({
                "run_id": run_id,
                "text_model": text_model,
                "image_model": image_model,
                "embedding_model": embedding_model,
                "initial_prompt": prompt,
                "seed": seed,
                "sequence_number": seq_num,
                "semantic_drift": random.uniform(0, seq_num / 20),
                "experiment_id": 1,  # Dummy experiment ID
            })

    return pl.DataFrame(data)


def test_plot_persistence_diagram(mock_runs_df):
    # Verify we have persistence diagram data
    assert mock_runs_df.height > 0
    assert "homology_dimension" in mock_runs_df.columns
    assert "birth" in mock_runs_df.columns
    assert "death" in mock_runs_df.columns

    # Define output file
    output_file = "output/vis/test/persistence_diagram.png"

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Generate the plot
    plot_persistence_diagram(mock_runs_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_diagram_faceted(mock_runs_df):
    # Verify we have persistence diagram data
    assert mock_runs_df.height > 0
    assert "homology_dimension" in mock_runs_df.columns
    assert "birth" in mock_runs_df.columns
    assert "death" in mock_runs_df.columns

    # Define output file
    output_file = "output/vis/test/persistence_diagram_faceted.png"

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Generate the plot
    plot_persistence_diagram_faceted(mock_runs_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_diagram_by_run(mock_runs_df):
    # Verify we have persistence diagram data
    assert mock_runs_df.height > 0
    assert "homology_dimension" in mock_runs_df.columns
    assert "birth" in mock_runs_df.columns
    assert "death" in mock_runs_df.columns
    assert "run_id" in mock_runs_df.columns

    # Define output file
    output_file = "output/vis/test/persistence_diagram_by_run.png"

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Get a sample run_id from the dataframe
    sample_run_id = mock_runs_df["run_id"][0]

    # Generate the plot
    plot_persistence_diagram_by_run(mock_runs_df, sample_run_id, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_semantic_drift(mock_embeddings_df):
    # Verify we have semantic dispersion data
    assert mock_embeddings_df.height > 0
    assert "semantic_drift" in mock_embeddings_df.columns
    assert "sequence_number" in mock_embeddings_df.columns

    # Define output file
    output_file = "output/vis/test/semantic_drift.png"

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Generate the plot
    plot_semantic_drift(mock_embeddings_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"


def test_plot_persistence_entropy(mock_runs_df):
    # Verify we have persistence diagram data with entropy values
    assert mock_runs_df.height > 0
    assert "homology_dimension" in mock_runs_df.columns
    assert "entropy" in mock_runs_df.columns
    assert "run_id" in mock_runs_df.columns

    # Define output file
    output_file = "output/vis/test/persistence_entropy.png"

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Generate the plot
    plot_persistence_entropy(mock_runs_df, output_file)

    # Verify file was created
    assert os.path.exists(output_file), f"File was not created: {output_file}"
