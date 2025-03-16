import json
import subprocess
import sys
import os
from pathlib import Path

def test_run_experiment_command(tmp_path):
    """Test the run-experiment command with a simple configuration."""

    # Create a simple test configuration
    config = {
        "networks": [["DummyT2I", "DummyI2T"]],
        "seeds": [42],
        "prompts": ["A test prompt for CLI testing"],
        "embedding_models": ["Dummy"],
        "run_length": 3
    }

    # Save the config to a temp file
    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)

    # Create a database path
    db_path = tmp_path / "test_output.sqlite"

    # Use subprocess to run the command directly
    cmd = [
        sys.executable, "-m", "trajectory_tracer.main", "run-experiment",
        str(config_file), "--db-path", str(db_path), "--verbose"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check the result
    assert result.returncode == 0
    assert "Experiment completed successfully" in result.stdout
    assert db_path.exists()


def test_list_runs_command(tmp_path):
    """Test the list-runs command after creating a run."""
    import json
    import subprocess
    import sys

    # First create a run with a simple configuration
    config = {
        "networks": [["DummyT2I", "DummyI2T"]],
        "seeds": [42],
        "prompts": ["A test prompt for list-runs"],
        "embedding_models": ["Dummy"],
        "run_length": 2
    }

    # Save the config to a temp file
    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)

    # Create a database path
    db_path = tmp_path / "test_output.sqlite"

    # Run the experiment command first
    cmd = [
        sys.executable, "-m", "trajectory_tracer.main", "run-experiment",
        str(config_file), "--db-path", str(db_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0

    # Now test the list-runs command
    cmd = [
        sys.executable, "-m", "trajectory_tracer.main", "list-runs",
        "--db-path", str(db_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check the result
    assert result.returncode == 0
    assert "Found 1 runs" in result.stdout

    # Test verbose mode
    cmd = [
        sys.executable, "-m", "trajectory_tracer.main", "list-runs",
        "--db-path", str(db_path), "--verbose"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Run ID:" in result.stdout
    assert "Network:" in result.stdout


def test_export_images_command(tmp_path):
    """Test the export-images command after creating a run."""
    import json
    import subprocess
    import sys
    import re

    # First create a run with a simple configuration
    config = {
        "networks": [["DummyT2I", "DummyI2T"]],
        "seeds": [42],
        "prompts": ["A test prompt for export-images"],
        "embedding_models": ["Dummy"],
        "run_length": 2
    }

    # Save the config to a temp file
    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)

    # Create database and output paths
    db_path = tmp_path / "test_output.sqlite"
    output_dir = tmp_path / "test_images"

    # Run the experiment command first
    cmd = [
        sys.executable, "-m", "trajectory_tracer.main", "run-experiment",
        str(config_file), "--db-path", str(db_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0

    # Get the run ID from the list command
    cmd = [
        sys.executable, "-m", "trajectory_tracer.main", "list-runs",
        "--db-path", str(db_path), "--verbose"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract run ID from output
    run_id_match = re.search(r"Run ID: ([0-9a-f-]+)", result.stdout)
    assert run_id_match
    run_id = run_id_match.group(1)

    # Now test the export-images command for specific run
    cmd = [
        sys.executable, "-m", "trajectory_tracer.main", "export-images",
        run_id, "--db-path", str(db_path), "--output-dir", str(output_dir)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check the result
    assert result.returncode == 0
    assert "Image export completed successfully" in result.stdout

    # Verify image directory was created with run ID
    assert (output_dir / run_id).exists()
    # Should have at least one image file for the DummyT2I invocation
    assert len(list((output_dir / run_id).glob("*.jpg"))) >= 1

    # Test export-images for all runs
    all_output_dir = tmp_path / "all_test_images"
    cmd = [
        sys.executable, "-m", "trajectory_tracer.main", "export-images",
        "all", "--db-path", str(db_path), "--output-dir", str(all_output_dir)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check the result
    assert result.returncode == 0
    assert "Image export completed successfully" in result.stdout
    # Should have created a directory with the same run ID
    assert (all_output_dir / run_id).exists()
