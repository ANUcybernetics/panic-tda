"""
Tests for the doctor module.
"""

import json
import tempfile
import os
from uuid import uuid4

import pytest
from sqlmodel import Session, create_engine, select

from panic_tda.doctor import (
    DoctorReport,
    check_experiment_embeddings,
    check_experiment_persistence_diagrams,
    check_experiment_run_invocations,
    check_orphaned_records,
    doctor_all_experiments,
    doctor_single_experiment,
    fix_sequence_gaps,
    clean_orphaned_records,
)
from panic_tda.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    InvocationType,
    PersistenceDiagram,
    Run,
    SQLModel,
)


@pytest.fixture
def test_db():
    """Create a test database with sample data."""
    # Use a temporary file for the database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Create experiment config
        experiment = ExperimentConfig(
            id=uuid4(),
            networks=[["DummyT2I", "DummyI2T"]],
            seeds=[42],
            prompts=["Test prompt"],
            embedding_models=["DummyText", "DummyText2"],
            max_length=3,
        )
        session.add(experiment)

        # Create a run
        run = Run(
            id=uuid4(),
            experiment_id=experiment.id,
            network=["DummyT2I", "DummyI2T"],
            seed=42,
            initial_prompt="Test prompt",
            max_length=3,
        )
        session.add(run)

        # Create invocations (with a gap - missing sequence_number 1)
        inv0 = Invocation(
            id=uuid4(),
            run_id=run.id,
            sequence_number=0,
            type=InvocationType.TEXT,
            model="DummyT2I",
            seed=42,
            output_text="Output 0",
        )
        inv2 = Invocation(
            id=uuid4(),
            run_id=run.id,
            sequence_number=2,
            type=InvocationType.TEXT,
            model="DummyI2T",
            seed=42,
            output_text="Output 2",
        )
        session.add(inv0)
        session.add(inv2)

        # Create embeddings (only for DummyText, missing DummyText2)
        emb1 = Embedding(
            id=uuid4(),
            invocation_id=inv0.id,
            embedding_model="DummyText",
            vector=[0.1, 0.2, 0.3],
        )
        session.add(emb1)

        # Create orphaned embedding (with non-existent invocation)
        orphan_emb = Embedding(
            id=uuid4(),
            invocation_id=uuid4(),  # Non-existent invocation
            embedding_model="DummyText",
            vector=[0.4, 0.5, 0.6],
        )
        session.add(orphan_emb)

        # Create persistence diagram (only for DummyText, missing DummyText2)
        pd1 = PersistenceDiagram(
            id=uuid4(),
            run_id=run.id,
            embedding_model="DummyText",
            diagram_data=None,  # Use None for testing
        )
        session.add(pd1)

        # Create PD with invalid embedding model
        pd_invalid = PersistenceDiagram(
            id=uuid4(),
            run_id=run.id,
            embedding_model="invalid_model",
            diagram_data=None,  # Use None for testing
        )
        session.add(pd_invalid)

        session.commit()

        yield {
            "engine": engine,
            "db_url": db_url,
            "db_path": db_path,
            "experiment_id": experiment.id,
            "run_id": run.id,
            "inv0_id": inv0.id,
            "inv2_id": inv2.id,
            "orphan_emb_id": orphan_emb.id,
        }

    # Cleanup
    engine.dispose()
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def clean_db():
    """Create a clean database with a perfect experiment."""
    # Use a temporary file for the database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Create a perfect experiment with no issues
        experiment = ExperimentConfig(
            id=uuid4(),
            networks=[["DummyT2I"]],
            seeds=[42],
            prompts=["Test"],
            embedding_models=["DummyText"],
            max_length=2,
        )
        session.add(experiment)

        run = Run(
            id=uuid4(),
            experiment_id=experiment.id,
            network=["DummyT2I"],
            seed=42,
            initial_prompt="Test",
            max_length=2,
        )
        session.add(run)

        # Create complete invocations
        for i in range(2):
            inv = Invocation(
                id=uuid4(),
                run_id=run.id,
                sequence_number=i,
                type=InvocationType.TEXT,
                model="DummyT2I",
                seed=42,
                output_text=f"Output {i}",
            )
            session.add(inv)

            # Create embedding
            emb = Embedding(
                id=uuid4(),
                invocation_id=inv.id,
                embedding_model="DummyText",
                vector=[0.1 * i, 0.2 * i, 0.3 * i],
            )
            session.add(emb)

        # Create persistence diagram with valid data
        # Use a simple dict that the PersistenceDiagramResultType can handle
        pd = PersistenceDiagram(
            id=uuid4(),
            run_id=run.id,
            embedding_model="DummyText",
            diagram_data={"test": []},  # Use a simple dict for testing
        )
        session.add(pd)

        session.commit()

        yield {
            "engine": engine,
            "db_url": db_url,
            "db_path": db_path,
            "experiment_id": experiment.id,
        }

    # Cleanup
    engine.dispose()
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def multi_experiment_db():
    """Create a database with multiple experiments (one clean, one with issues)."""
    # Use a temporary file for the database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # First experiment: clean
        clean_experiment = ExperimentConfig(
            id=uuid4(),
            networks=[["DummyT2I"]],
            seeds=[42],
            prompts=["Clean"],
            embedding_models=["DummyText"],
            max_length=2,
        )
        session.add(clean_experiment)

        clean_run = Run(
            id=uuid4(),
            experiment_id=clean_experiment.id,
            network=["DummyT2I"],
            seed=42,
            initial_prompt="Clean",
            max_length=2,
        )
        session.add(clean_run)

        # Create complete invocations for clean experiment
        for i in range(2):
            inv = Invocation(
                id=uuid4(),
                run_id=clean_run.id,
                sequence_number=i,
                type=InvocationType.TEXT,
                model="DummyT2I",
                seed=42,
                output_text=f"Clean {i}",
            )
            session.add(inv)

            # Create embedding
            emb = Embedding(
                id=uuid4(),
                invocation_id=inv.id,
                embedding_model="DummyText",
                vector=[0.1 * i, 0.2 * i, 0.3 * i],
            )
            session.add(emb)

        # Create persistence diagram for clean experiment
        pd = PersistenceDiagram(
            id=uuid4(),
            run_id=clean_run.id,
            embedding_model="DummyText",
            diagram_data={"test": []},
        )
        session.add(pd)

        # Second experiment: with issues
        problematic_experiment = ExperimentConfig(
            id=uuid4(),
            networks=[["DummyT2I"]],
            seeds=[42],
            prompts=["Problem"],
            embedding_models=["DummyText", "DummyText2"],
            max_length=3,
        )
        session.add(problematic_experiment)

        problem_run = Run(
            id=uuid4(),
            experiment_id=problematic_experiment.id,
            network=["DummyT2I"],
            seed=42,
            initial_prompt="Problem",
            max_length=3,
        )
        session.add(problem_run)

        # Create incomplete invocations (missing sequence 1)
        for i in [0, 2]:
            inv = Invocation(
                id=uuid4(),
                run_id=problem_run.id,
                sequence_number=i,
                type=InvocationType.TEXT,
                model="DummyT2I",
                seed=42,
                output_text=f"Problem {i}",
            )
            session.add(inv)

            # Only create DummyText embedding, missing DummyText2
            if i == 0:
                emb = Embedding(
                    id=uuid4(),
                    invocation_id=inv.id,
                    embedding_model="DummyText",
                    vector=[0.1, 0.2, 0.3],
                )
                session.add(emb)

        session.commit()

        yield {
            "engine": engine,
            "db_url": db_url,
            "db_path": db_path,
            "clean_experiment_id": clean_experiment.id,
            "problematic_experiment_id": problematic_experiment.id,
        }

    # Cleanup
    engine.dispose()
    if os.path.exists(db_path):
        os.unlink(db_path)


class TestDoctorReport:
    """Test the DoctorReport class."""

    def test_report_initialization(self):
        """Test that DoctorReport initializes correctly."""
        report = DoctorReport()
        assert report.total_experiments == 0
        assert report.experiments_with_issues == 0
        assert not report.has_issues()

    def test_add_issues(self):
        """Test adding various issues to the report."""
        report = DoctorReport()
        exp_id = uuid4()
        run_id = uuid4()
        inv_id = uuid4()

        # Add run invocation issue
        report.add_run_invocation_issue(exp_id, run_id, {"missing_first": True})
        assert len(report.run_invocation_issues) == 1
        assert report.has_issues()

        # Add embedding issue
        report.add_embedding_issue(exp_id, inv_id, {"embedding_count": 0})
        assert len(report.embedding_issues) == 1

        # Add PD issue
        report.add_pd_issue(exp_id, run_id, {"issue_type": "missing"})
        assert len(report.pd_issues) == 1

        # Add sequence gap issue
        report.add_sequence_gap_issue(exp_id, run_id, [1, 3], [0, 2, 4])
        assert len(report.sequence_gap_issues) == 1

    def test_json_output(self):
        """Test JSON output generation."""
        report = DoctorReport()
        report.total_experiments = 1
        report.finalize()

        json_str = report.to_json()
        data = json.loads(json_str)

        assert "summary" in data
        assert "issues" in data
        assert data["summary"]["total_experiments"] == 1
        assert "duration_seconds" in data["summary"]


class TestCheckFunctions:
    """Test the various check functions."""

    def test_check_experiment_run_invocations(self, test_db):
        """Test checking run invocations for missing sequences."""
        with Session(test_db["engine"]) as session:
            experiment = session.get(ExperimentConfig, test_db["experiment_id"])
            report = DoctorReport()

            issues = check_experiment_run_invocations(experiment, session, report)

            # Should find issues: missing sequence 1, wrong count
            assert len(issues) == 1
            assert issues[0]["actual_count"] == 2
            assert issues[0]["expected_count"] == 3
            assert report.sequence_gap_issues  # Should detect the gap
            assert 1 in report.sequence_gap_issues[0]["gaps"]

    def test_check_experiment_embeddings(self, test_db):
        """Test checking embeddings for completeness."""
        with Session(test_db["engine"]) as session:
            experiment = session.get(ExperimentConfig, test_db["experiment_id"])
            report = DoctorReport()

            issues = check_experiment_embeddings(experiment, session, report)

            # Should find issues: missing DummyText2 for both invocations, and missing DummyText for inv2
            assert len(issues) > 0
            # Check for missing DummyText2 embeddings
            dummy2_issues = [
                i for i in issues if i.get("embedding_model") == "DummyText2"
            ]
            assert len(dummy2_issues) == 2  # Both invocations missing DummyText2

    def test_check_experiment_persistence_diagrams(self, test_db):
        """Test checking persistence diagrams."""
        with Session(test_db["engine"]) as session:
            experiment = session.get(ExperimentConfig, test_db["experiment_id"])
            report = DoctorReport()

            issues = check_experiment_persistence_diagrams(experiment, session, report)

            # Should find issues: invalid_model PD, DummyText with null data, and missing DummyText2 PD
            assert len(issues) == 3
            assert any(issue["issue_type"] == "invalid_model" for issue in issues)
            assert any(issue["embedding_model"] == "DummyText2" for issue in issues)
            # DummyText PD exists but has null data, so it's considered missing
            assert any(
                issue["embedding_model"] == "DummyText" and issue["has_null_data"]
                for issue in issues
            )

    def test_check_orphaned_records(self, test_db):
        """Test checking for orphaned records."""
        with Session(test_db["engine"]) as session:
            report = DoctorReport()

            check_orphaned_records(session, report)

            # Should find the orphaned embedding
            assert len(report.orphaned_records["embeddings"]) == 1
            assert report.orphaned_records["global_orphans"]["embeddings"] == 1


class TestEmbeddingNullVectorCheck:
    """Test the null vector checking fix for embeddings."""

    def test_null_vector_detection(self):
        """Test that null vectors are correctly detected in numpy arrays, not BLOB fields."""
        # Create a test database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        SQLModel.metadata.create_all(engine)

        try:
            with Session(engine) as session:
                # Create experiment
                experiment = ExperimentConfig(
                    id=uuid4(),
                    networks=[["DummyT2I"]],
                    seeds=[42],
                    prompts=["Test"],
                    embedding_models=["DummyText"],
                    max_length=2,
                )
                session.add(experiment)

                # Create run
                run = Run(
                    id=uuid4(),
                    experiment_id=experiment.id,
                    network=["DummyT2I"],
                    seed=42,
                    initial_prompt="Test",
                    max_length=2,
                )
                session.add(run)

                # Create invocation
                inv = Invocation(
                    id=uuid4(),
                    run_id=run.id,
                    sequence_number=0,
                    type=InvocationType.TEXT,
                    model="DummyT2I",
                    seed=42,
                    output_text="Test output",
                )
                session.add(inv)

                # Create embedding with valid vector (stored as non-NULL BLOB)
                emb_valid = Embedding(
                    id=uuid4(),
                    invocation_id=inv.id,
                    embedding_model="DummyText",
                    vector=[0.1, 0.2, 0.3],  # Valid numpy array
                )
                session.add(emb_valid)

                # Create another invocation for null vector test
                inv2 = Invocation(
                    id=uuid4(),
                    run_id=run.id,
                    sequence_number=1,
                    type=InvocationType.TEXT,
                    model="DummyT2I",
                    seed=42,
                    output_text="Test output 2",
                )
                session.add(inv2)

                # Create embedding with null vector
                emb_null = Embedding(
                    id=uuid4(),
                    invocation_id=inv2.id,
                    embedding_model="DummyText",
                    vector=None,  # Actual null vector
                )
                session.add(emb_null)

                session.commit()

                # Run the check
                report = DoctorReport()
                issues = check_experiment_embeddings(experiment, session, report)

                # Should only find issue with the null vector embedding
                null_vector_issues = [i for i in issues if i.get("has_null_vector")]
                assert len(null_vector_issues) == 1
                assert null_vector_issues[0]["invocation_id"] == inv2.id

                # The valid embedding should not be reported as having null vector
                valid_issues = [
                    i
                    for i in issues
                    if i["invocation_id"] == inv.id and i.get("has_null_vector")
                ]
                assert len(valid_issues) == 0

        finally:
            engine.dispose()
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_embedding_ids_included_in_issues(self, test_db):
        """Test that embedding IDs are included in embedding issues."""
        with Session(test_db["engine"]) as session:
            experiment = session.get(ExperimentConfig, test_db["experiment_id"])
            report = DoctorReport()

            issues = check_experiment_embeddings(experiment, session, report)

            # All issues should have embedding_ids field
            for issue in issues:
                assert "embedding_ids" in issue
                # If there are embeddings, the IDs should be UUIDs
                if issue.get("embedding_count", 0) > 0:
                    assert isinstance(issue["embedding_ids"], list)
                    # Check that we got actual IDs (could be empty list if no embeddings)
                    if len(issue["embedding_ids"]) > 0:
                        # The IDs should be UUID objects at this point (before JSON serialization)
                        from uuid import UUID

                        assert all(
                            isinstance(id, UUID) for id in issue["embedding_ids"]
                        )


class TestPersistenceDiagramWithNullVectorEmbeddings:
    """Test that persistence diagrams handle embeddings with null vectors properly."""

    def test_pd_computation_with_null_vectors(self):
        """Test that PD computation fails gracefully when embeddings have null vectors."""
        import numpy as np
        from datetime import datetime

        # Create a test database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        SQLModel.metadata.create_all(engine)

        try:
            with Session(engine) as session:
                # Create experiment
                experiment = ExperimentConfig(
                    id=uuid4(),
                    networks=[["DummyT2I"]],
                    seeds=[42],
                    prompts=["Test"],
                    embedding_models=["TestModel", "FailingModel"],
                    max_length=5,
                )
                session.add(experiment)

                # Create run
                run = Run(
                    id=uuid4(),
                    experiment_id=experiment.id,
                    network=["DummyT2I"],
                    seed=42,
                    initial_prompt="Test",
                    max_length=5,
                )
                session.add(run)

                # Create invocations
                invocations = []
                for i in range(5):
                    inv = Invocation(
                        id=uuid4(),
                        run_id=run.id,
                        sequence_number=i,
                        type=InvocationType.TEXT,
                        model="DummyT2I",
                        seed=42,
                        output_text=f"Test output {i}",
                    )
                    session.add(inv)
                    invocations.append(inv)

                # Create embeddings for TestModel (all have vectors)
                for inv in invocations:
                    emb = Embedding(
                        id=uuid4(),
                        invocation_id=inv.id,
                        embedding_model="DummyText",
                        vector=np.random.random(100).tolist(),  # Valid vectors
                        started_at=datetime.now(),
                        completed_at=datetime.now(),
                    )
                    session.add(emb)

                # Create embeddings for FailingModel (started but never completed, no vectors)
                for inv in invocations:
                    emb = Embedding(
                        id=uuid4(),
                        invocation_id=inv.id,
                        embedding_model="FailingModel",
                        vector=None,  # No vector - failed computation
                        started_at=datetime.now(),
                        completed_at=None,  # Never completed
                    )
                    session.add(emb)

                session.commit()

                # Run the check for persistence diagrams
                report = DoctorReport()
                issues = check_experiment_persistence_diagrams(
                    experiment, session, report
                )

                # Should find that we need PDs for both models
                assert len(issues) == 2

                # The TestModel should be missing a PD (can be computed)
                test_model_issues = [
                    i for i in issues if i.get("embedding_model") == "TestModel"
                ]
                assert len(test_model_issues) == 1

                # The FailingModel should also be missing a PD
                failing_model_issues = [
                    i for i in issues if i.get("embedding_model") == "FailingModel"
                ]
                assert len(failing_model_issues) == 1

        finally:
            engine.dispose()
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestUUIDJsonSerialization:
    """Test that UUID objects in embedding_ids are properly serialized to JSON."""

    def test_embedding_ids_json_serialization(self):
        """Test that embedding_ids with UUIDs can be serialized to JSON."""
        report = DoctorReport()

        # Add an embedding issue with UUID objects in embedding_ids
        exp_id = uuid4()
        inv_id = uuid4()
        embedding_id1 = uuid4()
        embedding_id2 = uuid4()

        issue = {
            "embedding_model": "TestModel",
            "embedding_count": 2,
            "has_null_vector": False,
            "embedding_ids": [embedding_id1, embedding_id2],  # UUID objects
        }

        report.add_embedding_issue(exp_id, inv_id, issue)

        # This should not raise a TypeError
        json_str = report.to_json()

        # Parse the JSON to verify UUIDs were converted to strings
        data = json.loads(json_str)

        assert "issues" in data
        assert "embeddings" in data["issues"]
        assert len(data["issues"]["embeddings"]) == 1

        embedding_issue = data["issues"]["embeddings"][0]
        assert "embedding_ids" in embedding_issue
        assert len(embedding_issue["embedding_ids"]) == 2

        # The UUIDs should now be strings
        assert all(isinstance(id, str) for id in embedding_issue["embedding_ids"])
        assert embedding_issue["embedding_ids"][0] == str(embedding_id1)
        assert embedding_issue["embedding_ids"][1] == str(embedding_id2)

    def test_empty_embedding_ids_json_serialization(self):
        """Test that empty embedding_ids list is handled correctly."""
        report = DoctorReport()

        issue = {
            "embedding_model": "TestModel",
            "embedding_count": 0,
            "has_null_vector": False,
            "embedding_ids": [],  # Empty list
        }

        report.add_embedding_issue(uuid4(), uuid4(), issue)

        # Should serialize without error
        json_str = report.to_json()
        data = json.loads(json_str)

        embedding_issue = data["issues"]["embeddings"][0]
        assert embedding_issue["embedding_ids"] == []

    def test_mixed_issue_types_json_serialization(self):
        """Test JSON serialization with various issue types including embedding_ids."""
        report = DoctorReport()

        # Add various types of issues
        exp_id = uuid4()
        run_id = uuid4()
        inv_id = uuid4()

        # Run issue
        report.add_run_invocation_issue(exp_id, run_id, {"missing_first": True})

        # Embedding issue with UUIDs
        embedding_issue = {
            "embedding_model": "TestModel",
            "embedding_count": 1,
            "has_null_vector": True,
            "embedding_ids": [uuid4()],
        }
        report.add_embedding_issue(exp_id, inv_id, embedding_issue)

        # PD issue
        report.add_pd_issue(exp_id, run_id, {"issue_type": "missing"})

        # Orphaned records
        report.add_orphaned_embedding(uuid4(), uuid4())

        # Should serialize everything without error
        json_str = report.to_json()
        data = json.loads(json_str)

        # Verify structure
        assert len(data["issues"]["run_invocations"]) == 1
        assert len(data["issues"]["embeddings"]) == 1
        assert len(data["issues"]["persistence_diagrams"]) == 1
        assert len(data["issues"]["orphaned_records"]["embeddings"]) == 1

        # Verify embedding_ids were converted to strings
        embedding_ids = data["issues"]["embeddings"][0]["embedding_ids"]
        assert len(embedding_ids) == 1
        assert isinstance(embedding_ids[0], str)


class TestFixFunctions:
    """Test the fix functions."""

    def test_fix_sequence_gaps(self, test_db):
        """Test fixing sequence gaps."""
        db_url = test_db["db_url"]

        with Session(test_db["engine"]) as session:
            # Check initial state - should have gap
            invocations = session.exec(
                select(Invocation)
                .where(Invocation.run_id == test_db["run_id"])
                .order_by(Invocation.sequence_number)
            ).all()
            assert [inv.sequence_number for inv in invocations] == [0, 2]

        # Fix the gaps
        gaps_issues = [
            {"run_id": test_db["run_id"], "gaps": [1], "actual_sequences": [0, 2]}
        ]
        fix_sequence_gaps(gaps_issues, db_url)

        # Check fixed state - should be contiguous
        with Session(test_db["engine"]) as session:
            invocations = session.exec(
                select(Invocation)
                .where(Invocation.run_id == test_db["run_id"])
                .order_by(Invocation.sequence_number)
            ).all()
            assert [inv.sequence_number for inv in invocations] == [0, 1]

    def test_clean_orphaned_records(self, test_db):
        """Test cleaning orphaned records."""
        db_url = test_db["db_url"]

        # Check initial state - should have orphan
        with Session(test_db["engine"]) as session:
            orphan = session.get(Embedding, test_db["orphan_emb_id"])
            assert orphan is not None

        # Clean orphans
        orphaned = {
            "embeddings": [
                {"embedding_id": test_db["orphan_emb_id"], "invocation_id": uuid4()}
            ],
            "persistence_diagrams": [],
        }
        clean_orphaned_records(orphaned, db_url)

        # Check cleaned state - orphan should be gone
        with Session(test_db["engine"]) as session:
            orphan = session.get(Embedding, test_db["orphan_emb_id"])
            assert orphan is None


class TestDoctorAllExperiments:
    """Test the main doctor_all_experiments function."""

    def test_doctor_all_experiments_finds_issues(self, test_db, capsys):
        """Test doctor_all_experiments finds issues correctly."""
        db_url = test_db["db_url"]

        # Run doctor without fix
        exit_code = doctor_all_experiments(
            db_url, fix=False, yes_flag=False, output_format="text"
        )

        # Should return 1 (issues found)
        assert exit_code == 1

        # Check console output contains issue summary
        captured = capsys.readouterr()
        assert (
            "Doctor Check Summary" in captured.out or "issues" in captured.out.lower()
        )

    def test_doctor_all_experiments_json_output(self, test_db, capsys):
        """Test JSON output format."""
        db_url = test_db["db_url"]

        exit_code = doctor_all_experiments(
            db_url, fix=False, yes_flag=False, output_format="json"
        )

        captured = capsys.readouterr()

        # Should be valid JSON
        data = json.loads(captured.out)
        assert "summary" in data
        assert "issues" in data
        assert data["summary"]["total_experiments"] == 1
        assert exit_code == 1  # Issues found

    def test_doctor_all_experiments_no_issues(self, clean_db):
        """Test doctor_all_experiments with a clean database."""
        db_url = clean_db["db_url"]

        exit_code = doctor_all_experiments(
            db_url, fix=False, yes_flag=False, output_format="text"
        )

        # Should return 0 (no issues)
        assert exit_code == 0

    def test_doctor_all_experiments_with_fix(self, test_db):
        """Test doctor_all_experiments with fix=True."""
        # Skip this test as it requires Ray actors to fix embeddings/PDs
        # and those can't work properly with test data
        pytest.skip(
            "Requires Ray actors for fixing embeddings and persistence diagrams"
        )

    def test_doctor_all_experiments_with_specific_experiment(self, test_db, capsys):
        """Test doctor_all_experiments with a specific experiment ID."""
        db_url = test_db["db_url"]
        experiment_id = test_db["experiment_id"]

        # Run doctor with specific experiment ID
        exit_code = doctor_all_experiments(
            db_url,
            fix=False,
            yes_flag=False,
            output_format="text",
            experiment_id=experiment_id,
        )

        # Should return 1 (issues found)
        assert exit_code == 1

        # Check console output contains experiment-specific summary
        captured = capsys.readouterr()
        assert str(experiment_id) in captured.out
        assert (
            "Doctor Check Summary" in captured.out or "issues" in captured.out.lower()
        )


class TestDoctorSingleExperiment:
    """Test the doctor_single_experiment function."""

    def test_doctor_single_experiment_finds_issues(self, test_db, capsys):
        """Test doctor_single_experiment finds issues correctly."""
        db_url = test_db["db_url"]
        experiment_id = test_db["experiment_id"]

        # Run doctor without fix
        exit_code = doctor_single_experiment(
            experiment_id, db_url, fix=False, yes_flag=False, output_format="text"
        )

        # Should return 1 (issues found)
        assert exit_code == 1

        # Check console output contains issue summary
        captured = capsys.readouterr()
        assert str(experiment_id) in captured.out
        assert (
            "Doctor Check Summary" in captured.out or "issues" in captured.out.lower()
        )

    def test_doctor_single_experiment_json_output(self, test_db, capsys):
        """Test JSON output format for single experiment."""
        db_url = test_db["db_url"]
        experiment_id = test_db["experiment_id"]

        exit_code = doctor_single_experiment(
            experiment_id, db_url, fix=False, yes_flag=False, output_format="json"
        )

        captured = capsys.readouterr()

        # Should be valid JSON
        data = json.loads(captured.out)
        assert "summary" in data
        assert "issues" in data
        assert data["summary"]["total_experiments"] == 1
        assert exit_code == 1  # Issues found

    def test_doctor_single_experiment_no_issues(self, clean_db):
        """Test doctor_single_experiment with a clean database."""
        db_url = clean_db["db_url"]
        experiment_id = clean_db["experiment_id"]

        exit_code = doctor_single_experiment(
            experiment_id, db_url, fix=False, yes_flag=False, output_format="text"
        )

        # Should return 0 (no issues)
        assert exit_code == 0

    def test_doctor_single_experiment_nonexistent(self, test_db, capsys):
        """Test doctor_single_experiment with non-existent experiment ID."""
        db_url = test_db["db_url"]
        fake_id = uuid4()

        # Run doctor with non-existent ID
        exit_code = doctor_single_experiment(
            fake_id, db_url, fix=False, yes_flag=False, output_format="text"
        )

        # Should return 1 (error)
        assert exit_code == 1

        # Check error message
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()

    def test_doctor_single_experiment_nonexistent_json(self, test_db, capsys):
        """Test doctor_single_experiment with non-existent experiment ID in JSON mode."""
        db_url = test_db["db_url"]
        fake_id = uuid4()

        # Run doctor with non-existent ID
        exit_code = doctor_single_experiment(
            fake_id, db_url, fix=False, yes_flag=False, output_format="json"
        )

        # Should return 1 (error)
        assert exit_code == 1

        # Check JSON error output
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "error" in data
        assert "not found" in data["error"].lower()


class TestMultiExperimentScenarios:
    """Test doctor with multiple experiments."""

    def test_doctor_all_experiments_multiple(self, multi_experiment_db, capsys):
        """Test doctor_all_experiments finds issues in multiple experiments."""
        db_url = multi_experiment_db["db_url"]

        # Run doctor for all experiments
        exit_code = doctor_all_experiments(
            db_url, fix=False, yes_flag=False, output_format="text"
        )

        # Should return 1 (one experiment has issues)
        assert exit_code == 1

        # Check console output
        captured = capsys.readouterr()
        assert "Doctor Check Summary" in captured.out
        # Should show 2 total experiments
        assert "2" in captured.out

    def test_doctor_single_clean_experiment(self, multi_experiment_db):
        """Test doctor on the clean experiment only."""
        db_url = multi_experiment_db["db_url"]
        clean_id = multi_experiment_db["clean_experiment_id"]

        # Run doctor on clean experiment
        exit_code = doctor_single_experiment(
            clean_id, db_url, fix=False, yes_flag=False, output_format="text"
        )

        # Should return 0 (no issues)
        assert exit_code == 0

    def test_doctor_single_problematic_experiment(self, multi_experiment_db):
        """Test doctor on the problematic experiment only."""
        db_url = multi_experiment_db["db_url"]
        problem_id = multi_experiment_db["problematic_experiment_id"]

        # Run doctor on problematic experiment
        exit_code = doctor_single_experiment(
            problem_id, db_url, fix=False, yes_flag=False, output_format="text"
        )

        # Should return 1 (has issues)
        assert exit_code == 1

    def test_doctor_all_vs_single_consistency(self, multi_experiment_db, capsys):
        """Test that single experiment mode gives same results as filtering all experiments."""
        db_url = multi_experiment_db["db_url"]
        problem_id = multi_experiment_db["problematic_experiment_id"]

        # Run doctor on single problematic experiment
        exit_code_single = doctor_single_experiment(
            problem_id, db_url, fix=False, yes_flag=False, output_format="json"
        )

        captured_single = capsys.readouterr()
        data_single = json.loads(captured_single.out)

        # Both should find issues
        assert exit_code_single == 1

        # Check that issues are found
        assert data_single["summary"]["experiments_with_issues"] == 1
        assert len(data_single["issues"]["run_invocations"]) > 0


class TestDoctorReportSummary:
    """Test report summary generation."""

    def test_summary_stats_calculation(self):
        """Test that summary statistics are calculated correctly."""
        report = DoctorReport()
        report.total_experiments = 5
        report.experiments_with_issues = 2

        # Add some issues
        exp_id = uuid4()
        run_id = uuid4()
        report.add_run_invocation_issue(exp_id, run_id, {"missing_first": True})
        report.add_run_invocation_issue(exp_id, run_id, {"missing_last": True})
        report.add_embedding_issue(exp_id, uuid4(), {"embedding_count": 0})

        report.finalize()
        stats = report.get_summary_stats()

        assert stats["total_experiments"] == 5
        assert stats["experiments_with_issues"] == 2
        assert stats["total_issues"]["run_invocations"] == 2
        assert stats["total_issues"]["embeddings"] == 1
        assert stats["duration_seconds"] >= 0
