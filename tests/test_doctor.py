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
    fix_run_invocations,
    fix_embeddings,
    fix_persistence_diagrams,
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
            embedding_models=["Dummy", "Dummy2"],
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
        
        # Create embeddings (only for Dummy, missing Dummy2)
        emb1 = Embedding(
            id=uuid4(),
            invocation_id=inv0.id,
            embedding_model="Dummy",
            vector=[0.1, 0.2, 0.3],
        )
        session.add(emb1)
        
        # Create orphaned embedding (with non-existent invocation)
        orphan_emb = Embedding(
            id=uuid4(),
            invocation_id=uuid4(),  # Non-existent invocation
            embedding_model="Dummy",
            vector=[0.4, 0.5, 0.6],
        )
        session.add(orphan_emb)
        
        # Create persistence diagram (only for Dummy, missing Dummy2)
        pd1 = PersistenceDiagram(
            id=uuid4(),
            run_id=run.id,
            embedding_model="Dummy",
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
            embedding_models=["Dummy"],
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
                embedding_model="Dummy",
                vector=[0.1 * i, 0.2 * i, 0.3 * i],
            )
            session.add(emb)
        
        # Create persistence diagram with valid data
        # Use a simple dict that the PersistenceDiagramResultType can handle
        pd = PersistenceDiagram(
            id=uuid4(),
            run_id=run.id,
            embedding_model="Dummy",
            diagram_data={"test": []},  # Use a simple dict for testing
        )
        session.add(pd)
        
        session.commit()
        
        yield {
            "engine": engine,
            "db_url": db_url,
            "db_path": db_path,
            "experiment_id": experiment.id
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
            
            # Should find issues: missing Dummy2 for both invocations, and missing Dummy for inv2
            assert len(issues) > 0
            # Check for missing Dummy2 embeddings
            dummy2_issues = [i for i in issues if i.get("embedding_model") == "Dummy2"]
            assert len(dummy2_issues) == 2  # Both invocations missing Dummy2
    
    def test_check_experiment_persistence_diagrams(self, test_db):
        """Test checking persistence diagrams."""
        with Session(test_db["engine"]) as session:
            experiment = session.get(ExperimentConfig, test_db["experiment_id"])
            report = DoctorReport()
            
            issues = check_experiment_persistence_diagrams(experiment, session, report)
            
            # Should find issues: invalid_model PD, Dummy with null data, and missing Dummy2 PD
            assert len(issues) == 3
            assert any(issue["issue_type"] == "invalid_model" for issue in issues)
            assert any(issue["embedding_model"] == "Dummy2" for issue in issues)
            # Dummy PD exists but has null data, so it's considered missing
            assert any(issue["embedding_model"] == "Dummy" and issue["has_null_data"] for issue in issues)
    
    def test_check_orphaned_records(self, test_db):
        """Test checking for orphaned records."""
        with Session(test_db["engine"]) as session:
            report = DoctorReport()
            
            check_orphaned_records(session, report)
            
            # Should find the orphaned embedding
            assert len(report.orphaned_records["embeddings"]) == 1
            assert report.orphaned_records["global_orphans"]["embeddings"] == 1


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
        gaps_issues = [{
            "run_id": test_db["run_id"],
            "gaps": [1],
            "actual_sequences": [0, 2]
        }]
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
            "embeddings": [{"embedding_id": test_db["orphan_emb_id"], "invocation_id": uuid4()}],
            "persistence_diagrams": []
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
        exit_code = doctor_all_experiments(db_url, fix=False, yes_flag=False, output_format="text")
        
        # Should return 1 (issues found)
        assert exit_code == 1
        
        # Check console output contains issue summary
        captured = capsys.readouterr()
        assert "Doctor Check Summary" in captured.out or "issues" in captured.out.lower()
    
    def test_doctor_all_experiments_json_output(self, test_db, capsys):
        """Test JSON output format."""
        db_url = test_db["db_url"]
        
        exit_code = doctor_all_experiments(db_url, fix=False, yes_flag=False, output_format="json")
        
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
        
        exit_code = doctor_all_experiments(db_url, fix=False, yes_flag=False, output_format="text")
        
        # Should return 0 (no issues)
        assert exit_code == 0
    
    def test_doctor_all_experiments_with_fix(self, test_db):
        """Test doctor_all_experiments with fix=True."""
        # Skip this test as it requires Ray actors to fix embeddings/PDs
        # and those can't work properly with test data
        pytest.skip("Requires Ray actors for fixing embeddings and persistence diagrams")


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