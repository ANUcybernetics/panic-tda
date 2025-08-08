"""
Doctor module for checking and fixing database integrity issues.

This module provides functionality to diagnose and repair data consistency issues
across all experiments in the database, including:
- Missing or incomplete invocation sequences
- Missing or null embeddings
- Missing or invalid persistence diagrams
- Orphaned records
- Non-contiguous sequence numbers
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from sqlmodel import Session, func, select

from panic_tda.schemas import (
    Embedding,
    ExperimentConfig,
    Invocation,
    InvocationType,
    PersistenceDiagram,
    Run,
)
from panic_tda.db import get_session_from_connection_string

logger = logging.getLogger(__name__)
console = Console()


class DoctorReport:
    """Container for doctor check results and statistics."""

    def __init__(self):
        self.total_experiments = 0
        self.experiments_with_issues = 0
        self.run_invocation_issues = []
        self.embedding_issues = []
        self.pd_issues = []
        self.sequence_gap_issues = []
        self.orphaned_records = {
            "embeddings": [],
            "persistence_diagrams": [],
            "global_orphans": {
                "embeddings": 0,
                "persistence_diagrams": 0,
                "invocations": 0,
            },
        }
        self.start_time = datetime.now()
        self.end_time = None

    def add_run_invocation_issue(
        self, experiment_id: UUID, run_id: UUID, issue_details: Dict[str, Any]
    ):
        """Add a run invocation issue to the report."""
        self.run_invocation_issues.append({
            "experiment_id": experiment_id,
            "run_id": run_id,
            **issue_details,
        })

    def add_embedding_issue(
        self, experiment_id: UUID, invocation_id: UUID, issue_details: Dict[str, Any]
    ):
        """Add an embedding issue to the report."""
        self.embedding_issues.append({
            "experiment_id": experiment_id,
            "invocation_id": invocation_id,
            **issue_details,
        })

    def add_pd_issue(
        self, experiment_id: UUID, run_id: UUID, issue_details: Dict[str, Any]
    ):
        """Add a persistence diagram issue to the report."""
        self.pd_issues.append({
            "experiment_id": experiment_id,
            "run_id": run_id,
            **issue_details,
        })

    def add_sequence_gap_issue(
        self,
        experiment_id: UUID,
        run_id: UUID,
        gaps: List[int],
        actual_sequences: List[int],
    ):
        """Add a sequence gap issue to the report."""
        self.sequence_gap_issues.append({
            "experiment_id": experiment_id,
            "run_id": run_id,
            "gaps": gaps,
            "actual_sequences": actual_sequences,
        })

    def add_orphaned_embedding(self, embedding_id: UUID, invocation_id: UUID):
        """Add an orphaned embedding to the report."""
        self.orphaned_records["embeddings"].append({
            "embedding_id": embedding_id,
            "invocation_id": invocation_id,
        })

    def add_orphaned_pd(self, pd_id: UUID, run_id: UUID):
        """Add an orphaned persistence diagram to the report."""
        self.orphaned_records["persistence_diagrams"].append({
            "pd_id": pd_id,
            "run_id": run_id,
        })

    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return (
            len(self.run_invocation_issues) > 0
            or len(self.embedding_issues) > 0
            or len(self.pd_issues) > 0
            or len(self.sequence_gap_issues) > 0
            or len(self.orphaned_records["embeddings"]) > 0
            or len(self.orphaned_records["persistence_diagrams"]) > 0
            or any(self.orphaned_records["global_orphans"].values())
        )

    def finalize(self):
        """Mark the report as complete."""
        self.end_time = datetime.now()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the report."""
        duration = (
            (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        )
        return {
            "total_experiments": self.total_experiments,
            "experiments_with_issues": self.experiments_with_issues,
            "total_issues": {
                "run_invocations": len(self.run_invocation_issues),
                "embeddings": len(self.embedding_issues),
                "persistence_diagrams": len(self.pd_issues),
                "sequence_gaps": len(self.sequence_gap_issues),
                "orphaned_embeddings": len(self.orphaned_records["embeddings"]),
                "orphaned_pds": len(self.orphaned_records["persistence_diagrams"]),
                "global_orphans": self.orphaned_records["global_orphans"],
            },
            "duration_seconds": duration,
        }

    def to_json(self) -> str:
        """Convert the report to JSON format."""
        return json.dumps(
            {
                "summary": self.get_summary_stats(),
                "issues": {
                    "run_invocations": [
                        {
                            "experiment_id": str(issue["experiment_id"]),
                            "run_id": str(issue["run_id"]),
                            **{
                                k: v
                                for k, v in issue.items()
                                if k not in ["experiment_id", "run_id"]
                            },
                        }
                        for issue in self.run_invocation_issues
                    ],
                    "embeddings": [
                        {
                            "experiment_id": str(issue["experiment_id"]),
                            "invocation_id": str(issue["invocation_id"]),
                            **{
                                k: (
                                    [str(id) for id in v] if k == "embedding_ids" else v
                                )
                                for k, v in issue.items()
                                if k not in ["experiment_id", "invocation_id"]
                            },
                        }
                        for issue in self.embedding_issues
                    ],
                    "persistence_diagrams": [
                        {
                            "experiment_id": str(issue["experiment_id"]),
                            "run_id": str(issue["run_id"]),
                            **{
                                k: v
                                for k, v in issue.items()
                                if k not in ["experiment_id", "run_id"]
                            },
                        }
                        for issue in self.pd_issues
                    ],
                    "sequence_gaps": [
                        {
                            "experiment_id": str(issue["experiment_id"]),
                            "run_id": str(issue["run_id"]),
                            "gaps": issue["gaps"],
                            "actual_sequences": issue["actual_sequences"],
                        }
                        for issue in self.sequence_gap_issues
                    ],
                    "orphaned_records": {
                        "embeddings": [
                            {
                                "embedding_id": str(record["embedding_id"]),
                                "invocation_id": str(record["invocation_id"]),
                            }
                            for record in self.orphaned_records["embeddings"]
                        ],
                        "persistence_diagrams": [
                            {
                                "pd_id": str(record["pd_id"]),
                                "run_id": str(record["run_id"]),
                            }
                            for record in self.orphaned_records["persistence_diagrams"]
                        ],
                        "global_orphans": self.orphaned_records["global_orphans"],
                    },
                },
            },
            indent=2,
        )


def check_experiment_run_invocations(
    experiment: ExperimentConfig, session: Session, report: DoctorReport
) -> List[Dict[str, Any]]:
    """Check that all runs in an experiment have the expected invocations."""
    issues = []

    runs = session.exec(select(Run).where(Run.experiment_id == experiment.id)).all()

    for run in runs:
        # Check invocation count
        invocation_count = session.exec(
            select(func.count())
            .select_from(Invocation)
            .where(Invocation.run_id == run.id)
        ).one()

        # Check for sequence_number 0
        has_first = (
            session.exec(
                select(Invocation).where(
                    Invocation.run_id == run.id, Invocation.sequence_number == 0
                )
            ).first()
            is not None
        )

        # Check for sequence_number (max_length - 1)
        has_last = (
            session.exec(
                select(Invocation).where(
                    Invocation.run_id == run.id,
                    Invocation.sequence_number == run.max_length - 1,
                )
            ).first()
            is not None
        )

        if not has_first or not has_last or invocation_count != run.max_length:
            issue = {
                "missing_first": not has_first,
                "missing_last": not has_last,
                "expected_count": run.max_length,
                "actual_count": invocation_count,
            }
            issues.append({"run_id": run.id, **issue})
            report.add_run_invocation_issue(experiment.id, run.id, issue)

        # Check for sequence gaps
        invocation_sequences = session.exec(
            select(Invocation.sequence_number)
            .where(Invocation.run_id == run.id)
            .order_by(Invocation.sequence_number)
        ).all()

        if invocation_sequences:
            expected_sequences = list(range(run.max_length))
            actual_sequences = list(invocation_sequences)
            gaps = [seq for seq in expected_sequences if seq not in actual_sequences]

            if gaps:
                report.add_sequence_gap_issue(
                    experiment.id, run.id, gaps, actual_sequences
                )

    return issues


def check_experiment_embeddings(
    experiment: ExperimentConfig, session: Session, report: DoctorReport
) -> List[Dict[str, Any]]:
    """Check that all invocations in an experiment have proper embeddings."""
    from panic_tda.embeddings import get_model_type, EmbeddingModelType

    issues = []

    # Get all invocations (both TEXT and IMAGE) for all runs in this experiment
    invocations = session.exec(
        select(Invocation)
        .join(Run, Invocation.run_id == Run.id)
        .where(Run.experiment_id == experiment.id)
    ).all()

    for invocation in invocations:
        # First check for mismatched embeddings (e.g., image model on text invocation)
        for embedding_model in experiment.embedding_models:
            try:
                model_type = get_model_type(embedding_model)
                is_mismatched = False

                # Check if this model type is incompatible with invocation type
                if (
                    invocation.type == InvocationType.TEXT
                    and model_type == EmbeddingModelType.IMAGE
                ):
                    is_mismatched = True
                elif (
                    invocation.type == InvocationType.IMAGE
                    and model_type == EmbeddingModelType.TEXT
                ):
                    is_mismatched = True

                if is_mismatched:
                    # Check if there are any embeddings of this mismatched type
                    mismatched_embeddings = session.exec(
                        select(Embedding).where(
                            Embedding.invocation_id == invocation.id,
                            Embedding.embedding_model == embedding_model,
                        )
                    ).all()

                    if mismatched_embeddings:
                        # Report this as a mismatched embedding issue
                        embedding_ids = [e.id for e in mismatched_embeddings]
                        issue = {
                            "invocation_type": invocation.type.value,
                            "embedding_model": embedding_model,
                            "issue_type": "mismatched",
                            "embedding_count": len(mismatched_embeddings),
                            "has_null_vector": any(
                                e.vector is None for e in mismatched_embeddings
                            ),
                            "embedding_ids": embedding_ids,
                        }
                        issues.append({"invocation_id": invocation.id, **issue})
                        report.add_embedding_issue(experiment.id, invocation.id, issue)
                    continue  # Don't check for missing/null embeddings for mismatched types

            except ValueError:
                # If we can't determine model type, skip (will be logged elsewhere)
                logger.warning(
                    f"Could not determine type for embedding model {embedding_model}"
                )
                continue

        # Now check for missing or null embeddings for compatible model types
        for embedding_model in experiment.embedding_models:
            # Check if this embedding model is compatible with the invocation type
            try:
                model_type = get_model_type(embedding_model)
                # Skip if model type doesn't match invocation type
                if (
                    invocation.type == InvocationType.TEXT
                    and model_type != EmbeddingModelType.TEXT
                ):
                    continue
                if (
                    invocation.type == InvocationType.IMAGE
                    and model_type != EmbeddingModelType.IMAGE
                ):
                    continue
            except ValueError:
                # Already warned above
                continue

            # Count embeddings for this invocation and model
            embedding_count = session.exec(
                select(func.count())
                .select_from(Embedding)
                .where(
                    Embedding.invocation_id == invocation.id,
                    Embedding.embedding_model == embedding_model,
                )
            ).one()

            # Check for null vectors by retrieving and checking actual numpy arrays
            embeddings = session.exec(
                select(Embedding).where(
                    Embedding.invocation_id == invocation.id,
                    Embedding.embedding_model == embedding_model,
                )
            ).all()

            null_vectors = sum(1 for e in embeddings if e.vector is None)

            if embedding_count != 1 or null_vectors > 0:
                # Get embedding IDs for debugging
                embedding_ids = [e.id for e in embeddings]
                issue = {
                    "invocation_type": invocation.type.value,
                    "embedding_model": embedding_model,
                    "embedding_count": embedding_count,
                    "has_null_vector": null_vectors > 0,
                    "embedding_ids": embedding_ids,
                }
                issues.append({"invocation_id": invocation.id, **issue})
                report.add_embedding_issue(experiment.id, invocation.id, issue)

    return issues


def check_experiment_persistence_diagrams(
    experiment: ExperimentConfig, session: Session, report: DoctorReport
) -> List[Dict[str, Any]]:
    """Check that all runs in an experiment have proper persistence diagrams."""
    issues = []

    runs = session.exec(select(Run).where(Run.experiment_id == experiment.id)).all()
    run_ids = [run.id for run in runs]

    # Check for PDs with invalid embedding models
    if run_ids:
        all_pds = session.exec(
            select(PersistenceDiagram).where(PersistenceDiagram.run_id.in_(run_ids))
        ).all()

        for pd in all_pds:
            if pd.embedding_model not in experiment.embedding_models:
                issue = {
                    "embedding_model": pd.embedding_model,
                    "issue_type": "invalid_model",
                    "pd_count": 1,
                    "has_null_data": pd.diagram_data is None,
                }
                issues.append({"run_id": pd.run_id, **issue})
                report.add_pd_issue(experiment.id, pd.run_id, issue)

    # Check for missing or duplicate PDs for valid models
    for run in runs:
        for embedding_model in experiment.embedding_models:
            pd_count = session.exec(
                select(func.count())
                .select_from(PersistenceDiagram)
                .where(
                    PersistenceDiagram.run_id == run.id,
                    PersistenceDiagram.embedding_model == embedding_model,
                )
            ).one()

            null_data = session.exec(
                select(func.count())
                .select_from(PersistenceDiagram)
                .where(
                    PersistenceDiagram.run_id == run.id,
                    PersistenceDiagram.embedding_model == embedding_model,
                    PersistenceDiagram.diagram_data.is_(None),
                )
            ).one()

            if pd_count == 0 or null_data == pd_count:
                issue = {
                    "embedding_model": embedding_model,
                    "issue_type": "missing",
                    "pd_count": pd_count,
                    "has_null_data": null_data > 0,
                }
                issues.append({"run_id": run.id, **issue})
                report.add_pd_issue(experiment.id, run.id, issue)
            elif pd_count > 1:
                issue = {
                    "embedding_model": embedding_model,
                    "issue_type": "duplicate",
                    "pd_count": pd_count,
                    "has_null_data": null_data > 0,
                }
                issues.append({"run_id": run.id, **issue})
                report.add_pd_issue(experiment.id, run.id, issue)

    return issues


def check_orphaned_records(session: Session, report: DoctorReport):
    """Check for orphaned records across all tables."""
    # Check for orphaned embeddings (embeddings without valid invocations)
    orphaned_embeddings = session.exec(
        select(Embedding.id, Embedding.invocation_id)
        .outerjoin(Invocation, Embedding.invocation_id == Invocation.id)
        .where(Invocation.id.is_(None))
    ).all()

    for emb_id, inv_id in orphaned_embeddings:
        report.add_orphaned_embedding(emb_id, inv_id)

    # Check for orphaned persistence diagrams (PDs without valid runs)
    orphaned_pds = session.exec(
        select(PersistenceDiagram.id, PersistenceDiagram.run_id)
        .outerjoin(Run, PersistenceDiagram.run_id == Run.id)
        .where(Run.id.is_(None))
    ).all()

    for pd_id, run_id in orphaned_pds:
        report.add_orphaned_pd(pd_id, run_id)

    # Check for orphaned invocations (invocations without valid runs)
    orphaned_invocations = session.exec(
        select(func.count())
        .select_from(Invocation)
        .outerjoin(Run, Invocation.run_id == Run.id)
        .where(Run.id.is_(None))
    ).one()

    report.orphaned_records["global_orphans"]["invocations"] = orphaned_invocations
    report.orphaned_records["global_orphans"]["embeddings"] = len(orphaned_embeddings)
    report.orphaned_records["global_orphans"]["persistence_diagrams"] = len(
        orphaned_pds
    )


def confirm_fix(message: str, yes_flag: bool) -> bool:
    """Prompt for confirmation unless --yes flag is set."""
    if yes_flag:
        return True
    return typer.confirm(message)


def fix_run_invocations(issues: List[Dict], experiment: ExperimentConfig, db_str: str):
    """Fix missing or extra invocations for runs."""
    from panic_tda.engine import perform_runs_stage

    with get_session_from_connection_string(db_str) as session:
        run_ids_to_fix = []
        for issue in issues:
            run_id = issue["run_id"]
            session.get(Run, run_id)

            # Delete all existing invocations for this run
            session.exec(select(Invocation).where(Invocation.run_id == run_id)).all()
            for inv in session.exec(
                select(Invocation).where(Invocation.run_id == run_id)
            ):
                session.delete(inv)

            run_ids_to_fix.append(str(run_id))

        session.commit()

    # Re-run the runs to generate proper invocations
    if run_ids_to_fix:
        logger.info(f"Re-running {len(run_ids_to_fix)} runs to fix invocations")
        perform_runs_stage(run_ids_to_fix, db_str)


def fix_embeddings(issues: List[Dict], experiment: ExperimentConfig, db_str: str):
    """Fix missing or invalid embeddings."""
    from panic_tda.engine import perform_embeddings_stage
    from panic_tda.embeddings import get_model_type, EmbeddingModelType

    # First, handle mismatched embeddings from the issues list
    mismatched_issues = [i for i in issues if i.get("issue_type") == "mismatched"]
    if mismatched_issues:
        with get_session_from_connection_string(db_str) as session:
            mismatched_count = 0
            for issue in mismatched_issues:
                # Delete the mismatched embeddings referenced in the issue
                for embedding_id in issue.get("embedding_ids", []):
                    emb = session.get(Embedding, embedding_id)
                    if emb:
                        session.delete(emb)
                        mismatched_count += 1

            if mismatched_count > 0:
                logger.info(
                    f"Deleted {mismatched_count} mismatched embeddings from issues"
                )
                session.commit()

    # Also do a comprehensive cleanup of all mismatched embeddings for this experiment
    with get_session_from_connection_string(db_str) as session:
        # Get all invocations for this experiment
        invocations = session.exec(
            select(Invocation)
            .join(Run, Invocation.run_id == Run.id)
            .where(Run.experiment_id == experiment.id)
        ).all()

        mismatched_count = 0
        for invocation in invocations:
            for embedding_model in experiment.embedding_models:
                try:
                    model_type = get_model_type(embedding_model)
                    # Check for mismatched embeddings
                    should_delete = False
                    if (
                        invocation.type == InvocationType.TEXT
                        and model_type == EmbeddingModelType.IMAGE
                    ):
                        should_delete = True
                    elif (
                        invocation.type == InvocationType.IMAGE
                        and model_type == EmbeddingModelType.TEXT
                    ):
                        should_delete = True

                    if should_delete:
                        # Delete any embeddings of this model for this invocation
                        embeddings_to_delete = session.exec(
                            select(Embedding).where(
                                Embedding.invocation_id == invocation.id,
                                Embedding.embedding_model == embedding_model,
                            )
                        ).all()

                        for emb in embeddings_to_delete:
                            session.delete(emb)
                            mismatched_count += 1

                except ValueError:
                    # Skip models without defined types
                    continue

        if mismatched_count > 0:
            logger.info(
                f"Deleted {mismatched_count} additional mismatched embeddings in cleanup"
            )
            session.commit()

    # Now compute embeddings for invocations that need them (excluding mismatched ones)
    invocation_ids_to_fix = list(
        set(
            str(issue["invocation_id"])
            for issue in issues
            if issue.get("issue_type") != "mismatched"
        )
    )

    if invocation_ids_to_fix:
        logger.info(
            f"Computing embeddings for {len(invocation_ids_to_fix)} invocations"
        )
        perform_embeddings_stage(
            invocation_ids_to_fix, experiment.embedding_models, db_str
        )


def fix_persistence_diagrams(
    issues: List[Dict], experiment: ExperimentConfig, db_str: str
):
    """Fix missing or invalid persistence diagrams."""
    from panic_tda.engine import perform_pd_stage

    # Group issues by type
    runs_to_recompute = set()

    with get_session_from_connection_string(db_str) as session:
        for issue in issues:
            if issue["issue_type"] == "invalid_model":
                # Delete PDs with invalid embedding models
                pds = session.exec(
                    select(PersistenceDiagram).where(
                        PersistenceDiagram.run_id == issue["run_id"],
                        PersistenceDiagram.embedding_model == issue["embedding_model"],
                    )
                ).all()
                for pd in pds:
                    session.delete(pd)
            elif issue["issue_type"] in ["missing", "duplicate"]:
                runs_to_recompute.add(str(issue["run_id"]))
                # Delete existing PDs if duplicates
                if issue["issue_type"] == "duplicate":
                    pds = session.exec(
                        select(PersistenceDiagram).where(
                            PersistenceDiagram.run_id == issue["run_id"],
                            PersistenceDiagram.embedding_model
                            == issue["embedding_model"],
                        )
                    ).all()
                    for pd in pds:
                        session.delete(pd)

        session.commit()

    # Recompute PDs for affected runs
    if runs_to_recompute:
        run_ids = list(runs_to_recompute)
        logger.info(f"Computing persistence diagrams for {len(run_ids)} runs")
        perform_pd_stage(run_ids, experiment.embedding_models, db_str)


def fix_sequence_gaps(gaps_issues: List[Dict], db_str: str):
    """Fix sequence number gaps by re-sequencing invocations."""
    with get_session_from_connection_string(db_str) as session:
        for issue in gaps_issues:
            run_id = issue["run_id"]

            # Get all invocations for this run, ordered by current sequence
            invocations = session.exec(
                select(Invocation)
                .where(Invocation.run_id == run_id)
                .order_by(Invocation.sequence_number)
            ).all()

            # Re-sequence them to be contiguous
            for new_seq, inv in enumerate(invocations):
                inv.sequence_number = new_seq
                session.add(inv)

        session.commit()
        logger.info(f"Fixed sequence gaps for {len(gaps_issues)} runs")


def clean_orphaned_records(orphaned: Dict, db_str: str):
    """Clean up orphaned records from the database."""
    with get_session_from_connection_string(db_str) as session:
        # Clean orphaned embeddings
        for record in orphaned["embeddings"]:
            emb = session.get(Embedding, record["embedding_id"])
            if emb:
                session.delete(emb)

        # Clean orphaned persistence diagrams
        for record in orphaned["persistence_diagrams"]:
            pd = session.get(PersistenceDiagram, record["pd_id"])
            if pd:
                session.delete(pd)

        # Clean globally orphaned invocations
        orphaned_invs = session.exec(
            select(Invocation)
            .outerjoin(Run, Invocation.run_id == Run.id)
            .where(Run.id.is_(None))
        ).all()

        for inv in orphaned_invs:
            session.delete(inv)

        session.commit()

        total_cleaned = (
            len(orphaned["embeddings"])
            + len(orphaned["persistence_diagrams"])
            + len(orphaned_invs)
        )
        logger.info(f"Cleaned {total_cleaned} orphaned records")


def _check_and_fix_experiment(
    experiment_id: UUID,
    db_str: str,
    report: DoctorReport,
    fix: bool,
    yes_flag: bool,
    progress=None,
    task=None,
    experiment_index: Optional[int] = None,
    total_experiments: Optional[int] = None,
) -> bool:
    """
    Helper function to check and optionally fix a single experiment.

    Returns:
        True if the experiment has issues, False otherwise.
    """
    experiment_has_issues = False

    with get_session_from_connection_string(db_str) as session:
        # Load experiment in this session
        experiment = session.get(ExperimentConfig, experiment_id)
        if not experiment:
            raise ValueError(f"Experiment with ID {experiment_id} not found")

        # Check run invocations
        run_issues = check_experiment_run_invocations(experiment, session, report)
        if progress and task:
            progress.advance(task)

        # Check embeddings
        embedding_issues = check_experiment_embeddings(experiment, session, report)
        if progress and task:
            progress.advance(task)

        # Check persistence diagrams
        pd_issues = check_experiment_persistence_diagrams(experiment, session, report)
        if progress and task:
            progress.advance(task)

        if run_issues or embedding_issues or pd_issues:
            experiment_has_issues = True
            report.experiments_with_issues += 1

            if fix:
                # Fix issues for this experiment
                if run_issues and confirm_fix(
                    f"Fix {len(run_issues)} run invocation issues for experiment {experiment_id}?",
                    yes_flag,
                ):
                    fix_run_invocations(run_issues, experiment, db_str)
                    # Re-check embeddings after fixing invocations
                    with get_session_from_connection_string(db_str) as session:
                        experiment = session.get(ExperimentConfig, experiment_id)
                        embedding_issues = check_experiment_embeddings(
                            experiment, session, DoctorReport()
                        )

                if embedding_issues and confirm_fix(
                    f"Fix {len(embedding_issues)} embedding issues for experiment {experiment_id}?",
                    yes_flag,
                ):
                    fix_embeddings(embedding_issues, experiment, db_str)
                    # Re-check persistence diagrams after fixing embeddings
                    with get_session_from_connection_string(db_str) as session:
                        experiment = session.get(ExperimentConfig, experiment_id)
                        pd_issues = check_experiment_persistence_diagrams(
                            experiment, session, DoctorReport()
                        )

                if pd_issues and confirm_fix(
                    f"Fix {len(pd_issues)} persistence diagram issues for experiment {experiment_id}?",
                    yes_flag,
                ):
                    fix_persistence_diagrams(pd_issues, experiment, db_str)

    # Update progress if provided
    if progress and task:
        if experiment_index is not None and total_experiments is not None:
            description = f"Experiment {experiment_index + 1}/{total_experiments} ({experiment_id}): {'issues found' if experiment_has_issues else 'OK'}"
        else:
            description = f"Experiment {experiment_id}: {'issues found' if experiment_has_issues else 'OK'}"
        progress.update(task, completed=True, description=description)

    return experiment_has_issues


def doctor_single_experiment(
    experiment_id: UUID,
    db_str: str,
    fix: bool = False,
    yes_flag: bool = False,
    output_format: str = "text",
) -> int:
    """
    Check and optionally fix a single experiment in the database.

    Args:
        experiment_id: UUID of the experiment to check
        db_str: Database connection string
        fix: If True, fix issues; if False, only report
        yes_flag: Skip confirmation prompts if True
        output_format: Output format ('text' or 'json')

    Returns:
        Exit code (0 if no issues, 1 if issues found)
    """
    report = DoctorReport()

    # Only show progress indicator for text output
    show_progress = output_format == "text"

    if show_progress:
        progress_context = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        )
    else:
        # Create a dummy context manager that does nothing
        from contextlib import nullcontext

        progress_context = nullcontext()

    with progress_context as progress:
        # Load the specific experiment
        if show_progress:
            task = progress.add_task(
                f"Loading experiment {experiment_id}...", total=None
            )

        with get_session_from_connection_string(db_str) as session:
            experiment = session.get(ExperimentConfig, experiment_id)
            if not experiment:
                if output_format == "json":
                    error_report = {
                        "error": f"Experiment with ID {experiment_id} not found",
                        "summary": {"total_experiments": 0},
                        "issues": {},
                    }
                    print(json.dumps(error_report, indent=2))
                else:
                    console.print(
                        f"[red]Error: Experiment with ID {experiment_id} not found[/red]"
                    )
                return 1

            report.total_experiments = 1

        if show_progress:
            progress.update(
                task, completed=True, description=f"Found experiment {experiment_id}"
            )

        # Process the experiment
        if show_progress:
            task = progress.add_task(
                f"Checking experiment {experiment_id}",
                total=3,
            )
        else:
            task = None

        # Use the helper function to check and fix the experiment
        try:
            _check_and_fix_experiment(
                experiment_id=experiment_id,
                db_str=db_str,
                report=report,
                fix=fix,
                yes_flag=yes_flag,
                progress=progress if show_progress else None,
                task=task,
            )
        except ValueError as e:
            # Should not happen since we already checked, but handle gracefully
            if output_format == "json":
                error_report = {
                    "error": str(e),
                    "summary": {"total_experiments": 0},
                    "issues": {},
                }
                print(json.dumps(error_report, indent=2))
            else:
                console.print(f"[red]Error: {e}[/red]")
            return 1

        # Check for orphaned records related to this experiment
        if show_progress:
            task = progress.add_task("Checking for orphaned records...", total=None)

        with get_session_from_connection_string(db_str) as session:
            # We still check all orphaned records, but this could be optimized
            # to check only records related to this experiment
            check_orphaned_records(session, report)

        if show_progress:
            progress.update(task, completed=True, description="Orphan check complete")

        # Fix orphaned records if requested
        if fix and (
            report.orphaned_records["embeddings"]
            or report.orphaned_records["persistence_diagrams"]
        ):
            if confirm_fix(
                "Clean orphaned records (globally)?",
                yes_flag,
            ):
                clean_orphaned_records(report.orphaned_records, db_str)

        # Fix sequence gaps if requested
        if fix and report.sequence_gap_issues:
            if confirm_fix(
                f"Fix sequence gaps in {len(report.sequence_gap_issues)} runs for experiment {experiment_id}?",
                yes_flag,
            ):
                fix_sequence_gaps(report.sequence_gap_issues, db_str)

    report.finalize()

    # Output results
    if output_format == "json":
        print(report.to_json())
    else:
        # Display summary table
        table = Table(title=f"Doctor Check Summary for Experiment {experiment_id}")
        table.add_column("Category", style="cyan")
        table.add_column("Issue Details", style="yellow")

        stats = report.get_summary_stats()

        # Summary row
        table.add_row(
            "Summary",
            f"{stats['experiments_with_issues']}/{stats['total_experiments']} experiments with issues",
        )

        # Run invocation issues
        if report.run_invocation_issues:
            for issue in report.run_invocation_issues[:10]:  # Show first 10
                run_id_str = str(issue["run_id"])[:8]
                details = []
                if issue.get("missing_first"):
                    details.append("missing seq 0")
                if issue.get("missing_last"):
                    details.append(f"missing seq {issue.get('expected_count', 0) - 1}")
                if issue.get("actual_count") != issue.get("expected_count"):
                    details.append(
                        f"has {issue.get('actual_count')}/{issue.get('expected_count')} invocations"
                    )
                table.add_row(f"Run {run_id_str}", ", ".join(details))
            if len(report.run_invocation_issues) > 10:
                table.add_row(
                    "",
                    f"... and {len(report.run_invocation_issues) - 10} more run issues",
                )

        # Embedding issues
        if report.embedding_issues:
            shown = 0
            for issue in report.embedding_issues[:10]:  # Show first 10
                inv_id_str = str(issue["invocation_id"])[:8]
                details = []
                if issue.get("embedding_count", 0) == 0:
                    details.append(
                        f"missing embedding for {issue.get('embedding_model')}"
                    )
                elif issue.get("embedding_count", 0) > 1:
                    details.append(
                        f"{issue.get('embedding_count')} duplicates for {issue.get('embedding_model')}"
                    )
                if issue.get("has_null_vector"):
                    details.append(f"null vector for {issue.get('embedding_model')}")
                table.add_row(f"Invocation {inv_id_str}", ", ".join(details))
                shown += 1
            if len(report.embedding_issues) > 10:
                table.add_row(
                    "",
                    f"... and {len(report.embedding_issues) - 10} more embedding issues",
                )

        # Persistence diagram issues
        if report.pd_issues:
            for issue in report.pd_issues[:10]:  # Show first 10
                run_id_str = str(issue["run_id"])[:8]
                issue_type = issue.get("issue_type", "unknown")
                model = issue.get("embedding_model", "unknown")
                if issue_type == "missing":
                    details = f"missing PD for {model}"
                elif issue_type == "duplicate":
                    details = f"{issue.get('pd_count', 0)} duplicate PDs for {model}"
                elif issue_type == "invalid_model":
                    details = f"PD with invalid model {model}"
                else:
                    details = f"{issue_type} issue for {model}"
                if issue.get("has_null_data"):
                    details += " (null data)"
                table.add_row(f"Run {run_id_str}", details)
            if len(report.pd_issues) > 10:
                table.add_row(
                    "", f"... and {len(report.pd_issues) - 10} more PD issues"
                )

        # Sequence gap issues
        if report.sequence_gap_issues:
            for issue in report.sequence_gap_issues[:10]:  # Show first 10
                run_id_str = str(issue["run_id"])[:8]
                gaps = issue.get("gaps", [])
                if len(gaps) <= 5:
                    gap_str = ", ".join(str(g) for g in gaps)
                else:
                    gap_str = f"{', '.join(str(g) for g in gaps[:3])}... ({len(gaps)} gaps total)"
                table.add_row(f"Run {run_id_str}", f"missing sequences: {gap_str}")
            if len(report.sequence_gap_issues) > 10:
                table.add_row(
                    "",
                    f"... and {len(report.sequence_gap_issues) - 10} more sequence gap issues",
                )

        # Orphaned records
        if report.orphaned_records["embeddings"]:
            count = len(report.orphaned_records["embeddings"])
            sample = report.orphaned_records["embeddings"][:3]
            sample_str = ", ".join(str(r["embedding_id"])[:8] for r in sample)
            if count > 3:
                sample_str += f"... ({count} total)"
            table.add_row("Orphaned Embeddings", sample_str)

        if report.orphaned_records["persistence_diagrams"]:
            count = len(report.orphaned_records["persistence_diagrams"])
            sample = report.orphaned_records["persistence_diagrams"][:3]
            sample_str = ", ".join(str(r["pd_id"])[:8] for r in sample)
            if count > 3:
                sample_str += f"... ({count} total)"
            table.add_row("Orphaned PDs", sample_str)

        # Global orphans
        global_orphans = report.orphaned_records["global_orphans"]
        if any(global_orphans.values()):
            details = []
            if global_orphans["invocations"]:
                details.append(f"{global_orphans['invocations']} invocations")
            if global_orphans["embeddings"]:
                details.append(f"{global_orphans['embeddings']} embeddings")
            if global_orphans["persistence_diagrams"]:
                details.append(f"{global_orphans['persistence_diagrams']} PDs")
            table.add_row("Global Orphans", ", ".join(details))

        # Duration
        table.add_row("Duration", f"{stats['duration_seconds']:.2f} seconds")

        console.print(table)

        if not fix and report.has_issues():
            console.print("\n[yellow]Use --fix flag to repair issues[/yellow]")

    return 1 if report.has_issues() else 0


def doctor_all_experiments(
    db_str: str,
    fix: bool = False,
    yes_flag: bool = False,
    output_format: str = "text",
    experiment_id: Optional[UUID] = None,
) -> int:
    """
    Check and optionally fix all experiments in the database, or a specific experiment if provided.

    Args:
        db_str: Database connection string
        fix: If True, fix issues; if False, only report
        yes_flag: Skip confirmation prompts if True
        output_format: Output format ('text' or 'json')
        experiment_id: Optional UUID of a specific experiment to check

    Returns:
        Exit code (0 if no issues, 1 if issues found)
    """
    # If a specific experiment is requested, use the single experiment function
    if experiment_id is not None:
        return doctor_single_experiment(
            experiment_id, db_str, fix, yes_flag, output_format
        )

    report = DoctorReport()

    # Only show progress indicator for text output
    show_progress = output_format == "text"

    if show_progress:
        progress_context = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        )
    else:
        # Create a dummy context manager that does nothing
        from contextlib import nullcontext

        progress_context = nullcontext()

    with progress_context as progress:
        # Load all experiments
        if show_progress:
            task = progress.add_task("Loading experiments...", total=None)

        with get_session_from_connection_string(db_str) as session:
            experiments = session.exec(select(ExperimentConfig)).all()
            # Store just the IDs to avoid detached instance issues
            experiment_ids = [exp.id for exp in experiments]
            report.total_experiments = len(experiment_ids)

        if show_progress:
            progress.update(
                task,
                completed=True,
                description=f"Found {report.total_experiments} experiments",
            )

        # Process each experiment
        for i, experiment_id in enumerate(experiment_ids):
            if show_progress:
                task = progress.add_task(
                    f"Checking experiment {i + 1}/{report.total_experiments} ({experiment_id})",
                    total=3,
                )
            else:
                task = None

            # Use the helper function to check and fix the experiment
            try:
                _check_and_fix_experiment(
                    experiment_id=experiment_id,
                    db_str=db_str,
                    report=report,
                    fix=fix,
                    yes_flag=yes_flag,
                    progress=progress if show_progress else None,
                    task=task,
                    experiment_index=i,
                    total_experiments=report.total_experiments,
                )
            except ValueError as e:
                # Log error and continue with next experiment
                logger.error(f"Error processing experiment {experiment_id}: {e}")
                if show_progress:
                    progress.update(
                        task,
                        completed=True,
                        description=f"Experiment {i + 1}/{report.total_experiments} ({experiment_id}): ERROR",
                    )

        # Check for global orphaned records
        if show_progress:
            task = progress.add_task("Checking for orphaned records...", total=None)

        with get_session_from_connection_string(db_str) as session:
            check_orphaned_records(session, report)

        if show_progress:
            progress.update(task, completed=True, description="Orphan check complete")

        # Fix orphaned records if requested
        if fix and report.orphaned_records["global_orphans"]["invocations"] > 0:
            if confirm_fix(
                f"Clean {sum(report.orphaned_records['global_orphans'].values())} orphaned records?",
                yes_flag,
            ):
                clean_orphaned_records(report.orphaned_records, db_str)

        # Fix sequence gaps if requested
        if fix and report.sequence_gap_issues:
            if confirm_fix(
                f"Fix sequence gaps in {len(report.sequence_gap_issues)} runs?",
                yes_flag,
            ):
                fix_sequence_gaps(report.sequence_gap_issues, db_str)

    report.finalize()

    # Output results
    if output_format == "json":
        print(report.to_json())
    else:
        # Display summary table
        table = Table(title="Doctor Check Summary")
        table.add_column("Category", style="cyan")
        table.add_column("Issue Details", style="yellow")

        stats = report.get_summary_stats()

        # Summary row
        table.add_row(
            "Summary",
            f"{stats['experiments_with_issues']}/{stats['total_experiments']} experiments with issues",
        )

        # Run invocation issues
        if report.run_invocation_issues:
            for issue in report.run_invocation_issues[:10]:  # Show first 10
                exp_id_str = str(issue["experiment_id"])[:8]
                run_id_str = str(issue["run_id"])[:8]
                details = []
                if issue.get("missing_first"):
                    details.append("missing seq 0")
                if issue.get("missing_last"):
                    details.append(f"missing seq {issue.get('expected_count', 0) - 1}")
                if issue.get("actual_count") != issue.get("expected_count"):
                    details.append(
                        f"has {issue.get('actual_count')}/{issue.get('expected_count')} invocations"
                    )
                table.add_row(
                    f"Run {run_id_str} (exp {exp_id_str})", ", ".join(details)
                )
            if len(report.run_invocation_issues) > 10:
                table.add_row(
                    "",
                    f"... and {len(report.run_invocation_issues) - 10} more run issues",
                )

        # Embedding issues
        if report.embedding_issues:
            shown = 0
            for issue in report.embedding_issues[:10]:  # Show first 10
                exp_id_str = str(issue["experiment_id"])[:8]
                inv_id_str = str(issue["invocation_id"])[:8]
                details = []
                if issue.get("embedding_count", 0) == 0:
                    details.append(
                        f"missing embedding for {issue.get('embedding_model')}"
                    )
                elif issue.get("embedding_count", 0) > 1:
                    details.append(
                        f"{issue.get('embedding_count')} duplicates for {issue.get('embedding_model')}"
                    )
                if issue.get("has_null_vector"):
                    details.append(f"null vector for {issue.get('embedding_model')}")
                table.add_row(
                    f"Invocation {inv_id_str} (exp {exp_id_str})", ", ".join(details)
                )
                shown += 1
            if len(report.embedding_issues) > 10:
                table.add_row(
                    "",
                    f"... and {len(report.embedding_issues) - 10} more embedding issues",
                )

        # Persistence diagram issues
        if report.pd_issues:
            for issue in report.pd_issues[:10]:  # Show first 10
                exp_id_str = str(issue["experiment_id"])[:8]
                run_id_str = str(issue["run_id"])[:8]
                issue_type = issue.get("issue_type", "unknown")
                model = issue.get("embedding_model", "unknown")
                if issue_type == "missing":
                    details = f"missing PD for {model}"
                elif issue_type == "duplicate":
                    details = f"{issue.get('pd_count', 0)} duplicate PDs for {model}"
                elif issue_type == "invalid_model":
                    details = f"PD with invalid model {model}"
                else:
                    details = f"{issue_type} issue for {model}"
                if issue.get("has_null_data"):
                    details += " (null data)"
                table.add_row(f"Run {run_id_str} (exp {exp_id_str})", details)
            if len(report.pd_issues) > 10:
                table.add_row(
                    "", f"... and {len(report.pd_issues) - 10} more PD issues"
                )

        # Sequence gap issues
        if report.sequence_gap_issues:
            for issue in report.sequence_gap_issues[:10]:  # Show first 10
                exp_id_str = str(issue["experiment_id"])[:8]
                run_id_str = str(issue["run_id"])[:8]
                gaps = issue.get("gaps", [])
                if len(gaps) <= 5:
                    gap_str = ", ".join(str(g) for g in gaps)
                else:
                    gap_str = f"{', '.join(str(g) for g in gaps[:3])}... ({len(gaps)} gaps total)"
                table.add_row(
                    f"Run {run_id_str} (exp {exp_id_str})",
                    f"missing sequences: {gap_str}",
                )
            if len(report.sequence_gap_issues) > 10:
                table.add_row(
                    "",
                    f"... and {len(report.sequence_gap_issues) - 10} more sequence gap issues",
                )

        # Orphaned records
        if report.orphaned_records["embeddings"]:
            count = len(report.orphaned_records["embeddings"])
            sample = report.orphaned_records["embeddings"][:3]
            sample_str = ", ".join(str(r["embedding_id"])[:8] for r in sample)
            if count > 3:
                sample_str += f"... ({count} total)"
            table.add_row("Orphaned Embeddings", sample_str)

        if report.orphaned_records["persistence_diagrams"]:
            count = len(report.orphaned_records["persistence_diagrams"])
            sample = report.orphaned_records["persistence_diagrams"][:3]
            sample_str = ", ".join(str(r["pd_id"])[:8] for r in sample)
            if count > 3:
                sample_str += f"... ({count} total)"
            table.add_row("Orphaned PDs", sample_str)

        # Global orphans
        global_orphans = report.orphaned_records["global_orphans"]
        if any(global_orphans.values()):
            details = []
            if global_orphans["invocations"]:
                details.append(f"{global_orphans['invocations']} invocations")
            if global_orphans["embeddings"]:
                details.append(f"{global_orphans['embeddings']} embeddings")
            if global_orphans["persistence_diagrams"]:
                details.append(f"{global_orphans['persistence_diagrams']} PDs")
            table.add_row("Global Orphans", ", ".join(details))

        # Duration
        table.add_row("Duration", f"{stats['duration_seconds']:.2f} seconds")

        console.print(table)

        if not fix and report.has_issues():
            console.print("\n[yellow]Use --fix flag to repair issues[/yellow]")

    return 1 if report.has_issues() else 0
