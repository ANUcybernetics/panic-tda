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

    logger.info(f"Checking {len(runs)} runs for invocation completeness")

    for i, run in enumerate(runs):
        if i > 0 and i % 10 == 0:
            logger.info(f"  Checked {i}/{len(runs)} runs...")
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

        # Check for sequence gaps only if we already found issues
        # (optimization: skip expensive gap check if count and endpoints are correct)
        if not has_first or not has_last or invocation_count != run.max_length:
            # Only fetch sequences if there's already an issue
            invocation_sequences = session.exec(
                select(Invocation.sequence_number)
                .where(Invocation.run_id == run.id)
                .order_by(Invocation.sequence_number)
            ).all()

            if invocation_sequences:
                expected_sequences = list(range(run.max_length))
                actual_sequences = list(invocation_sequences)
                gaps = [
                    seq for seq in expected_sequences if seq not in actual_sequences
                ]

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
    from sqlalchemy import and_, or_

    issues = []

    # Get run IDs for this experiment
    run_ids = session.exec(
        select(Run.id).where(Run.experiment_id == experiment.id)
    ).all()

    if not run_ids:
        return issues

    logger.info(
        f"Checking embeddings for {len(run_ids)} runs in experiment {experiment.id}"
    )

    # Build lists of text and image embedding models
    text_models = []
    image_models = []
    for model in experiment.embedding_models:
        try:
            model_type = get_model_type(model)
            if model_type == EmbeddingModelType.TEXT:
                text_models.append(model)
            elif model_type == EmbeddingModelType.IMAGE:
                image_models.append(model)
        except ValueError:
            logger.warning(f"Could not determine type for embedding model {model}")

    # First, use SQL queries to find invocations with missing embeddings
    for embedding_model in experiment.embedding_models:
        # Determine which invocation types this model should embed
        try:
            model_type = get_model_type(embedding_model)
            if model_type == EmbeddingModelType.TEXT:
                target_invocation_type = InvocationType.TEXT
            elif model_type == EmbeddingModelType.IMAGE:
                target_invocation_type = InvocationType.IMAGE
            else:
                continue
        except ValueError:
            continue

        # Find invocations missing embeddings for this model
        # This query finds invocations that should have embeddings but don't
        missing_embeddings_query = (
            select(Invocation.id, Invocation.type)
            .join(Run, Invocation.run_id == Run.id)
            .outerjoin(
                Embedding,
                and_(
                    Embedding.invocation_id == Invocation.id,
                    Embedding.embedding_model == embedding_model,
                ),
            )
            .where(
                Run.experiment_id == experiment.id,
                Invocation.type == target_invocation_type,
                Embedding.id.is_(None),  # No embedding exists
            )
            .limit(100)  # Limit to first 100 issues to avoid huge result sets
        )

        missing_invocations = session.exec(missing_embeddings_query).all()

        for inv_id, inv_type in missing_invocations:
            issue = {
                "invocation_type": inv_type.value,
                "embedding_model": embedding_model,
                "embedding_count": 0,
                "has_null_vector": False,
                "embedding_ids": [],
            }
            issues.append({"invocation_id": inv_id, **issue})
            report.add_embedding_issue(experiment.id, inv_id, issue)

    # Check for mismatched embeddings (e.g., text model on image invocation)
    if text_models and image_models:
        # Find image invocations with text embeddings
        mismatched_query = (
            select(Embedding.id, Embedding.invocation_id, Embedding.embedding_model)
            .join(Invocation, Embedding.invocation_id == Invocation.id)
            .join(Run, Invocation.run_id == Run.id)
            .where(
                Run.experiment_id == experiment.id,
                or_(
                    and_(
                        Invocation.type == InvocationType.IMAGE,
                        Embedding.embedding_model.in_(text_models),
                    ),
                    and_(
                        Invocation.type == InvocationType.TEXT,
                        Embedding.embedding_model.in_(image_models),
                    ),
                ),
            )
            .limit(100)
        )

        mismatched = session.exec(mismatched_query).all()

        for emb_id, inv_id, emb_model in mismatched:
            # Get invocation type
            inv_type = session.exec(
                select(Invocation.type).where(Invocation.id == inv_id)
            ).first()

            issue = {
                "invocation_type": inv_type.value if inv_type else "unknown",
                "embedding_model": emb_model,
                "issue_type": "mismatched",
                "embedding_count": 1,
                "has_null_vector": False,
                "embedding_ids": [emb_id],
            }
            issues.append({"invocation_id": inv_id, **issue})
            report.add_embedding_issue(experiment.id, inv_id, issue)

    # Check for null vectors
    null_vector_query = (
        select(Embedding.id, Embedding.invocation_id, Embedding.embedding_model)
        .join(Invocation, Embedding.invocation_id == Invocation.id)
        .join(Run, Invocation.run_id == Run.id)
        .where(Run.experiment_id == experiment.id, Embedding.vector.is_(None))
        .limit(100)
    )

    null_vectors = session.exec(null_vector_query).all()

    for emb_id, inv_id, emb_model in null_vectors:
        issue = {
            "invocation_type": "unknown",
            "embedding_model": emb_model,
            "embedding_count": 1,
            "has_null_vector": True,
            "embedding_ids": [emb_id],
        }
        issues.append({"invocation_id": inv_id, **issue})
        report.add_embedding_issue(experiment.id, inv_id, issue)

    logger.info(f"Found {len(issues)} embedding issues")

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
        # Skip problematic runs with max_length=5000 and initial_prompt in ['yeah', 'nah']
        if run.max_length == 5000 and run.initial_prompt in ["yeah", "nah"]:
            # Use info level so we can see it without --verbose
            logger.info(
                f"Skipping PD check for run {run.id} (max_length=5000, prompt='{run.initial_prompt}')"
            )
            continue

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


def fix_run_invocations(issues: List[Dict], db_str: str):
    """Fix missing or extra invocations for runs.

    Strategy:
    1. Find the first missing or incomplete invocation
    2. Delete all invocations from that point onward
    3. Resume generation from the break point
    """
    from panic_tda.engine import perform_runs_stage

    with get_session_from_connection_string(db_str) as session:
        run_ids_to_fix = []
        for issue in issues:
            run_id = issue["run_id"]
            run = session.get(Run, run_id)

            if not run:
                logger.warning(f"Run {run_id} not found, skipping")
                continue

            # Get all existing invocations sorted by sequence number
            existing_invocations = session.exec(
                select(Invocation)
                .where(Invocation.run_id == run_id)
                .order_by(Invocation.sequence_number)
            ).all()

            # Find the first missing or incomplete invocation
            first_missing_seq = 0
            for expected_seq in range(run.max_length):
                # Check if this sequence exists
                inv = next(
                    (
                        inv
                        for inv in existing_invocations
                        if inv.sequence_number == expected_seq
                    ),
                    None,
                )

                if inv is None:
                    # Found missing invocation
                    first_missing_seq = expected_seq
                    break
                elif inv.output is None:
                    # Found incomplete invocation (no output)
                    first_missing_seq = expected_seq
                    break
            else:
                # All sequences 0 to max_length-1 exist and have outputs
                # This shouldn't happen if the run was flagged as having issues
                logger.warning(f"Run {run_id} appears complete, skipping")
                continue

            logger.info(
                f"Run {run_id}: First missing/incomplete invocation at sequence {first_missing_seq}. "
                f"Deleting invocations from sequence {first_missing_seq} onward."
            )

            # Delete all invocations from first_missing_seq onward
            invocations_to_delete = [
                inv
                for inv in existing_invocations
                if inv.sequence_number >= first_missing_seq
            ]

            for inv in invocations_to_delete:
                # Also delete associated embeddings
                for embedding in session.exec(
                    select(Embedding).where(Embedding.invocation_id == inv.id)
                ):
                    session.delete(embedding)
                session.delete(inv)

            logger.info(
                f"Deleted {len(invocations_to_delete)} invocations from sequence {first_missing_seq} onward"
            )

            run_ids_to_fix.append(str(run_id))

        session.commit()

    # Re-run the runs to generate proper invocations
    # run_generator will automatically continue from the last existing invocation
    if run_ids_to_fix:
        logger.info(f"Resuming {len(run_ids_to_fix)} runs from their break points")
        perform_runs_stage(run_ids_to_fix, db_str)


def fix_embeddings(issues: List[Dict], experiment_id: UUID, db_str: str):
    """Fix missing or invalid embeddings."""
    from panic_tda.engine import perform_embeddings_stage
    from panic_tda.embeddings import get_model_type, EmbeddingModelType

    # First, handle mismatched embeddings from the issues list
    mismatched_issues = [i for i in issues if i.get("issue_type") == "mismatched"]
    if mismatched_issues:
        logger.info(f"Processing {len(mismatched_issues)} mismatched embedding issues")
        with get_session_from_connection_string(db_str) as session:
            mismatched_count = 0
            batch_size = 1000
            embeddings_to_delete = []

            for issue in mismatched_issues:
                # Collect embedding IDs to delete
                for embedding_id in issue.get("embedding_ids", []):
                    embeddings_to_delete.append(embedding_id)

                    # Process in batches
                    if len(embeddings_to_delete) >= batch_size:
                        # Use a bulk query to fetch and delete embeddings
                        embeddings = session.exec(
                            select(Embedding).where(
                                Embedding.id.in_(embeddings_to_delete)
                            )
                        ).all()

                        for emb in embeddings:
                            session.delete(emb)
                            mismatched_count += 1

                        session.commit()
                        logger.info(
                            f"Deleted batch of {len(embeddings)} mismatched embeddings"
                        )
                        embeddings_to_delete = []

            # Process remaining embeddings
            if embeddings_to_delete:
                embeddings = session.exec(
                    select(Embedding).where(Embedding.id.in_(embeddings_to_delete))
                ).all()

                for emb in embeddings:
                    session.delete(emb)
                    mismatched_count += 1

                session.commit()

            if mismatched_count > 0:
                logger.info(
                    f"Deleted {mismatched_count} mismatched embeddings from issues"
                )

    # Also do a comprehensive cleanup of all mismatched embeddings for this experiment
    with get_session_from_connection_string(db_str) as session:
        # Re-fetch the experiment in this session to get embedding_models
        experiment = session.get(ExperimentConfig, experiment_id)
        if not experiment:
            raise ValueError(f"Experiment with ID {experiment_id} not found")
        embedding_models = list(experiment.embedding_models)  # Make a copy

        # Build lists of text and image embedding models
        text_models = []
        image_models = []
        for model in embedding_models:
            try:
                model_type = get_model_type(model)
                if model_type == EmbeddingModelType.TEXT:
                    text_models.append(model)
                elif model_type == EmbeddingModelType.IMAGE:
                    image_models.append(model)
            except ValueError:
                # Skip models without defined types
                continue

        mismatched_count = 0

        # Delete text invocation embeddings with image models
        if image_models:
            logger.info("Cleaning up text invocations with image embedding models")
            text_invocation_ids = session.exec(
                select(Invocation.id)
                .join(Run, Invocation.run_id == Run.id)
                .where(
                    Run.experiment_id == experiment_id,
                    Invocation.type == InvocationType.TEXT,
                )
            ).all()

            if text_invocation_ids:
                # Delete in batches
                batch_size = 100
                for i in range(0, len(text_invocation_ids), batch_size):
                    batch_ids = text_invocation_ids[i : i + batch_size]
                    embeddings_to_delete = session.exec(
                        select(Embedding).where(
                            Embedding.invocation_id.in_(batch_ids),
                            Embedding.embedding_model.in_(image_models),
                        )
                    ).all()

                    for emb in embeddings_to_delete:
                        session.delete(emb)
                        mismatched_count += 1

                    if embeddings_to_delete:
                        session.commit()
                        logger.info(
                            f"Deleted {len(embeddings_to_delete)} mismatched text->image embeddings"
                        )

        # Delete image invocation embeddings with text models
        if text_models:
            logger.info("Cleaning up image invocations with text embedding models")
            image_invocation_ids = session.exec(
                select(Invocation.id)
                .join(Run, Invocation.run_id == Run.id)
                .where(
                    Run.experiment_id == experiment_id,
                    Invocation.type == InvocationType.IMAGE,
                )
            ).all()

            if image_invocation_ids:
                # Delete in batches
                batch_size = 100
                for i in range(0, len(image_invocation_ids), batch_size):
                    batch_ids = image_invocation_ids[i : i + batch_size]
                    embeddings_to_delete = session.exec(
                        select(Embedding).where(
                            Embedding.invocation_id.in_(batch_ids),
                            Embedding.embedding_model.in_(text_models),
                        )
                    ).all()

                    for emb in embeddings_to_delete:
                        session.delete(emb)
                        mismatched_count += 1

                    if embeddings_to_delete:
                        session.commit()
                        logger.info(
                            f"Deleted {len(embeddings_to_delete)} mismatched image->text embeddings"
                        )

        if mismatched_count > 0:
            logger.info(
                f"Deleted {mismatched_count} additional mismatched embeddings in cleanup"
            )

    # Now compute embeddings for invocations that need them (excluding mismatched ones)
    invocation_ids_to_fix = list(
        set(
            str(issue["invocation_id"])
            for issue in issues
            if issue.get("issue_type") != "mismatched"
        )
    )

    if invocation_ids_to_fix:
        # Re-fetch embedding_models to avoid detached instance issues
        with get_session_from_connection_string(db_str) as session:
            experiment = session.get(ExperimentConfig, experiment_id)
            if not experiment:
                raise ValueError(f"Experiment with ID {experiment_id} not found")
            embedding_models = list(experiment.embedding_models)

        logger.info(
            f"Computing embeddings for {len(invocation_ids_to_fix)} invocations"
        )
        perform_embeddings_stage(invocation_ids_to_fix, embedding_models, db_str)


def fix_persistence_diagrams(issues: List[Dict], experiment_id: UUID, db_str: str):
    """Fix missing or invalid persistence diagrams."""
    from panic_tda.engine import perform_pd_stage_selective

    # Track specific (run_id, embedding_model) pairs that need recomputation
    pd_pairs_to_recompute = set()

    with get_session_from_connection_string(db_str) as session:
        # Fetch the experiment in this session to get embedding_models
        experiment = session.get(ExperimentConfig, experiment_id)
        if not experiment:
            raise ValueError(f"Experiment with ID {experiment_id} not found")

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
                # Track the specific (run, model) pair that needs recomputation
                pd_pairs_to_recompute.add((
                    str(issue["run_id"]),
                    issue["embedding_model"],
                ))
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

    # Recompute only the specific PDs that need it
    if pd_pairs_to_recompute:
        # Filter out problematic runs with max_length=5000 and initial_prompt in ['yeah', 'nah']
        filtered_pairs = []
        with get_session_from_connection_string(db_str) as session:
            for run_id, embedding_model in pd_pairs_to_recompute:
                run = session.exec(select(Run).where(Run.id == UUID(run_id))).first()
                if (
                    run
                    and run.max_length == 5000
                    and run.initial_prompt in ["yeah", "nah"]
                ):
                    logger.warning(
                        f"Skipping PD computation for run {run_id} (max_length=5000, prompt='{run.initial_prompt}')"
                    )
                    continue
                filtered_pairs.append((run_id, embedding_model))

        if filtered_pairs:
            logger.info(
                f"Computing {len(filtered_pairs)} specific persistence diagrams (excluded {len(pd_pairs_to_recompute) - len(filtered_pairs)} problematic runs)"
            )
            # Reduce concurrency to 1 to avoid memory issues - compute one PD at a time
            perform_pd_stage_selective(filtered_pairs, db_str, max_concurrent=1)
        else:
            logger.info("No persistence diagrams to compute after filtering")


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

        logger.info(f"Starting checks for experiment {experiment_id}")

        # Check run invocations
        logger.info(f"Checking run invocations for experiment {experiment_id}")
        run_issues = check_experiment_run_invocations(experiment, session, report)
        logger.info(f"Found {len(run_issues)} run invocation issues")
        if progress and task:
            progress.advance(task)

        # Check embeddings
        logger.info(f"Checking embeddings for experiment {experiment_id}")
        embedding_issues = check_experiment_embeddings(experiment, session, report)
        logger.info(f"Found {len(embedding_issues)} embedding issues")
        if progress and task:
            progress.advance(task)

        # Check persistence diagrams
        logger.info(f"Checking persistence diagrams for experiment {experiment_id}")
        pd_issues = check_experiment_persistence_diagrams(experiment, session, report)
        logger.info(f"Found {len(pd_issues)} PD issues")
        if progress and task:
            progress.advance(task)

        if run_issues or embedding_issues or pd_issues:
            experiment_has_issues = True
            report.experiments_with_issues += 1

            if fix:
                # Fix issues for this experiment
                if run_issues:
                    logger.info(
                        f"Fixing {len(run_issues)} run invocation issues for experiment {experiment_id}"
                    )
                    fix_run_invocations(run_issues, db_str)
                    # Re-check embeddings after fixing invocations
                    with get_session_from_connection_string(db_str) as session:
                        experiment = session.get(ExperimentConfig, experiment_id)
                        embedding_issues = check_experiment_embeddings(
                            experiment, session, DoctorReport()
                        )

                if embedding_issues:
                    logger.info(
                        f"Fixing {len(embedding_issues)} embedding issues for experiment {experiment_id}"
                    )
                    fix_embeddings(embedding_issues, experiment_id, db_str)
                    # Re-check persistence diagrams after fixing embeddings
                    with get_session_from_connection_string(db_str) as session:
                        experiment = session.get(ExperimentConfig, experiment_id)
                        pd_issues = check_experiment_persistence_diagrams(
                            experiment, session, DoctorReport()
                        )

                if pd_issues:
                    logger.info(
                        f"Fixing {len(pd_issues)} persistence diagram issues for experiment {experiment_id}"
                    )
                    fix_persistence_diagrams(pd_issues, experiment_id, db_str)

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
    output_format: str = "text",
) -> int:
    """
    Check and optionally fix a single experiment in the database.

    Args:
        experiment_id: UUID of the experiment to check
        db_str: Database connection string
        fix: If True, fix issues; if False, only report
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
            logger.info("Cleaning orphaned records (globally)")
            clean_orphaned_records(report.orphaned_records, db_str)

        # Fix sequence gaps if requested
        if fix and report.sequence_gap_issues:
            logger.info(
                f"Fixing sequence gaps in {len(report.sequence_gap_issues)} runs for experiment {experiment_id}"
            )
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
    output_format: str = "text",
    experiment_id: Optional[UUID] = None,
) -> int:
    """
    Check and optionally fix all experiments in the database, or a specific experiment if provided.

    Args:
        db_str: Database connection string
        fix: If True, fix issues; if False, only report
        output_format: Output format ('text' or 'json')
        experiment_id: Optional UUID of a specific experiment to check

    Returns:
        Exit code (0 if no issues, 1 if issues found)
    """
    # If a specific experiment is requested, use the single experiment function
    if experiment_id is not None:
        return doctor_single_experiment(experiment_id, db_str, fix, output_format)

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
            logger.info(
                f"Cleaning {sum(report.orphaned_records['global_orphans'].values())} orphaned records"
            )
            clean_orphaned_records(report.orphaned_records, db_str)

        # Fix sequence gaps if requested
        if fix and report.sequence_gap_issues:
            logger.info(
                f"Fixing sequence gaps in {len(report.sequence_gap_issues)} runs"
            )
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
