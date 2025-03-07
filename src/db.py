import io
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import numpy as np
from PIL import Image
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine, select

from src.schemas import ContentType, Network, Run
from src.schemas import Embedding as SchemaEmbedding
from src.schemas import Invocation as SchemaInvocation

# Default database path
DB_PATH = Path("trajectory_tracer.db")

# Database Engine
_engine = None

def get_engine(db_path: Path = DB_PATH):
    """Get a SQLAlchemy engine, creating one if needed."""
    global _engine
    if _engine is None or db_path != DB_PATH:
        _engine = create_engine(f"sqlite:///{db_path}")
    return _engine

def get_in_memory_engine():
    """Get a SQLAlchemy engine for an in-memory database."""
    # Create an in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")

    # Create all tables in the in-memory database
    SQLModel.metadata.create_all(engine)

    return engine

# SQLModel models
class DBInvocation(SQLModel, table=True):
    __tablename__ = "invocations"

    id: str = Field(primary_key=True)
    timestamp: str
    model: str
    input_type: str
    input_text: Optional[str] = None
    input_image: Optional[bytes] = None
    output_type: str
    output_text: Optional[str] = None
    output_image: Optional[bytes] = None
    seed: int
    run_id: int
    network: str  # Stored as JSON
    sequence_number: int = 0

    embeddings: List["DBEmbedding"] = Relationship(back_populates="invocation")


class DBEmbedding(SQLModel, table=True):
    __tablename__ = "embeddings"

    id: Optional[int] = Field(default=None, primary_key=True)
    invocation_id: str = Field(foreign_key="invocations.id")
    embedding_model: str
    vector: bytes

    invocation: DBInvocation = Relationship(back_populates="embeddings")


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize the database tables."""
    global DB_PATH
    DB_PATH = db_path

    # Create tables
    engine = get_engine(db_path)
    SQLModel.metadata.create_all(engine)


def _image_to_binary(image: Image.Image) -> bytes:
    """Convert a PIL Image to WebP binary data."""
    buffer = io.BytesIO()
    # Save as WebP for better compression with good quality
    image.save(buffer, format="WEBP", quality=90)
    return buffer.getvalue()


def _binary_to_image(binary_data: bytes) -> Image.Image:
    """Convert WebP binary data back to a PIL Image."""
    return Image.open(io.BytesIO(binary_data))


def save_invocation(invocation: SchemaInvocation) -> None:
    """Save an Invocation object to the database."""
    import json

    # Prepare data for image input/output if needed
    input_text = None
    input_image = None
    if invocation.input_type == ContentType.TEXT:
        input_text = invocation.input
    else:
        input_image = _image_to_binary(invocation.input)

    output_text = None
    output_image = None
    if invocation.output_type == ContentType.TEXT:
        output_text = invocation.output
    else:
        output_image = _image_to_binary(invocation.output)

    # Create DB model from schema
    db_invocation = DBInvocation(
        id=str(invocation.id),
        timestamp=invocation.timestamp.isoformat(),
        model=invocation.model,
        input_type=invocation.input_type.value,
        input_text=input_text,
        input_image=input_image,
        output_type=invocation.output_type.value,
        output_text=output_text,
        output_image=output_image,
        seed=invocation.seed,
        run_id=invocation.run_id,
        network=json.dumps({"models": invocation.network.models}),
        sequence_number=invocation.sequence_number
    )

    # Save to database
    engine = get_engine()
    with Session(engine) as session:
        session.add(db_invocation)
        session.commit()


def get_invocation(invocation_id: UUID) -> Optional[SchemaInvocation]:
    """Retrieve an Invocation by its ID."""
    import json

    engine = get_engine()
    with Session(engine) as session:
        statement = select(DBInvocation).where(DBInvocation.id == str(invocation_id))
        db_invocation = session.exec(statement).first()

        if not db_invocation:
            return None

        # Parse the input/output based on type
        if db_invocation.input_type == ContentType.TEXT.value:
            input_content = db_invocation.input_text
        else:
            input_content = _binary_to_image(db_invocation.input_image)

        if db_invocation.output_type == ContentType.TEXT.value:
            output_content = db_invocation.output_text
        else:
            output_content = _binary_to_image(db_invocation.output_image)

        # Parse network from JSON
        network_data = json.loads(db_invocation.network)

        return SchemaInvocation(
            id=UUID(db_invocation.id),
            timestamp=datetime.fromisoformat(db_invocation.timestamp),
            model=db_invocation.model,
            input=input_content,
            output=output_content,
            seed=db_invocation.seed,
            run_id=db_invocation.run_id,
            network=Network(models=network_data.get('models', [])),
            sequence_number=db_invocation.sequence_number
        )


def save_embedding(embedding: SchemaEmbedding) -> None:
    """Save an Embedding object to the database."""
    # Convert vector to binary
    vector_blob = np.array(embedding.vector, dtype=np.float32).tobytes()

    # Create DB model
    db_embedding = DBEmbedding(
        invocation_id=str(embedding.invocation_id),
        embedding_model=embedding.embedding_model,
        vector=vector_blob
    )

    # Save to database
    engine = get_engine()
    with Session(engine) as session:
        session.add(db_embedding)
        session.commit()


def get_embeddings_for_invocation(invocation_id: UUID) -> List[SchemaEmbedding]:
    """Retrieve all Embeddings for a specific invocation."""
    engine = get_engine()
    with Session(engine) as session:
        statement = select(DBEmbedding).where(DBEmbedding.invocation_id == str(invocation_id))
        db_embeddings = session.exec(statement).all()

        embeddings = []
        for db_embedding in db_embeddings:
            # Convert binary blob back to list of floats
            vector = np.frombuffer(db_embedding.vector, dtype=np.float32).tolist()

            embeddings.append(SchemaEmbedding(
                invocation_id=UUID(db_embedding.invocation_id),
                embedding_model=db_embedding.embedding_model,
                vector=vector
            ))

        return embeddings


def save_run(run: Run) -> None:
    """Save a Run object and all its invocations."""
    for invocation in run.invocations:
        save_invocation(invocation)


def get_run(run_id: int) -> Optional[Run]:
    """Retrieve a Run by its ID with all associated invocations."""
    engine = get_engine()
    with Session(engine) as session:
        statement = select(DBInvocation).where(DBInvocation.run_id == run_id).order_by(DBInvocation.sequence_number)
        db_invocations = session.exec(statement).all()

        if not db_invocations:
            return None

        invocations = []
        for db_invoc in db_invocations:
            invocation = get_invocation(UUID(db_invoc.id))
            if invocation:
                invocations.append(invocation)

        return Run(invocations=invocations)


# For testing
def setup_test_db(suffix=""):
    """Set up a test database and return its path."""
    import os
    import tempfile

    # Create unique test database path
    db_path = Path(tempfile.gettempdir()) / f"test_db_{os.getpid()}{suffix}.sqlite"

    # Delete if exists
    if db_path.exists():
        os.unlink(db_path)

    # Initialize the DB with test path
    init_db(db_path)
    return db_path


def cleanup_test_db(db_path=None):
    """Clean up the test database."""
    import os

    # Get the test database path
    if db_path is None:
        db_path = Path(DB_PATH)

    # Delete if exists
    if db_path.exists() and "test_db_" in str(db_path):  # Safety check
        os.unlink(db_path)


# Initialize the database on module import
init_db()
