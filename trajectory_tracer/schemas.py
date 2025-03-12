import io
from datetime import datetime
from enum import Enum
from typing import List, Optional, Union
from uuid import UUID

import numpy as np
from PIL import Image
from pydantic import field_validator
from sqlalchemy import Column, LargeBinary, TypeDecorator
from sqlmodel import JSON, Field, Relationship, SQLModel
from uuid_v7.base import uuid7

## numpy storage helper classes

class NumpyArrayType(TypeDecorator):
    """SQLAlchemy type for storing numpy arrays as binary data."""
    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32).tobytes()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return np.frombuffer(value, dtype=np.float32)


class NumpyArrayListType(TypeDecorator):
    """SQLAlchemy type for storing a list of numpy arrays as binary data."""
    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None

        buffer = io.BytesIO()
        # Save the number of arrays
        num_arrays = len(value)
        buffer.write(np.array([num_arrays], dtype=np.int32).tobytes())

        # For each array, save its shape, dtype, and data
        for arr in value:
            arr_np = np.asarray(arr, dtype=np.float32)  # Ensure it's a numpy array with float32 dtype
            shape = np.array(arr_np.shape, dtype=np.int32)

            # Save shape dimensions
            shape_length = len(shape)
            buffer.write(np.array([shape_length], dtype=np.int32).tobytes())
            buffer.write(shape.tobytes())

            # Save the array data
            buffer.write(arr_np.tobytes())

        return buffer.getvalue()

    def process_result_value(self, value, dialect):
        if value is None:
            return []

        buffer = io.BytesIO(value)
        # Read the number of arrays
        num_arrays_bytes = buffer.read(4)  # int32 is 4 bytes
        num_arrays = np.frombuffer(num_arrays_bytes, dtype=np.int32)[0]

        arrays = []
        for _ in range(num_arrays):
            # Read shape information
            shape_length_bytes = buffer.read(4)  # int32 is 4 bytes
            shape_length = np.frombuffer(shape_length_bytes, dtype=np.int32)[0]

            shape_bytes = buffer.read(4 * shape_length)  # Each dimension is an int32 (4 bytes)
            shape = tuple(np.frombuffer(shape_bytes, dtype=np.int32))

            # Calculate number of elements and bytes needed
            n_elements = np.prod(shape)
            n_bytes = n_elements * 4  # float32 is 4 bytes per element

            # Read array data
            array_bytes = buffer.read(n_bytes)
            array_data = np.frombuffer(array_bytes, dtype=np.float32)
            arrays.append(array_data.reshape(shape))

        return arrays


## main DB classes

class InvocationType(str, Enum):
    TEXT = "text"
    IMAGE = "image"


# NOTE: output can't be passed to the constructor, has to be set afterwards (otherwise the setter won't work)
class Invocation(SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    model: str  # Store the model class name
    type: InvocationType
    seed: int
    run_id: UUID = Field(foreign_key="run.id")
    sequence_number: int = 0
    input_invocation_id: Optional[UUID] = Field(default=None, foreign_key="invocation.id")
    output_text: Optional[str] = None
    output_image_data: Optional[bytes] = None

    # Relationship attributes
    run: "Run" = Relationship(back_populates="invocations")
    embeddings: List["Embedding"] = Relationship(back_populates="invocation")
    input_invocation: Optional["Invocation"] = Relationship(
        sa_relationship_kwargs={"remote_side": "Invocation.id"}
    )

    @property
    def output(self) -> Union[str, Image.Image, None]:
        if self.type == InvocationType.TEXT:
            return self.output_text
        elif self.type == InvocationType.IMAGE and self.output_image_data:
            return Image.open(io.BytesIO(self.output_image_data))
        return None

    @output.setter
    def output(self, value: Union[str, Image.Image, None]) -> None:
        if value is None:
            self.output_text = None
            self.output_image_data = None
        elif isinstance(value, str):
            self.output_text = value
            self.output_image_data = None
        elif isinstance(value, Image.Image):
            self.output_text = None
            buffer = io.BytesIO()
            value.save(buffer, format="WEBP")
            self.output_image_data = buffer.getvalue()
        else:
            raise TypeError(f"Expected str, Image, or None, got {type(value)}")

    @property
    def duration(self) -> float:
        """Return the duration of the embedding computation in seconds."""
        if self.started_at is None or self.completed_at is None:
            return 0.0
        delta = self.completed_at - self.started_at
        return delta.total_seconds()


class Run(SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    network: List[str] = Field(default=[], sa_type=JSON)
    seed: int
    length: int
    initial_prompt: str
    invocations: List[Invocation] = Relationship(
        back_populates="run",
        sa_relationship_kwargs={"order_by": "Invocation.sequence_number"}
    )
    persistence_diagrams: List["PersistenceDiagram"] = Relationship(back_populates="run")


class Embedding(SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    invocation_id: UUID = Field(foreign_key="invocation.id")
    embedding_model: str  # Store the embedding model class name
    vector: np.ndarray = Field(default=None, sa_column=Column(NumpyArrayType))

    # Relationship attribute
    invocation: Invocation = Relationship(back_populates="embeddings")

    @property
    def dimension(self) -> int:
        if self.vector is None:
            return 0
        return len(self.vector)

    @property
    def duration(self) -> float:
        """Return the duration of the embedding computation in seconds."""
        if self.started_at is None or self.completed_at is None:
            return 0.0
        delta = self.completed_at - self.started_at
        return delta.total_seconds()


class PersistenceDiagram(SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    generators: List[np.ndarray] = Field(
        default=[],
        sa_column=Column(NumpyArrayListType)
    )

    run_id: UUID = Field(foreign_key="run.id")
    run: Run = Relationship(back_populates="persistence_diagrams")

    def get_generators_as_arrays(self) -> List[np.ndarray]:
        """Return generators as numpy arrays."""
        return self.generators  # Already numpy arrays

    @property
    def duration(self) -> float:
        """Return the duration of the embedding computation in seconds."""
        if self.started_at is None or self.completed_at is None:
            return 0.0
        delta = self.completed_at - self.started_at
        return delta.total_seconds()


class ExperimentConfig(SQLModel):
    model_config = {"arbitrary_types_allowed": True}

    """Configuration for a trajectory tracer experiment."""
    networks: List[List[str]] = Field(..., description="List of networks (each network is a list of model names)")
    seeds: List[int] = Field(..., description="List of random seeds to use")
    prompts: List[str] = Field(..., description="List of initial text prompts")
    embedders: List[str] = Field(..., description="List of embedding model class names")
    run_length: int = Field(..., description="Number of invocations in each run")

    @field_validator('networks', 'seeds', 'prompts', 'embedders', check_fields=False)
    @classmethod
    def check_non_empty_lists(cls, value):
        if not value:
            raise ValueError("List cannot be empty")
        return value

    @field_validator('run_length', check_fields=False)
    @classmethod
    def check_positive_run_length(cls, value):
        if value <= 0:
            raise ValueError("Run length must be greater than 0")
        return value

    def validate_equal_lengths(self):
        """Validate that all parameter lists have the same length."""
        lengths = [
            len(self.networks),
            len(self.seeds),
            len(self.prompts),
            len(self.embedders)
        ]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All parameter lists must have the same length. "
                f"Got: networks={len(self.networks)}, seeds={len(self.seeds)}, "
                f"prompts={len(self.prompts)}, embedders={len(self.embedders)}"
            )
        return True
