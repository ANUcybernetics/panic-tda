import io
from datetime import datetime
from enum import Enum
from typing import List, Optional, Union
from uuid import UUID, uuid4

import numpy as np
from PIL import Image
from pydantic import field_validator
from sqlalchemy import Column, LargeBinary, TypeDecorator
from sqlmodel import JSON, Field, Relationship, SQLModel


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


class InvocationType(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class Invocation(SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime = Field(default_factory=datetime.now)
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
    def model(self) -> str:
        """Get the model type (class name)"""
        return self.model

    @model.setter
    def model(self, value: str):
        """Store the class name"""
        if isinstance(value, str):
            self.model = value
        else:
            # If a class was passed, store its name
            self.model = value.__name__

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


class Run(SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    network: List[str] = Field(default=[], sa_type=JSON)
    seed: int
    length: int
    initial_prompt: str
    invocations: List[Invocation] = Relationship(
        back_populates="run",
        sa_relationship_kwargs={"order_by": "Invocation.sequence_number"}
    )


class Embedding(SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    invocation_id: UUID = Field(foreign_key="invocation.id")
    embedding_model: str
    vector: np.ndarray = Field(default=None, sa_column=Column(NumpyArrayType))

    # Relationship attribute
    invocation: Invocation = Relationship(back_populates="embeddings")

    @property
    def dimension(self) -> int:
        if self.vector is None:
            return 0
        return len(self.vector)


class ExperimentConfig(SQLModel):
    model_config = {"arbitrary_types_allowed": True}

    """Configuration for a trajectory tracer experiment."""
    networks: List[List[str]] = Field(..., description="List of networks (each network is a list of model names)")
    seeds: List[int] = Field(..., description="List of random seeds to use")
    prompts: List[str] = Field(..., description="List of initial text prompts")
    embedders: List[str] = Field(..., description="List of embedding model names")
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
