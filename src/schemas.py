from datetime import datetime
from enum import Enum
from typing import List, Union
from uuid import UUID, uuid4

from PIL import Image
from pydantic import BaseModel, Field, field_validator


class ContentType(Enum):
    """Enum representing possible content types for invocation inputs/outputs."""
    TEXT = "text"
    IMAGE = "image"


class Network(BaseModel):
    models: List[str] = Field(default_factory=list)


class Invocation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    model: str
    input: Union[str, Image.Image]
    output: Union[str, Image.Image]
    seed: int
    run_id: int
    network: Network = Field(default_factory=Network)
    sequence_number: int = 0

    # Configure the model to allow arbitrary types like PIL.Image
    model_config = {
        "arbitrary_types_allowed": True
    }

    # Helper method to detect content type
    def type(self, content: Union[str, Image.Image]) -> ContentType:
        """Returns ContentType.TEXT if content is a string, ContentType.IMAGE if content is a PIL Image."""
        return ContentType.TEXT if isinstance(content, str) else ContentType.IMAGE

    @property
    def input_type(self) -> ContentType:
        """Get the type of the input content."""
        return self.type(self.input)

    @property
    def output_type(self) -> ContentType:
        """Get the type of the output content."""
        return self.type(self.output)


class Run(BaseModel):
    invocations: List[Invocation] = Field(default_factory=list)

    @field_validator('invocations')
    @classmethod
    def validate_invocations(cls, invocations: List[Invocation]) -> List[Invocation]:
        # Validate sequence_number matches position
        for i, invocation in enumerate(invocations):
            if invocation.sequence_number != i:
                raise ValueError(f"Invocation at position {i} has sequence_number {invocation.sequence_number}")

        return invocations


class Embedding(BaseModel):
    invocation_id: UUID
    embedding_model: str
    vector: List[float]

    @property
    def dimension(self) -> int:
        return len(self.vector)


class ExperimentConfig(BaseModel):
    """Configuration for a trajectory tracer experiment."""
    networks: List[List[str]] = Field(..., description="List of networks (each network is a list of model names)")
    seeds: List[int] = Field(..., description="List of random seeds to use")
    prompts: List[str] = Field(..., description="List of initial text prompts")
    embedders: List[str] = Field(..., description="List of embedding model names")
    run_length: int = Field(..., description="Number of invocations in each run")

    @field_validator('networks', 'seeds', 'prompts', 'embedders')
    @classmethod
    def check_non_empty_lists(cls, value):
        if not value:
            raise ValueError("List cannot be empty")
        return value

    @field_validator('run_length')
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
