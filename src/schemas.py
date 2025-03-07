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
