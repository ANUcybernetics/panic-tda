from dataclasses import field
from datetime import datetime
from typing import List, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class Network:
    models: List[str] = field(default_factory=list)


class Invocation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    model: str
    input: Union[str, bytes]
    output: Union[str, bytes]
    seed: int
    run_id: int
    network: Network = Field(default_factory=Network)
    sequence_number: int = 0

    # Helper method to detect content type
    def type(self, content: Union[str, bytes]) -> str:
        """Returns 'text' if content is a string, 'image' if content is bytes."""
        return "text" if isinstance(content, str) else "image"


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
