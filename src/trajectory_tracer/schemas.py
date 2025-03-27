import io
from datetime import datetime
from enum import Enum
from typing import List, Optional, Union
from uuid import UUID

import numpy as np
from PIL import Image
from pydantic import model_validator
from sqlalchemy import Column, LargeBinary, TypeDecorator
from sqlmodel import JSON, Field, Relationship, SQLModel
from uuid_v7.base import uuid7

## numpy storage helper classes


class NumpyArrayType(TypeDecorator):
    """
    SQLAlchemy type for storing numpy arrays as binary data.

    This custom type allows numpy arrays to be stored efficiently in the database
    by converting them to binary data. All arrays are stored as float32 dtype
    for consistency.
    """

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(
        self, value: Optional[np.ndarray], dialect
    ) -> Optional[bytes]:
        """
        Convert a numpy array to bytes for storage.

        Args:
            value: The numpy array to convert, or None
            dialect: SQLAlchemy dialect (unused)

        Returns:
            Bytes representation of the array, or None if input is None
        """
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32).tobytes()

    def process_result_value(
        self, value: Optional[bytes], dialect
    ) -> Optional[np.ndarray]:
        """
        Convert stored bytes back to a numpy array.

        Args:
            value: Bytes to convert, or None
            dialect: SQLAlchemy dialect (unused)

        Returns:
            The restored numpy array, or None if input is None
        """
        if value is None:
            return None
        return np.frombuffer(value, dtype=np.float32)


class NumpyArrayListType(TypeDecorator):
    """
    SQLAlchemy type for storing a list of numpy arrays as binary data.

    This custom type enables efficient storage of multiple numpy arrays in a single
    database field by serializing their shapes and contents into a compact binary format.
    All arrays are stored with float32 dtype for consistency.
    """

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(
        self, value: Optional[List[np.ndarray]], dialect
    ) -> Optional[bytes]:
        """
        Convert a list of numpy arrays to bytes for storage.

        The format stores:
        1. The number of arrays (int32)
        2. For each array:
           a. The dimensionality of the shape (int32)
           b. The shape dimensions (each as int32)
           c. The array data (as float32)

        Args:
            value: List of numpy arrays to convert, or None
            dialect: SQLAlchemy dialect (unused)

        Returns:
            Bytes representation of the array list, or None if input is None
        """
        if value is None:
            return None

        buffer = io.BytesIO()
        # Save the number of arrays
        num_arrays = len(value)
        buffer.write(np.array([num_arrays], dtype=np.int32).tobytes())

        # For each array, save its shape, dtype, and data
        for arr in value:
            arr_np = np.asarray(
                arr, dtype=np.float32
            )  # Ensure it's a numpy array with float32 dtype
            shape = np.array(arr_np.shape, dtype=np.int32)

            # Save shape dimensions
            shape_length = len(shape)
            buffer.write(np.array([shape_length], dtype=np.int32).tobytes())
            buffer.write(shape.tobytes())

            # Save the array data
            buffer.write(arr_np.tobytes())

        return buffer.getvalue()

    def process_result_value(
        self, value: Optional[bytes], dialect
    ) -> Optional[List[np.ndarray]]:
        """
        Convert stored bytes back to a list of numpy arrays.

        Decodes the binary format created by process_bind_param to reconstruct
        the original list of arrays with their proper shapes.

        Args:
            value: Bytes to convert, or None
            dialect: SQLAlchemy dialect (unused)

        Returns:
            List of restored numpy arrays, or None if input is None
        """
        if value is None:
            return None

        buffer = io.BytesIO(value)
        # Read the number of arrays
        num_arrays_bytes = buffer.read(4)  # int32 is 4 bytes
        num_arrays = np.frombuffer(num_arrays_bytes, dtype=np.int32)[0]

        arrays = []
        for _ in range(num_arrays):
            # Read shape information
            shape_length_bytes = buffer.read(4)  # int32 is 4 bytes
            shape_length = np.frombuffer(shape_length_bytes, dtype=np.int32)[0]

            shape_bytes = buffer.read(
                4 * shape_length
            )  # Each dimension is an int32 (4 bytes)
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
    """
    Enum defining the types of model invocations supported.

    TEXT: Invocation that produces text output
    IMAGE: Invocation that produces image output
    """

    TEXT = "text"
    IMAGE = "image"


# NOTE: output can't be passed to the constructor, has to be set afterwards (otherwise the setter won't work)
class Invocation(SQLModel, table=True):
    """
    Represents a single invocation of a generative AI model with inputs and outputs.

    This class stores details about a model invocation, including the model used,
    seed value, timing information, and the actual input/output data (which can
    be text or images). It maintains relationships to its parent run and any
    embedding calculations performed on its outputs.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    model: str = Field(..., description="Model class name")
    type: InvocationType
    seed: int
    run_id: UUID = Field(foreign_key="run.id", index=True)
    sequence_number: int = 0
    input_invocation_id: Optional[UUID] = Field(
        default=None, foreign_key="invocation.id", index=True
    )
    output_text: Optional[str] = None
    output_image_data: Optional[bytes] = None

    # Relationship attributes
    run: "Run" = Relationship(back_populates="invocations")
    embeddings: List["Embedding"] = Relationship(
        back_populates="invocation",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    input_invocation: Optional["Invocation"] = Relationship(
        sa_relationship_kwargs={"remote_side": "Invocation.id"}
    )

    @property
    def output(self) -> Union[str, Image.Image, None]:
        """
        Get the output of this invocation as the appropriate type.

        Returns:
            A string for text invocations, a PIL Image for image invocations,
            or None if no output is available.
        """
        if self.type == InvocationType.TEXT:
            return self.output_text
        elif self.type == InvocationType.IMAGE and self.output_image_data:
            return Image.open(io.BytesIO(self.output_image_data))
        return None

    @output.setter
    def output(self, value: Union[str, Image.Image, None]) -> None:
        """
        Set the output of this invocation, handling the appropriate type conversion.

        For string values, stores in output_text.
        For PIL Image values, converts to WEBP format and stores in output_image_data.
        For None, clears both output fields.

        Args:
            value: The output to set (string, PIL Image, or None)

        Raises:
            TypeError: If the value is not a string, PIL Image, or None
        """
        if value is None:
            self.output_text = None
            self.output_image_data = None
        elif isinstance(value, str):
            self.output_text = value
            self.output_image_data = None
        elif isinstance(value, Image.Image):
            self.output_text = None
            buffer = io.BytesIO()
            value.save(buffer, format="WEBP", lossless=True, quality=100)
            self.output_image_data = buffer.getvalue()
        else:
            raise TypeError(f"Expected str, Image, or None, got {type(value)}")

    @property
    def input(self) -> Union[str, Image.Image, None]:
        """
        Get the input to this invocation.

        For the first invocation in a run, returns the initial prompt.
        For later invocations, returns the output of the previous invocation.

        Returns:
            A string, PIL Image, or None depending on the input type
        """
        if self.sequence_number == 0:
            return self.run.initial_prompt
        elif self.input_invocation:
            return self.input_invocation.output
        else:
            return None

    @property
    def duration(self) -> float:
        """
        Calculate the duration of the invocation in seconds.

        Returns:
            Duration in seconds between started_at and completed_at timestamps,
            or 0.0 if either timestamp is missing
        """
        if self.started_at is None or self.completed_at is None:
            return 0.0
        delta = self.completed_at - self.started_at
        return delta.total_seconds()

    def embedding(self, model_name: str) -> Optional["Embedding"]:
        """
        Get the embedding for this invocation created by a specific model.

        Args:
            model_name: Name of the embedding model to retrieve

        Returns:
            The matching Embedding object, or None if no embedding exists for this model
        """
        for embedding in self.embeddings:
            if embedding.embedding_model == model_name:
                return embedding
        return None


class Run(SQLModel, table=True):
    """
    Represents a complete run of a generative AI trajectory experiment.

    A Run consists of a series of model invocations in a specific network configuration,
    starting from an initial prompt. It tracks the entire sequence of generations
    and their embeddings, allowing for trajectory analysis.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    network: List[str] = Field(default=None, sa_type=JSON)
    seed: int
    max_length: int
    initial_prompt: str
    experiment_id: Optional[UUID] = Field(default=None, foreign_key="experimentconfig.id", index=True)
    invocations: List[Invocation] = Relationship(
        back_populates="run",
        sa_relationship_kwargs={
            "order_by": "Invocation.sequence_number",
            "cascade": "all, delete-orphan",
        },
    )
    persistence_diagrams: List["PersistenceDiagram"] = Relationship(
        back_populates="run", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    experiment: Optional["ExperimentConfig"] = Relationship(back_populates="runs")

    @model_validator(mode="after")
    def validate_fields(self):
        """
        Validate that the run configuration is valid.

        Ensures:
        - network list is not empty
        - max_length is positive

        Returns:
            Self if validation passes

        Raises:
            ValueError: If validation fails
        """
        if not self.network:
            raise ValueError("Network list cannot be empty")
        if self.max_length <= 0:
            raise ValueError("Max. run length must be greater than 0")
        return self

    @property
    def embeddings(self) -> List["Embedding"]:
        """
        Get all embeddings for all invocations in this run.

        Returns:
            A flat list of all embedding objects across all invocations in the run
        """
        result = []
        for invocation in self.invocations:
            result.extend(invocation.embeddings)
        return result

    @property
    def stop_reason(self) -> Union[str, tuple]:
        """
        Determine why the run stopped.

        Analyzes the run to determine if it completed its intended length,
        stopped due to detecting duplicate outputs, or stopped for an unknown reason.

        Returns:
            "length": If the run completed its full length
            ("duplicate", loop_length): If the run was stopped because of a duplicate output, with loop_length indicating the distance between duplicates
            "unknown": If the reason can't be determined
        """
        # Check if we have all invocations up to the specified length
        if len(self.invocations) == self.max_length:
            # Make sure all invocations are complete (have outputs)
            all_complete = all(inv.output is not None for inv in self.invocations)
            if all_complete:
                return "length"

        # Check for duplicate outputs if seed is not -1
        if self.seed != -1 and len(self.invocations) > 1:
            # Track seen outputs with their sequence numbers
            seen_outputs = {}

            for invocation in sorted(
                self.invocations, key=lambda inv: inv.sequence_number
            ):
                if invocation.output is None:
                    continue

                # Convert output to a hashable representation based on type
                hashable_output = None
                if isinstance(invocation.output, str):
                    hashable_output = invocation.output
                elif isinstance(invocation.output, Image.Image):
                    # Convert image to a bytes representation for hashing
                    buffer = io.BytesIO()
                    invocation.output.save(buffer, format="JPEG", quality=30)
                    hashable_output = buffer.getvalue()
                else:
                    hashable_output = str(invocation.output)

                # Check if we've seen this output before
                if hashable_output in seen_outputs:
                    # Calculate loop length - difference in sequence numbers
                    previous_seq = seen_outputs[hashable_output]
                    loop_length = invocation.sequence_number - previous_seq
                    return ("duplicate", loop_length)

                seen_outputs[hashable_output] = invocation.sequence_number

        # If we can't determine a specific reason
        return "unknown"

    def embeddings_by_model(self, embedding_model: str) -> List["Embedding"]:
        """
        Get embeddings with a specific model name across all invocations in this run.

        Filters all embeddings in the run to return only those created by the
        specified embedding model.

        Args:
            embedding_model: Name of the embedding model to filter by

        Returns:
            List of Embedding objects matching the specified model
        """
        result = []
        for invocation in self.invocations:
            # Get embeddings for this invocation that match the model name
            matching_embeddings = [
                e for e in invocation.embeddings if e.embedding_model == embedding_model
            ]
            result.extend(matching_embeddings)
        return result


class Embedding(SQLModel, table=True):
    """
    Represents an embedded vector representation of a model invocation output.

    This class stores the embedding vector calculated from a model invocation's output
    along with metadata such as the embedding model used and timing information.
    Embeddings enable analysis of trajectories in a consistent vector space regardless
    of whether the original outputs were text or images.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    invocation_id: UUID = Field(foreign_key="invocation.id", index=True)
    embedding_model: str = Field(..., description="Embedding model class name")
    vector: np.ndarray = Field(default=None, sa_column=Column(NumpyArrayType))

    # Relationship attribute
    invocation: Invocation = Relationship(back_populates="embeddings")

    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embedding vector.

        Returns:
            The number of dimensions in the embedding vector, or 0 if the vector is None
        """
        if self.vector is None:
            return 0
        return len(self.vector)

    @property
    def duration(self) -> float:
        """
        Calculate the duration of the embedding computation in seconds.

        Returns:
            Duration in seconds between started_at and completed_at timestamps,
            or 0.0 if either timestamp is missing
        """
        if self.started_at is None or self.completed_at is None:
            return 0.0
        delta = self.completed_at - self.started_at
        return delta.total_seconds()


class PersistenceDiagram(SQLModel, table=True):
    """
    Represents the topological features of a run's trajectory through embedding space.

    This class stores the results of persistent homology computations performed on
    the sequence of embeddings from a run. The "generators" represent the birth-death
    pairs of topological features detected at different scales in the trajectory data.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    generators: Optional[List[np.ndarray]] = Field(
        default=None, sa_column=Column(NumpyArrayListType)
    )

    run_id: UUID = Field(foreign_key="run.id", index=True)
    embedding_model: str = Field(..., description="Embedding model class name")
    run: Run = Relationship(back_populates="persistence_diagrams")

    def get_generators_as_arrays(self) -> List[np.ndarray]:
        """
        Get the persistence diagram generators as a list of numpy arrays.

        Returns:
            List of numpy arrays representing the birth-death pairs of topological features
        """
        return (
            self.generators if self.generators is not None else []
        )  # Return empty list if None

    @property
    def duration(self) -> float:
        """
        Calculate the duration of the persistence diagram computation in seconds.

        Returns:
            Duration in seconds between started_at and completed_at timestamps,
            or 0.0 if either timestamp is missing
        """
        if self.started_at is None or self.completed_at is None:
            return 0.0
        delta = self.completed_at - self.started_at
        return delta.total_seconds()


class ExperimentConfig(SQLModel, table=True):
    """
    Configuration for a trajectory tracer experiment.

    This class defines all the parameters needed to run a complete experiment,
    including which model networks to use, initial prompts, random seeds, and
    embedding models. It provides validation to ensure the configuration is valid
    before an experiment begins.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    networks: List[List[str]] = Field(
        default=None, sa_type=JSON, description="List of networks (each network is a list of model names)"
    )
    seeds: List[int] = Field(default=None, sa_type=JSON, description="List of random seeds to use")
    prompts: List[str] = Field(default=None, sa_type=JSON, description="List of initial text prompts")
    embedding_models: List[str] = Field(
        default=None, sa_type=JSON, description="List of embedding model class names"
    )
    max_length: int = Field(..., description="Number of invocations in each run")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime = Field(default_factory=datetime.now)
    runs: List[Run] = Relationship(
        back_populates="experiment",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )

    @model_validator(mode="after")
    def validate_fields(self):
        """
        Validate that the experiment configuration is complete and consistent.

        Checks:
        - Networks list is not empty
        - Seeds list is not empty
        - Prompts list is not empty
        - Embedding models list is not empty
        - Maximum length is positive
        - All models in networks are valid models in genai_models.list_models()
        - All models in embedding_models are valid models in embeddings.list_models()

        Returns:
            Self if validation passes

        Raises:
            ValueError: If any validation check fails
        """
        if not self.networks:
            raise ValueError("Networks list cannot be empty")
        if not self.seeds:
            raise ValueError("Seeds list cannot be empty")
        if not self.prompts:
            raise ValueError("Prompts list cannot be empty")
        if not self.embedding_models:
            raise ValueError("embedding_models list cannot be empty")
        if self.max_length <= 0:
            raise ValueError("Run length must be greater than 0")

        # Import here to avoid circular imports
        from trajectory_tracer.genai_models import list_models as list_genai_models
        from trajectory_tracer.embeddings import list_models as list_embedding_models

        # Validate genai models
        valid_genai_models = list_genai_models()
        for network in self.networks:
            for model in network:
                if model not in valid_genai_models:
                    raise ValueError(f"Invalid generative model: {model}. Available models: {valid_genai_models}")

        # Validate embedding models
        valid_embedding_models = list_embedding_models()
        for model in self.embedding_models:
            if model not in valid_embedding_models:
                raise ValueError(f"Invalid embedding model: {model}. Available models: {valid_embedding_models}")

        return self
