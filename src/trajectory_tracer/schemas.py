import io
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
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


class PersistenceDiagramResultType(TypeDecorator):
    """
    SQLAlchemy type for storing persistence diagram results from TDA computations.

    This type is specifically designed to store the structure returned by persistence
    diagram calculations, which consists of:

    - Lists of numpy int64 arrays with various shapes (including empty arrays)
    - Dictionaries with metadata like 'entropy' as float32 arrays

    This specialized type preserves both data types and array shapes during serialization.
    """

    impl = JSON
    cache_ok = True

    # Type to represent the complex structure returned by persistence diagram calculations
    # Typically includes arrays, lists of arrays, and dictionaries with arrays
    DiagramResultType = Dict[str, Union[List[np.ndarray], np.ndarray]]

    def process_bind_param(
        self, value: Optional[DiagramResultType], dialect
    ) -> Optional[Dict]:
        """
        Convert a persistence diagram result structure to JSON-serializable format.

        Args:
            value: The persistence diagram data structure to convert, or None
            dialect: SQLAlchemy dialect (unused)

        Returns:
            JSON-serializable dictionary, or None if input is None
        """
        if value is None:
            return None

        result = {}

        # Process diagrams (dgms key) - list of arrays
        if "dgms" in value:
            result["dgms"] = []
            for arr in value["dgms"]:
                if isinstance(arr, np.ndarray):
                    result["dgms"].append({
                        "_type": "ndarray",
                        "shape": arr.shape,
                        "dtype": str(arr.dtype),
                        "data": arr.tolist()
                    })
                else:
                    result["dgms"].append(arr)

        # Process generators (gens key) - complex nested structure
        if "gens" in value:
            gens_serializable = []
            for gen_list in value["gens"]:
                if isinstance(gen_list, list):
                    # Each generator is a list of arrays
                    gen_list_serializable = []
                    for g in gen_list:
                        if isinstance(g, np.ndarray):
                            gen_list_serializable.append({
                                "_type": "ndarray",
                                "shape": g.shape,
                                "dtype": str(g.dtype),
                                "data": g.tolist()
                            })
                        else:
                            gen_list_serializable.append(g)
                    gens_serializable.append(gen_list_serializable)
                elif isinstance(gen_list, np.ndarray):
                    # Or sometimes a single array
                    gens_serializable.append({
                        "_type": "ndarray",
                        "shape": gen_list.shape,
                        "dtype": str(gen_list.dtype),
                        "data": gen_list.tolist()
                    })
                else:
                    gens_serializable.append(gen_list)
            result["gens"] = gens_serializable

        # Process entropy - usually a numpy array
        if "entropy" in value:
            if isinstance(value["entropy"], np.ndarray):
                result["entropy"] = {
                    "_type": "ndarray",
                    "shape": value["entropy"].shape,
                    "dtype": str(value["entropy"].dtype),
                    "data": value["entropy"].tolist()
                }
            else:
                result["entropy"] = value["entropy"]

        # Handle any other keys that might be present
        for key, val in value.items():
            if key not in result:
                if isinstance(val, np.ndarray):
                    result[key] = {
                        "_type": "ndarray",
                        "shape": val.shape,
                        "dtype": str(val.dtype),
                        "data": val.tolist()
                    }
                elif isinstance(val, list):
                    # Handle lists of arrays
                    serialized_list = []
                    for item in val:
                        if isinstance(item, np.ndarray):
                            serialized_list.append({
                                "_type": "ndarray",
                                "shape": item.shape,
                                "dtype": str(item.dtype),
                                "data": item.tolist()
                            })
                        else:
                            serialized_list.append(item)
                    result[key] = serialized_list
                else:
                    result[key] = val

        return result

    def process_result_value(
        self, value: Optional[Dict], dialect
    ) -> Optional[DiagramResultType]:
        """
        Convert stored JSON data back to a persistence diagram result structure.

        Args:
            value: JSON dictionary to convert, or None
            dialect: SQLAlchemy dialect (unused)

        Returns:
            The restored persistence diagram data structure, or None if input is None
        """
        if value is None:
            return None

        result = {}

        def restore_array(arr_dict):
            """Helper to restore a numpy array from its serialized form"""
            if isinstance(arr_dict, dict) and arr_dict.get("_type") == "ndarray":
                # Get the shape and dtype
                shape = tuple(arr_dict["shape"])
                dtype_str = arr_dict["dtype"]

                # Determine the right NumPy dtype
                if 'int' in dtype_str:
                    dtype = np.int64
                elif 'float' in dtype_str:
                    dtype = np.float32
                else:
                    dtype = np.dtype(dtype_str)

                # Handle empty arrays correctly
                if shape == (0,) or 0 in shape:
                    # Create empty array with the right shape
                    return np.zeros(shape, dtype=dtype)[:0].reshape(shape)
                else:
                    return np.array(arr_dict["data"], dtype=dtype)
            return arr_dict

        # Restore diagrams (dgms key) - convert lists back to numpy arrays
        if "dgms" in value:
            result["dgms"] = [restore_array(arr) for arr in value["dgms"]]

        # Restore generators (gens key)
        if "gens" in value:
            gens_arrays = []
            for gen_list in value["gens"]:
                if isinstance(gen_list, list):
                    # Each generator is a list of arrays
                    gen_arrays = [restore_array(g) for g in gen_list]
                    gens_arrays.append(gen_arrays)
                else:
                    # Or sometimes a single value
                    gens_arrays.append(restore_array(gen_list))
            result["gens"] = gens_arrays

        # Restore entropy
        if "entropy" in value:
            result["entropy"] = restore_array(value["entropy"])

        # Handle any other keys
        for key, val in value.items():
            if key not in result:
                if isinstance(val, dict) and val.get("_type") == "ndarray":
                    result[key] = restore_array(val)
                elif isinstance(val, list):
                    result[key] = [restore_array(item) for item in val]
                else:
                    result[key] = val

        return result


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
    def embeddings(self) -> Dict[str, List["Embedding"]]:
        """
        Get all embeddings for all invocations in this run, organized by model.

        Returns:
            A dictionary mapping embedding model names to lists of Embedding objects
        """
        result: Dict[str, List["Embedding"]] = {}
        for invocation in self.invocations:
            for embedding in invocation.embeddings:
                model_name = embedding.embedding_model
                if model_name not in result:
                    result[model_name] = []
                result[model_name].append(embedding)
        return result

    def missing_embeddings(self, model_name: str) -> List[Invocation]:
        """
        Get all invocations that either don't have an embedding for a specific model
        or have an embedding with a null vector.

        Args:
            model_name: Name of the embedding model to check

        Returns:
            List of invocations missing valid embeddings for the specified model
        """
        missing = []
        for invocation in self.invocations:
            embedding = invocation.embedding(model_name)
            if embedding is None or embedding.vector is None:
                missing.append(invocation)
        return missing

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


# Helper function to format time duration
def format_time_duration(seconds):
    return f"{int(seconds // 3600):02d}h {int((seconds % 3600) // 60):02d}m {int(seconds % 60):02d}s"


# Helper function to calculate time strings for stages
def get_time_string(percent_complete, start_time, end_time):
    if percent_complete >= 100.0:
        elapsed_seconds = (end_time - start_time).total_seconds()
        return f" (completed in {format_time_duration(elapsed_seconds)})"
    else:
        # Estimate time to completion
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        if percent_complete > 0:
            total_estimated_seconds = elapsed_seconds / (percent_complete / 100.0)
            remaining_seconds = total_estimated_seconds - elapsed_seconds
            return f" (est. {format_time_duration(remaining_seconds)} remaining)"
    return ""

class PersistenceDiagram(SQLModel, table=True):
    """
    Represents the topological features of a run's trajectory through embedding space.

    This class stores the results of persistent homology computations performed on
    the sequence of embeddings from a run. The persistent homology diagram contains
    information about topological features (connected components, loops, voids) detected
    at different scales in the trajectory data.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    diagram_data: Optional[Dict] = Field(
        default=None, sa_column=Column(PersistenceDiagramResultType)
    )

    run_id: UUID = Field(foreign_key="run.id", index=True)
    embedding_model: str = Field(..., description="Embedding model class name")
    run: Run = Relationship(back_populates="persistence_diagrams")

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

    def get_generators_as_arrays(self) -> List[np.ndarray]:
        """
        Get generators from the persistence diagram as a list of numpy arrays.

        Extracts and converts the persistence generators from the JSON-serialized
        format back to numpy arrays for computation and analysis.

        Returns:
            List of numpy arrays containing the generators
        """
        if not self.diagram_data or 'gens' not in self.diagram_data:
            return []

        result = []
        for gens in self.diagram_data['gens']:
            if isinstance(gens, list) and gens:
                for g in gens:
                    if isinstance(g, np.ndarray) and g.size > 0:
                        result.append(g)
                    elif isinstance(g, list) and len(g) > 0:
                        result.append(np.array(g))
        return result


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
        from trajectory_tracer.embeddings import list_models as list_embedding_models
        from trajectory_tracer.genai_models import list_models as list_genai_models

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

    def print_status(self) -> None:
        """
        Get a status report of the experiment's progress.

        Reports:
        - Invocation progress: Percentage of maximum sequence number out of max_length
        - Embedding progress: Percentage of completed embeddings broken down by model
        - Persistence diagram progress: Percentage of runs with completed diagrams

        Returns:
            A multi-line string with the formatted status report
        """

        # Initialize counters
        total_runs = len(self.runs)
        if total_runs == 0:
            return "No runs found in experiment"

        # Invocation progress
        min_sequence = float('inf')
        for run in self.runs:
            # Skip runs that stopped due to duplicate outputs
            if isinstance(run.stop_reason, tuple) and run.stop_reason[0] == "duplicate":
                continue

            # Find max sequence number for this run
            max_seq = max([inv.sequence_number for inv in run.invocations], default=0)
            min_sequence = min(min_sequence, max_seq)

        # If all runs were duplicates, set to 0
        if min_sequence == float('inf'):
            min_sequence = 0

        invocation_percent = ((min_sequence + 1) / self.max_length) * 100

        # Embedding progress - overall and per model
        total_invocations = sum(len(run.invocations) for run in self.runs)

        # Overall embedding statistics
        expected_embeddings_total = total_invocations * len(self.embedding_models)
        actual_embeddings_total = sum(len(run.embeddings.get(model, [])) for run in self.runs
                            for model in self.embedding_models)
        embedding_percent_total = (actual_embeddings_total / expected_embeddings_total) * 100 if expected_embeddings_total > 0 else 0

        # Per-model embedding statistics
        model_stats = {}
        for model in self.embedding_models:
            expected_for_model = total_invocations
            actual_for_model = sum(len(run.embeddings.get(model, [])) for run in self.runs)
            percent_for_model = (actual_for_model / expected_for_model) * 100 if expected_for_model > 0 else 0
            model_stats[model] = (actual_for_model, expected_for_model, percent_for_model)

        # Persistence diagram progress
        runs_with_diagrams = sum(1 for run in self.runs if len(run.persistence_diagrams) > 0)
        diagram_percent = (runs_with_diagrams / total_runs) * 100

        # Calculate invocation timing information
        invocation_start_time = min([min([inv.started_at for inv in run.invocations if inv.started_at is not None], default=datetime.now())
                                     for run in self.runs], default=datetime.now())
        invocation_end_time = max([max([inv.completed_at for inv in run.invocations if inv.completed_at is not None], default=datetime.now())
                                   for run in self.runs], default=datetime.now())
        invocation_time_str = get_time_string(invocation_percent, invocation_start_time, invocation_end_time)

        # Calculate embedding timing information
        embedding_start_time = min([min([emb.started_at for model in self.embedding_models
                                        for emb in run.embeddings.get(model, []) if emb.started_at is not None], default=datetime.now())
                                   for run in self.runs], default=datetime.now())
        embedding_end_time = max([max([emb.completed_at for model in self.embedding_models
                                      for emb in run.embeddings.get(model, []) if emb.completed_at is not None], default=datetime.now())
                                 for run in self.runs], default=datetime.now())
        embedding_time_str = get_time_string(embedding_percent_total, embedding_start_time, embedding_end_time)

        # Calculate persistence diagram timing information
        diagram_start_time = min([min([pd.started_at for pd in run.persistence_diagrams if pd.started_at is not None], default=datetime.now())
                                  for run in self.runs], default=datetime.now())
        diagram_end_time = max([max([pd.completed_at for pd in run.persistence_diagrams if pd.completed_at is not None], default=datetime.now())
                                for run in self.runs], default=datetime.now())
        diagram_time_str = get_time_string(diagram_percent, diagram_start_time, diagram_end_time)

        status_report = (
            f"Experiment Status:\n"
            f"  Invocation Progress: {invocation_percent:.1f}% ({min_sequence + 1}/{self.max_length}){invocation_time_str}\n"
            f"  Embedding Progress (Overall): {embedding_percent_total:.1f}% ({actual_embeddings_total}/{expected_embeddings_total}){embedding_time_str}\n"
        )

        # Add per-model embedding statistics
        for model, (actual, expected, percent) in model_stats.items():
            status_report += f"    - {model}: {percent:.1f}% ({actual}/{expected})\n"
            status_report += (
                f"  Persistence Diagrams: {diagram_percent:.1f}% ({runs_with_diagrams}/{total_runs}){diagram_time_str}"
                f"\n  Elapsed Time: {format_time_duration((self.completed_at - self.started_at).total_seconds() if self.completed_at else (datetime.now() - self.started_at).total_seconds())}"
            )

        print(status_report)
