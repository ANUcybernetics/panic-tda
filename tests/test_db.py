import datetime
import uuid

import numpy as np
from PIL import Image

from src.db import (
    get_embeddings_for_invocation,
    get_invocation,
    save_embedding,
    save_invocation,
)
from src.db_testing import db_sandbox
from src.schemas import ContentType, Embedding, Invocation, Network


def test_invocation_operations():
    with db_sandbox():
        # Test saving and retrieving a text-to-text invocation
        test_id = uuid.uuid4()
        text_invocation = Invocation(
            id=test_id,
            timestamp=datetime.datetime.now(),
            model="test_model",
            input="test input text",
            output="test output text",
            seed=42,
            run_id=1,
            network=Network(models=["model1", "model2"]),
            sequence_number=0
        )

        save_invocation(text_invocation)
        retrieved = get_invocation(test_id)

        assert retrieved is not None
        assert retrieved.id == test_id
        assert retrieved.model == "test_model"
        assert retrieved.input == "test input text"
        assert retrieved.output == "test output text"
        assert retrieved.input_type == ContentType.TEXT
        assert retrieved.output_type == ContentType.TEXT
        assert retrieved.seed == 42
        assert retrieved.run_id == 1
        assert retrieved.network.models == ["model1", "model2"]

        # Test saving and retrieving an image-to-text invocation
        image_id = uuid.uuid4()
        test_image = Image.new('RGB', (100, 100), color='red')
        image_invocation = Invocation(
            id=image_id,
            timestamp=datetime.datetime.now(),
            model="image_model",
            input=test_image,
            output="image description",
            seed=123,
            run_id=2,
            sequence_number=0
        )

        save_invocation(image_invocation)
        retrieved_image_inv = get_invocation(image_id)

        assert retrieved_image_inv is not None
        assert retrieved_image_inv.input_type == ContentType.IMAGE
        assert retrieved_image_inv.output_type == ContentType.TEXT
        assert isinstance(retrieved_image_inv.input, Image.Image)
        assert retrieved_image_inv.output == "image description"

        # Test embedding persistence
        test_embedding = Embedding(
            invocation_id=test_id,
            embedding_model="test_embedding_model",
            vector=[0.1, 0.2, 0.3, 0.4, 0.5]
        )

        save_embedding(test_embedding)
        retrieved_embeddings = get_embeddings_for_invocation(test_id)

        assert len(retrieved_embeddings) == 1
        assert retrieved_embeddings[0].invocation_id == test_id
        assert retrieved_embeddings[0].embedding_model == "test_embedding_model"
        assert len(retrieved_embeddings[0].vector) == 5
        assert np.isclose(retrieved_embeddings[0].vector[0], 0.1)
        assert np.isclose(retrieved_embeddings[0].vector[-1], 0.5)
