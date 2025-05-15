import os

import polars as pl
from sqlmodel import Session, select

from panic_tda.export import export_video
from panic_tda.schemas import Invocation, Run


def prompt_timeline_run_ids(session: Session):
    selected_prompts = [
        "a picture of a man",
        "a picture of a woman",
        # "a picture of a child",
        "a cat",
        "a dog",
        # "a rabbit",
    ]
    selected_networks = [
        ["FluxSchnell", "BLIP2"],
        ["FluxSchnell", "Moondream"],
        ["SDXLTurbo", "BLIP2"],
        ["SDXLTurbo", "Moondream"],
    ]
    # Initialize empty list for selected IDs
    selected_ids = []

    # Loop through each prompt in selected_prompts
    for prompt in selected_prompts:
        # For each network pair, get 2 runs
        for network in selected_networks:
            # Query the database for runs matching prompt and models
            matching_runs = (
                session.query(Run)
                .filter(
                    Run.initial_prompt == prompt,
                    Run.network == network,
                )
                .limit(2)
                .all()
            )

            # Extract run IDs and add to selected_ids
            run_ids = [str(run.id) for run in matching_runs]
            selected_ids.extend(run_ids)

    return selected_ids


def prompt_category_mapper(initial_prompt: str) -> str:
    """
    Map an initial prompt to its category based on the predefined mapping.

    Args:
        initial_prompt: The initial prompt to map

    Returns:
        The category of the prompt or None if not found
    """
    # Mapping from initial prompts to categories derived from prompt-counts.json
    prompt_to_category = {
        "a painting of a man": "people_portraits",
        "a picture of a child": "people_portraits",
        "a picture of a man": "people_portraits",
        "a photorealistic portrait of a child": "people_portraits",
        "a painting of a woman": "people_portraits",
        "a painting of a child": "people_portraits",
        "a photo of a child": "people_portraits",
        "a photo of a man": "people_portraits",
        "a photorealistic portrait of a woman": "people_portraits",
        "a photo of a woman": "people_portraits",
        "a photorealistic portrait of a man": "people_portraits",
        "a picture of a woman": "people_portraits",
        "a photorealistic portrait photo of a child": "people_portraits",
        "a photorealistic portrait photo of a woman": "people_portraits",
        "a photorealistic portrait photo of a man": "people_portraits",
        "nah": "abstract",
        "yeah": "abstract",
        "a giraffe": "animals",
        "a cat": "animals",
        "an elephant": "animals",
        "a hamster": "animals",
        "a rabbit": "animals",
        "a dog": "animals",
        "a lion": "animals",
        "a goldfish": "animals",
        "a red circle on a black background": "geometric_shapes",
        "a red circle on a yellow background": "geometric_shapes",
        "a blue circle on a yellow background": "geometric_shapes",
        "a yellow circle on a blue background": "geometric_shapes",
        "a blue circle on a red background": "geometric_shapes",
        "a yellow circle on a black background": "geometric_shapes",
        "a blue circle on a black background": "geometric_shapes",
        "a red circle on a blue background": "geometric_shapes",
        "a yellow circle on a red background": "geometric_shapes",
        "a pear": "food",
        "a banana": "food",
        "an apple": "food",
        "a boat": "transportation",
        "a train": "transportation",
        "a car": "transportation",
        "orange": "colours",
        "green": "colours",
        "yellow": "colours",
        "red": "colours",
        "blue": "colours",
        "indigo": "colours",
        "violet": "colours",
    }

    return prompt_to_category.get(initial_prompt)


def render_hallway_videos(session: Session) -> str:
    prompt_lists = {
        "men": [
            "a painting of a man",
            "a picture of a man",
            "a photo of a man",
            "a photorealistic portrait of a man",
        ],
        "women": [
            "a painting of a woman",
            "a picture of a woman",
            "a photo of a woman",
            "a photorealistic portrait of a woman",
        ],
        "child": [
            "a painting of a child",
            "a picture of a child",
            "a photo of a child",
            "a photorealistic portrait of a child",
        ],
        # "abstract": ["nah", "yeah"],
        "animals": [
            "a giraffe",
            "a cat",
            "an elephant",
            "a hamster",
            "a rabbit",
            "a dog",
            "a lion",
            "a goldfish",
        ],
        "shapes": [
            "a red circle on a black background",
            "a red circle on a yellow background",
            "a blue circle on a yellow background",
            "a yellow circle on a blue background",
            "a blue circle on a red background",
            "a yellow circle on a black background",
            "a blue circle on a black background",
            "a red circle on a blue background",
            "a yellow circle on a red background",
        ],
        "misc": ["a pear", "a banana", "an apple", "a boat", "a train", "a car"],
        "colours": ["orange", "green", "yellow", "red", "blue", "indigo", "violet"],
    }

    created_videos = []

    # Loop through each category in prompt_lists
    for category, prompts in prompt_lists.items():
        print(f"Processing category: {category}")

        # Query all runs matching any prompt in this category
        runs = []
        for prompt in prompts:
            # Query runs with matching initial prompt
            prompt_runs = session.exec(
                select(Run).where(Run.initial_prompt == prompt)
            ).all()
            runs.extend(prompt_runs)

        if runs:
            # Export runs to video named after the category
            output_file = f"output/video/{category}.mp4"
            export_video(
                session=session,
                fps=2,
                resolution="8K",
                run_ids=[str(run.id) for run in runs],
                output_file=output_file,
            )
            created_videos.append(output_file)
            print(f"Exported {len(runs)} runs to {output_file}")
        else:
            print(f"No runs found for category {category}")

    # Return summary of created videos
    return f"Created {len(created_videos)} videos: {', '.join(created_videos)}"


def droplet_and_leaf_invocations(
    session: Session,
) -> tuple[list[Invocation], list[Invocation]]:
    """
    Returns IDs of invocations related to "leaf" or "droplet".

    Finds:
    - Image invocations where input contains "leaf" or "droplet"
    - Text invocations where output contains "leaf" or "droplet"

    Args:
        session: Database session

    Returns:
        List of invocation IDs as strings
    """
    # Find text invocations with "leaf" or "droplet" in output
    query = select(Invocation).where(
        (Invocation.type == "text")
        & (
            Invocation.output_text.ilike("%leaf%")
            | Invocation.output_text.ilike("%droplet%")
        )
    )
    text_invocations = session.exec(query).all()

    # Find text invocations with leaf/droplet in output to use as input sources
    leaf_droplet_text_ids = [inv.id for inv in text_invocations]

    # Find image invocations that use these text invocations as input
    image_invocations = []
    if leaf_droplet_text_ids:
        query = select(Invocation).where(
            (Invocation.type == "image")
            & (Invocation.input_invocation_id.in_(leaf_droplet_text_ids))
        )
        image_invocations = session.exec(query).all()

    # Combine results
    return (image_invocations, text_invocations)


def create_top_class_image_grids(
    embeddings_df: pl.DataFrame, limit: int, session: Session
):
    """
    Creates image grids from the top embedding clusters.

    This function:
    1. Finds embeddings in the top cluster for each embedding_model and network
    2. Loads up to 1600 invocations from each grouping
    3. Creates image grids of the input images
    4. Exports the grids with descriptive filenames

    Args:
        embeddings_df: DataFrame containing embedding data
        session: Database session

    Returns:
        List of exported image paths
    """
    from uuid import UUID

    from panic_tda.data_prep import filter_top_n_clusters

    # Create output directory if it doesn't exist
    output_dir = "output/vis/cluster_grids"
    os.makedirs(output_dir, exist_ok=True)

    clustered_embeddings = embeddings_df.filter(
        (pl.col("cluster_label").is_not_null()) & (pl.col("cluster_label") != "OUTLIER")
    )

    top_clusters = filter_top_n_clusters(
        clustered_embeddings, 1, ["embedding_model", "network"]
    ).select("embedding_model", "network", "invocation_id", "cluster_label")

    # Create a nested dictionary organized by embedding_model and network
    nested_dict = {}
    for row in top_clusters.iter_rows(named=True):
        embedding_model = row["embedding_model"]
        network = row["network"]
        invocation_id = row["invocation_id"]

        # Initialize nested structure if it doesn't exist
        if embedding_model not in nested_dict:
            nested_dict[embedding_model] = {}
        if network not in nested_dict[embedding_model]:
            nested_dict[embedding_model][network] = []

        # Add invocation_id to the list
        nested_dict[embedding_model][network].append(invocation_id)

    # Loop through the nested dictionary by embedding model and network
    from panic_tda.export import image_grid

    for embedding_model, networks in nested_dict.items():
        for network, invocation_ids in networks.items():
            print(f"Processing {embedding_model} / {network}")

            limited_invocation_ids = [UUID(id) for id in invocation_ids[:limit]]

            # Fetch the invocations from the database
            query = select(Invocation).where(Invocation.id.in_(limited_invocation_ids))
            invocations = session.exec(query).all()

            # Get the cluster label for this embedding_model/network combo
            cluster_filters = (pl.col("embedding_model") == embedding_model) & (
                pl.col("network") == network
            )
            filtered_clusters = top_clusters.filter(cluster_filters)

            # Take the first cluster label (should be the same for all rows in this group)
            if len(filtered_clusters) > 0:
                cluster_name = filtered_clusters.select("cluster_label").row(0)[0]
            else:
                cluster_name = "unknown_cluster"

            output_file = f"{output_dir}/{embedding_model}_{network.replace(' → ', '')}__{cluster_name.replace(' ', '_').replace(',', '').replace('.', '')}.jpg"

            # export image grid (save to file)
            image_grid(
                [inv.input_invocation.output for inv in invocations],
                32,
                16 / 10,
                output_file,
            )
