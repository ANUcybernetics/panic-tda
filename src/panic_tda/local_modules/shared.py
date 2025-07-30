"""Shared utilities for local paper modules."""

import os
from uuid import UUID

import polars as pl
from sqlalchemy.orm import aliased
from sqlmodel import Session, func, select

from panic_tda.data_prep import filter_top_n_clusters
from panic_tda.datavis import plot_cluster_example_images
from panic_tda.export import export_video, image_grid
from panic_tda.schemas import Invocation, PersistenceDiagram, Run


def example_run_ids(session: Session):
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

    # TODO this could all be one SQL statement, but :shrug: for now
    # Loop through each prompt in selected_prompts
    for prompt in selected_prompts:
        # For each network pair, get 2 runs
        for network in selected_networks:
            # Query the database for runs matching prompt and models
            matching_runs = session.exec(
                select(Run)
                .where(
                    Run.initial_prompt == prompt,
                    Run.network == network,
                )
                .order_by(Run.id)
                .limit(2)
            ).all()

            # Extract run IDs and add to selected_ids
            run_ids = [str(run.id) for run in matching_runs]
            selected_ids.extend(run_ids)

    return selected_ids


def export_cluster_examples(embeddings_df, num_images, session):
    # Sort by cluster_label to get a deterministic ordering
    top_labels = (
        embeddings_df.group_by("cluster_label")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(num_images // 2)  # We'll show 2 images per cluster
        .get_column("cluster_label")
        .to_list()
    )

    # Filter to top clusters
    top_clusters_df = embeddings_df.filter(pl.col("cluster_label").is_in(top_labels))

    plot_cluster_example_images(
        top_clusters_df, num_images, "Nomic", session, "output/vis/cluster-examples.jpg"
    )


def run_counts(runs_df: pl.DataFrame, grouping_cols: list[str]):
    """Count runs by specified grouping columns.

    Args:
        runs_df: DataFrame containing run data
        grouping_cols: List of column names to group by

    Returns:
        DataFrame with counts and percentages for each group
    """
    total_runs = runs_df.height

    return (
        runs_df.group_by(grouping_cols)
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / total_runs * 100).round(2).alias("percentage"))
        .sort("count", descending=True)
    )


def cluster_counts(embeddings_df: pl.DataFrame, num_clusters: int) -> pl.DataFrame:
    """
    Count and rank clusters by frequency.

    Args:
        embeddings_df: DataFrame containing embeddings with cluster labels
        num_clusters: Number of top clusters to return

    Returns:
        DataFrame with cluster counts and percentages
    """
    # Filter out outliers
    df_filtered = embeddings_df.filter(
        (pl.col("cluster_label").is_not_null()) & (pl.col("cluster_label") != "OUTLIER")
    )

    total_non_outlier = df_filtered.height

    return (
        df_filtered.group_by(["embedding_model", "cluster_label"])
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / total_non_outlier * 100).alias("percentage"))
        .sort("count", descending=True)
        .head(num_clusters)
        .with_columns(
            pl.col("count")
            .rank(method="ordinal", descending=True)
            .over("embedding_model")
            .alias("cluster_index")
        )
    )


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
    # Create output directory if it doesn't exist
    output_dir = "output/vis/cluster_grids"
    os.makedirs(output_dir, exist_ok=True)

    clustered_embeddings = embeddings_df.filter(
        (pl.col("cluster_label").is_not_null()) & (pl.col("cluster_label") != "OUTLIER")
    )

    top_clusters = filter_top_n_clusters(clustered_embeddings, 1, ["network"]).select(
        "embedding_model", "network", "invocation_id", "cluster_label"
    )

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

            output_file = f"{output_dir}/{embedding_model}_{network.replace('→', '')}__{cluster_name.replace(' ', '_').replace(',', '').replace('.', '')}.jpg"

            # export image grid (save to file)
            image_grid(
                [inv.input_invocation.output for inv in invocations],
                32,
                16 / 10,
                output_file,
            )


def list_completed_run_ids(session: Session, first_n: int) -> list[str]:
    """
    Returns the first N run IDs for each combination of initial prompt and network
    that have an associated persistence diagram.

    This is handy when you want to select "balanced" groups (ind. var. combinations)
    for analysis.

    NOTE: if you've got a runs_df you're happy with, you can get the same data with:

    ```
    run_ids = (
        runs_df
        .unique(subset=["network", "initial_prompt", "run_id"])
        .group_by("network", "initial_prompt")
        .head(4)
    ).get_column("run_id").to_list()
    ```

    Args:
        session: Database session
        first_n: Maximum number of run IDs to return per initial prompt and network combination

    Returns:
        List of run IDs as strings, grouped by initial prompt and network
    """

    # Use a CTE (Common Table Expression) to rank runs within each prompt/network group
    ranked_runs = (
        select(
            Run.id,
            Run.initial_prompt,
            Run.network,
            func.row_number()
            .over(partition_by=[Run.initial_prompt, Run.network], order_by=Run.id)
            .label("row_num"),
        )
        .join(PersistenceDiagram, PersistenceDiagram.run_id == Run.id)
        .distinct()
        .cte("ranked_runs")
    )

    # Alias the CTE
    rr = aliased(ranked_runs, name="rr")

    # Get the run IDs with row_num ≤ first_n
    query = select(rr.c.id).where(rr.c.row_num <= first_n)

    run_ids = session.exec(query).all()

    return [str(run_id) for run_id in run_ids]


# these are the first 4 run IDs for all prompt + network combos (4 is currently the max value to have balanced groups)
TOP4_RUN_IDS = [
    "067f8931-1b3a-784b-8aca-fd3f58ba81b2",
    "067f8931-1b47-7bb2-88db-448cea799123",
    "067f8931-1b2d-731f-93aa-06af52db4364",
    "067f8931-1b55-7312-9bac-23eb655ed7e7",
    "067feecb-2c46-7483-8fd2-c95155c53a63",
    "067feecb-2e7a-75c3-be69-31241df1562e",
    "067feecb-2dc5-73c2-8aee-524725a98c5e",
    "067feecb-2cd2-73b5-800c-cd0ad630db57",
    "067f8931-1bb9-737b-a187-985cd139ab77",
    "067f8931-1b9d-7e95-81a4-3a90789ba455",
    "067f8931-1bab-7707-8a62-a0d87435bac1",
    "067f8931-1b90-7bf0-8f94-c77945f9f1dd",
    "067fcc93-26a3-75f5-93a4-9a6f5f0504e0",
    "067fcc93-26b1-7fcd-af6a-18727961f7ee",
    "067fcc93-2668-7f89-9061-dc6ec048d878",
    "067fcc93-26c0-7888-a58d-c4f2b064a868",
    "067fcc93-2702-74e9-b1a8-fc113ef34de1",
    "067fcc93-26e5-73fa-b439-680fb2f7ecc6",
    "067fcc93-272e-742f-9fc8-45d971b11db3",
    "067fcc93-271f-7ae0-9d9d-d4ed829f64a2",
    "067f8931-1b19-7f42-a5e6-eb14ebd9bd27",
    "067f8931-1aef-73d7-9e97-fc29cc895de1",
    "067f8931-1afe-7488-89c2-b489fd4b6943",
    "067f8931-1b0c-7608-93d6-8d74e9c7555b",
    "067ee281-7209-72f0-b5e5-77e68c61d676",
    "067ee281-71e9-7316-a502-b9ffade4fb0d",
    "067ee281-71f9-7331-a4eb-4e566d553149",
    "067ee281-7219-704a-8415-cfb03ba3742b",
    "067ee281-71b9-7e0b-b734-8ab57dcee282",
    "067ee281-71aa-70e5-aa7d-3264f7661eb3",
    "067ee281-71c9-7906-ad02-941ad32e0ceb",
    "067ee281-71d9-7728-8fb2-9f066113bd59",
    "067feecb-3181-7401-9069-f3e75f3ab1bf",
    "067feecb-307e-7201-998a-4c0c706d49e3",
    "067feecb-30a9-7957-9c1e-d6c3b2883639",
    "067feecb-3196-7b0d-b845-fb3693e50246",
    "067fcc93-2712-7e37-b8bf-3dcaa383dbd9",
    "067fcc93-2704-7364-bbe8-4aa72ac27492",
    "067fcc93-2730-717d-a42f-6d7b3537a64b",
    "067fcc93-26f5-7a58-9907-7d712c51cd54",
    "067ee281-71d2-7418-8eb5-1cac41dbe406",
    "067ee281-71a3-7050-a031-15b92d96b8ab",
    "067ee281-71c2-79c0-8835-01822b800480",
    "067ee281-71b2-7d1a-b739-1fe1576ca87d",
    "067f8931-1b2f-7176-94eb-6bad2a016445",
    "067f8931-1b3c-76d8-b174-6abc65a0a871",
    "067f8931-1b57-71bf-9f4d-67200d1ba354",
    "067f8931-1b49-7a04-a6c9-11a721e46a62",
    "067ed16c-ea24-7cd5-b404-f28751a51d22",
    "067ed16c-ea35-7405-bd47-44f3f0f2bef4",
    "067ed16c-ea45-7a2b-baff-11b19a698161",
    "067ed16c-ea14-7c5f-bcb0-3b72274d536c",
    "067fcc93-2672-74e0-a176-fd0ad0c3e185",
    "067fcc93-2680-7ec5-8055-80d0db1fe475",
    "067fcc93-269d-7e9e-b5ac-be787ba94d16",
    "067fcc93-2663-7746-a9b6-4f176db6dde4",
    "067fcc93-2600-7222-8d2a-43a8c7b12177",
    "067fcc93-2648-73ab-984e-f004c7244356",
    "067fcc93-262b-740b-a79b-2712dd93a5fd",
    "067fcc93-261c-7c75-8fed-149b6f737e54",
    "067feecb-33cf-73c2-b697-a7fb6a598ce8",
    "067feecb-33a3-7ea2-94ed-76f55faa5e6a",
    "067feecb-3528-7cea-ab15-d831f921c512",
    "067feecb-347c-7346-88be-543854a08c3f",
    "067ed16c-ea31-78e7-8c49-d4d9ecb6128d",
    "067ed16c-ea42-7032-ad21-68ad1c2a561c",
    "067ed16c-ea21-73bc-9910-32c81af0272d",
    "067ed16c-ea11-73b0-a7f6-5e81382c15b7",
    "067feecb-36df-735f-98ab-5aa7aa331fd6",
    "067feecb-3892-7286-8d39-4b005ce233d1",
    "067feecb-371f-7479-a8d2-d76feda77c24",
    "067feecb-3631-7e18-ba83-bb5420d0ebe3",
    "067ee281-71a4-7c38-ab34-2a8f0a75d5a7",
    "067ee281-71d4-7065-bef3-b5e071bf7d65",
    "067ee281-71b4-7976-a1c2-c7b72469d15b",
    "067ee281-71c4-75b3-90e1-508cbe392530",
    "067ed16c-e9f6-751d-b5be-33ad05a65bb0",
    "067ed16c-ea06-7901-9101-b7efbe19e9a0",
    "067ed16c-e9e5-7c69-805f-6cc87676fe92",
    "067ed16c-e9d5-7104-b8c4-c6b0e0c88938",
    "067ee281-7166-7097-8a30-63b7c3fd0941",
    "067ee281-7186-72c1-b769-d9c932c36376",
    "067ee281-7196-7a49-b429-7a16afd6497b",
    "067ee281-7175-7ff5-b8ea-b910a3e9336e",
    "067feecb-3127-73ff-beee-3a22efa1531f",
    "067feecb-32d6-7ccc-9a83-3c7c606fffdc",
    "067feecb-3256-7686-9506-9acc9fc1b984",
    "067feecb-304f-77a8-8940-ace0f4d66ac5",
    "067ee281-721a-7e17-a0dd-ee354dbb21ae",
    "067ee281-720a-7f0c-8c81-92ca98066564",
    "067ee281-71ea-7f2b-8d26-3eff4b190f19",
    "067ee281-71fa-7fc0-9927-906a84e5c49c",
    "067feecb-3596-7c2d-a3c0-a2fe0063b439",
    "067feecb-3350-740a-89e4-43e1dac0a5e8",
    "067feecb-3427-7e33-8a51-1278b74be901",
    "067feecb-34be-77c8-b3ee-b060a0f20889",
    "067fcc93-2718-77f6-bbbb-5d3a0ab31a02",
    "067fcc93-2726-7ef2-9811-4d9291b063fb",
    "067fcc93-2735-7802-b4c2-afa4457f341b",
    "067fcc93-26dd-787f-8324-1431b2342f9a",
    "067f8931-1bc0-7d30-919d-9fcb2664ebb9",
    "067f8931-1ba5-7ac8-b6ee-36d4f510e8e9",
    "067f8931-1bb3-7816-877a-596d2b12d240",
    "067f8931-1b98-73fc-a6dd-30230680e1ab",
    "067feecb-386c-7ae5-a3c8-a2cbcf0edff8",
    "067feecb-38ad-7b7e-a475-dd20c286052d",
    "067feecb-36fa-7165-ad14-7af3cb683180",
    "067feecb-3637-756b-a99d-473e986d702c",
    "067f8931-1b96-75c3-945f-e5bb92c5c315",
    "067f8931-1ba3-7cc4-94a1-9b01d3f47d0a",
    "067f8931-1bb1-7953-9c1c-15788e515676",
    "067f8931-1bbe-7f04-b148-ebe243c3ed90",
    "067feecb-331a-76c3-9b68-5828cd021473",
    "067feecb-35cd-70e2-aad0-2b673f94fb0f",
    "067feecb-33b0-76db-bc4e-282348487468",
    "067feecb-3560-7f0f-8205-914fc31e7256",
    "067feecb-32b3-743a-8a96-ef0ce94f906e",
    "067feecb-316f-7767-a7d2-3d42ff4a659f",
    "067feecb-30d8-756d-b929-c110ff4da7e5",
    "067feecb-3184-7d43-9af6-fc19ca18aa84",
    "067f8931-1b79-7ac0-9a85-92838a60efb7",
    "067f8931-1b87-73df-8a96-9da2be99d381",
    "067f8931-1b5e-7cf1-9233-d0718d600a1f",
    "067f8931-1b6c-74cb-81df-6a334ee4cb77",
    "067f8931-1b94-77c1-ba3c-1b45d257ba3e",
    "067f8931-1bbd-7030-9b6c-44a623044afe",
    "067f8931-1ba1-7d8c-9222-14fff0942d5c",
    "067f8931-1baf-7af2-afb0-49e1bcd54b59",
    "067fcc93-26db-7bc7-a175-7fd4efb24d72",
    "067fcc93-26f9-73a0-b336-700b66697ee3",
    "067fcc93-2707-7df9-9974-9f136ef739d6",
    "067fcc93-2716-7ac4-8502-95fc719ac957",
    "067feecb-2ebe-74ce-88e4-cdac66007ee9",
    "067feecb-3042-7f4c-aeee-a5a21eb5f413",
    "067feecb-2d97-7605-8d16-578ca5a6322f",
    "067feecb-302d-761b-b1b0-ca884aaacc60",
    "067f8931-1af8-7234-b4a0-0f17f9697b4a",
    "067f8931-1b06-7851-8216-79af775790b6",
    "067f8931-1b21-7a62-8552-011690ec10c8",
    "067f8931-1b14-722b-993b-d9e5f3a4815f",
    "067ed16c-ea65-7fab-aec0-6221fd6e5ad7",
    "067ed16c-ea55-7e23-84ea-f2bd2b93f865",
    "067ed16c-ea85-7e16-b515-0b35ad62fcec",
    "067ed16c-ea75-7e06-96a5-06af7e1c6602",
    "067f8931-1b53-74c8-b88a-62111f4e634c",
    "067f8931-1b45-7cd8-a7c3-2fa00bbd359a",
    "067f8931-1b38-7a24-becf-ae106ae83255",
    "067f8931-1b2b-7417-8af6-3b7b55cdf11d",
    "067ed16c-ea7e-7bce-8ed6-ed1c544f1307",
    "067ed16c-ea6e-7d42-928d-836e9c7bbb71",
    "067ed16c-ea4e-7aff-97ac-a3d5947b0b7b",
    "067ed16c-ea5e-7dc9-a6a5-aadbba1948d4",
    "067ee281-712d-7a9a-8191-ca1d8711f961",
    "067ee281-715e-7f90-a3cc-f8cbf22480ea",
    "067ee281-713e-7745-ad59-af122862434c",
    "067ee281-714e-7a1b-ad31-83ba5017b2a0",
    "067ed16c-eabd-78ae-a924-5fcb38d74ea7",
    "067ed16c-eaad-76f4-a246-56eab836e507",
    "067ed16c-ea9d-7172-9b7c-49f638771204",
    "067ed16c-ea8c-7ed0-b82c-2430c19295df",
    "067fcc93-25da-7665-a810-cf0cc84272bd",
    "067fcc93-25bc-7862-8dc8-d32b9b237215",
    "067fcc93-259f-7424-961a-5c2d2ad560d6",
    "067fcc93-25cb-77a1-81b0-8cfee6ab3b4d",
    "067ee281-71af-7479-b9be-45ed7504f0b0",
    "067ee281-71bf-71d0-bcaa-dac9265cb863",
    "067ee281-71de-7b6f-9608-e407501866fb",
    "067ee281-71ce-7c6b-985b-afa9041ef181",
    "067fcc93-25c0-72af-9481-cb4035478885",
    "067fcc93-25cf-72ef-ad7c-7572e3b97441",
    "067fcc93-2594-72ce-bfb2-50b1cc9d9828",
    "067fcc93-2575-7d32-88fd-76eabb4373e1",
    "067feecb-34ac-79a2-9453-e9c4dfba38ab",
    "067feecb-352e-7390-8c12-85f92ceb430c",
    "067feecb-3481-7a82-bdfa-c242a87c6dbf",
    "067feecb-34c2-7086-afba-e344b9900fff",
    "067f8931-1b8c-7f4f-8339-045356a6c2a4",
    "067f8931-1b64-79d6-9a50-b5acd9632dd2",
    "067f8931-1b71-7fcb-81fd-017fac0f1bd5",
    "067f8931-1b7f-7a69-a8cd-706ac6516d70",
    "067feecb-387e-765e-8ac0-aedda5f60601",
    "067feecb-3649-7306-b762-56502ff26f23",
    "067feecb-38aa-71db-ac7e-797c7f61dfbd",
    "067feecb-36a0-76bb-bc8b-f9c0d1b579b1",
    "067f8931-1b9f-7f40-a284-1fec8cc14515",
    "067f8931-1bad-754f-88c9-0389b331263e",
    "067f8931-1b92-79b7-afe4-570e451244ea",
    "067f8931-1bbb-7216-b0ed-c67a40941c23",
    "067ee281-7188-7167-a497-0e02765f2470",
    "067ee281-7167-7dce-84e7-9d5f33174621",
    "067ee281-7177-7bfe-89db-c6608feba458",
    "067ee281-7198-7657-a4c8-69d1afd71b31",
    "067feecb-36d0-7ec6-bd14-41e839e35494",
    "067feecb-36e6-7506-91df-7659cce175ce",
    "067feecb-373c-748c-8311-e39b9295e6e4",
    "067feecb-3752-7261-8ce6-2193cbf030ef",
    "067ed16c-ea52-746f-9f01-242f029b5629",
    "067ed16c-ea72-75f7-ae4c-3c7f6227db42",
    "067ed16c-ea62-76a7-ae94-5769bb442140",
    "067ed16c-ea82-752c-aa7e-9a4d3b273fdc",
    "067feecb-2e11-70c1-b888-6758f023bbbd",
    "067feecb-2e3d-7c33-8e76-8a59f21c4360",
    "067feecb-2d49-7a1b-99b2-3385c7413269",
    "067feecb-2d07-75fe-af01-3e664dc76480",
    "067ee281-71c0-7dc7-b555-23c08d224d44",
    "067ee281-71d0-7893-ae15-b38b8cf96b67",
    "067ee281-71b1-70a4-af97-7d7f77026f23",
    "067ee281-71e0-7713-b169-01e784a23e4c",
    "067fcc93-25a4-7a59-9f98-0927f458110e",
    "067fcc93-2577-7c6d-9cf2-9ad34e9400c1",
    "067fcc93-25b3-75ad-8f61-430f4be8aa1b",
    "067fcc93-2567-7d31-9153-35bf0b0cfad2",
    "067feecb-3516-7a1b-a7da-535569a9a54f",
    "067feecb-347f-7e35-8d03-e10dbff73b15",
    "067feecb-346a-73aa-a5c1-c22bfc470e17",
    "067feecb-3542-73f8-88bf-09df801f0442",
    "067ed16c-ea2a-7430-a393-e5a0e53037a3",
    "067ed16c-ea0a-7113-968c-ba5a10fe1fd9",
    "067ed16c-ea3a-7d57-8ab6-cf51214d9a4b",
    "067ed16c-ea1a-70db-b52a-c94616b1e736",
    "067ed16c-ea67-7bae-86ab-d63817bdcb97",
    "067ed16c-ea57-7afb-b5f4-5ace94d7f746",
    "067ed16c-ea77-7a1a-b907-4ce8ac13bac3",
    "067ed16c-ea87-7a94-bf8b-f26a6bb86027",
    "067fcc93-26f0-726f-9c64-28cc513c5676",
    "067fcc93-272a-7972-8cb8-135ac77aad0c",
    "067fcc93-26e1-79d7-8123-e6c8d65648ba",
    "067fcc93-270d-76da-bf0d-1ea274a15282",
    "067feecb-36b7-7e6b-81f1-524481a82f89",
    "067feecb-374e-7962-9fd1-74a02a985195",
    "067feecb-36e2-7c05-9790-3128cf82c56a",
    "067feecb-3763-7f4d-bbc2-1520c5a5ba60",
    "067fcc93-257f-76b4-9d27-dc6b0c5ff9e1",
    "067fcc93-2570-708e-93ca-4139a086e451",
    "067fcc93-25c9-79ca-b41e-166452fadf31",
    "067fcc93-258e-7952-b17b-497c3eb31a44",
    "067ed16c-e9cb-7422-8ad7-3e9af5c101ac",
    "067ed16c-e9fd-7a78-935c-1c081050c1f1",
    "067ed16c-e9dc-788c-92ac-b03c49c607de",
    "067ed16c-e9ed-70b7-869f-651767caec3f",
    "067feecb-37d8-78d4-ad9b-c71813e3ddab",
    "067feecb-36eb-7b37-a057-1547f9ffbe16",
    "067feecb-363e-7839-aa3d-bc960465009f",
    "067feecb-3695-7789-9e9d-4eafd6b1a520",
    "067ee281-7182-7646-913d-c2d4c229250e",
    "067ee281-7172-783c-bfa5-9be066b80bd0",
    "067ee281-7162-780b-9e98-4d442f267822",
    "067ee281-7193-7209-9ace-c4bbeea380f2",
    "067ee281-7156-71a1-925f-ff40f888cba1",
    "067ee281-7135-7353-a3c6-e764c395869a",
    "067ee281-7145-79d3-b56b-b93a747606fb",
    "067ee281-7123-7b99-b548-a961d102a73b",
    "067ed16c-ea84-71ac-b60e-960ed1471a2f",
    "067ed16c-ea74-7213-a77a-b73bfd87dd7e",
    "067ed16c-ea64-72e0-b83e-c5cd2b75f7e8",
    "067ed16c-ea54-7152-9b79-46d940d670a0",
    "067ee281-713a-7b72-8b58-2fda4835de4b",
    "067ee281-7129-7d1d-a887-65cbd6d9b078",
    "067ee281-715b-7635-8496-f11b04974270",
    "067ee281-714b-7046-949c-30f03d75259e",
    "067fcc93-2656-7d85-84f9-f57a5adb79e6",
    "067fcc93-2674-71f0-8add-c66fb0bad0d9",
    "067fcc93-2665-74bf-9190-20c2343ef19d",
    "067fcc93-2691-7341-a269-2fbf9f1bd0fb",
    "067f8931-1b8e-7dc9-9b0d-0dc8b9578ce6",
    "067f8931-1b66-7801-a624-01fccd04e4f5",
    "067f8931-1b81-7902-b564-c96dddf099c6",
    "067f8931-1b73-7dfb-8ebe-b06145ac032b",
    "067ed16c-ea38-7e53-9630-1e9243917645",
    "067ed16c-ea08-74d4-a36b-0b40b82bb281",
    "067ed16c-ea18-748e-bd4d-8ea191e33d03",
    "067ed16c-ea28-7768-acbf-3b9550f828ef",
    "067ee281-7127-7d44-a5d8-f096930dced7",
    "067ee281-7159-7a0b-b250-23ba2d698451",
    "067ee281-7149-730b-9fbc-9483c85f192c",
    "067ee281-7138-7e1a-9ae7-e7e6f31eae96",
    "067fcc93-257d-7849-baf1-615883840913",
    "067fcc93-256e-7100-ab4f-c64beb789bf2",
    "067fcc93-258c-7b59-8332-303e32c7467d",
    "067fcc93-25d6-791c-90d7-517132c411ca",
    "067f8931-1ba7-7904-a8d8-f7561756ed63",
    "067f8931-1bc2-7b9f-8411-13511dad9b89",
    "067f8931-1b9a-71bb-aaaf-0bbbe8c78b79",
    "067f8931-1bb5-768b-8a64-798358636414",
    "067ee281-7119-7b8c-b310-a7460b718e0f",
    "067ee281-7152-7744-9799-b66a417d442a",
    "067ee281-7142-70c7-a25f-33ca59639033",
    "067ee281-7131-7715-9d4f-789ea98812aa",
    "067f8931-1afc-74ba-8248-9994fc16851b",
    "067f8931-1b0a-7738-b35e-5abae5560221",
    "067f8931-1b18-7042-8710-5384a26ccc47",
    "067f8931-1ae9-7b59-a1e1-1889080abfa2",
    "067ed16c-ea9e-7f01-bd36-7a07a1f9d2c5",
    "067ed16c-eaaf-735c-b351-8fac0df598bd",
    "067ed16c-eabf-74b8-8f68-4657a7fa31ea",
    "067ed16c-ea8e-7b98-af84-6494ac25c1bd",
    "067ee281-712f-792c-9427-56b28bab4f21",
    "067ee281-7160-7b99-8777-dea194ea1c8b",
    "067ee281-7150-7a06-87d1-db10cc89f712",
    "067ee281-7140-7449-b0ac-05c8fd9db078",
    "067f8931-1b5a-7f34-ae8a-479b96955360",
    "067f8931-1b83-773d-8264-b08590ac047a",
    "067f8931-1b68-7811-b943-880da7ecf5c7",
    "067f8931-1b75-7dc0-85b0-14be35302c48",
    "067f8931-1b6a-769b-a656-47678978590d",
    "067f8931-1b85-75b0-83d2-8cab7f3fa955",
    "067f8931-1b77-7bd0-83db-72e1bbb34abb",
    "067f8931-1b5c-7de2-a4eb-e59b3ca14e08",
    "067feecb-3740-701d-86d9-cb449635f7e9",
    "067feecb-3780-77d4-ab1c-5f47c8b328e8",
    "067feecb-37c1-715e-8735-ec1653087381",
    "067feecb-37d6-7c2f-8a2d-dce22f60c925",
    "067ed16c-ea1f-76fa-a8c9-f6c941659ffe",
    "067ed16c-ea2f-7b97-bd6b-bb06a1a6dfb1",
    "067ed16c-ea0f-7740-acdf-e175d4675856",
    "067ed16c-ea40-73a3-9f3b-821a38a91dad",
    "067ed16c-eac8-7505-bac0-55956887875d",
    "067ed16c-eab8-7318-bc49-250168fd337a",
    "067ed16c-eaa8-7171-928f-e5c3c2967632",
    "067ed16c-ea97-794b-b387-e6306ac992ac",
    "067ee281-721c-7a5f-93d9-94319ab8026c",
    "067ee281-71fc-7c44-ba28-41bc5bc7bc0c",
    "067ee281-71ec-7c8d-9357-0a7ec955601a",
    "067ee281-720c-7b6a-97a1-259e376e696e",
    "067feecb-304d-7b8f-b00e-a55af9b937ae",
    "067feecb-30fa-734a-b482-6a0231083416",
    "067feecb-317b-7ed6-a561-350c911d8be0",
    "067feecb-327f-7419-b18e-cefacedc09ec",
    "067feecb-36e8-71c9-8e5d-1d0db86409b4",
    "067feecb-385a-7d79-b1f2-406b8666e941",
    "067feecb-3610-702f-b1f0-e39dc3c345c9",
    "067feecb-3753-7ec9-8516-342a369ac88d",
    "067ed16c-e9e0-753e-bb24-81ab84b5868e",
    "067ed16c-ea01-7361-a256-505ff2e1d7fb",
    "067ed16c-e9f0-7a45-8706-ba9d125c5b6e",
    "067ed16c-e9cf-7405-8186-226b79df9036",
    "067ed16c-e9e9-7699-9c3b-13d81830980f",
    "067ed16c-e9fa-70a1-ac8f-0b0807bf64fb",
    "067ed16c-e9c6-7f40-a074-6565a5e1cd8b",
    "067ed16c-e9d8-7d13-8eef-613a08dec975",
    "067fcc93-267d-7461-b5c9-1cec9bef91ac",
    "067fcc93-268b-7d06-93b9-4047f86ed91f",
    "067fcc93-26b7-7648-82fe-77f1f0352835",
    "067fcc93-269a-749f-950f-08cec0d2052b",
    "067f8931-1af3-7d08-91fc-2bb1c5ceb358",
    "067f8931-1b10-73de-b781-de1ad2c37e3f",
    "067f8931-1b1d-7c99-8894-8baa373f60c4",
    "067f8931-1b02-76c5-ada1-79dfdcf5ad41",
    "067feecb-342d-7361-af65-72605c951bc7",
    "067feecb-351a-7456-b9ac-b19910503652",
    "067feecb-3504-7bf3-9b7e-ecaec1be9017",
    "067feecb-34ef-70ba-9ea4-b8b005c797dc",
    "067fcc93-25f5-744b-9e6d-0ad88297b418",
    "067fcc93-263d-7236-9b64-46c5423c3d7b",
    "067fcc93-2612-703b-80d5-97db1c09a854",
    "067fcc93-25e6-7f65-9ff4-af2edffefdad",
    "067ee281-71e2-72f2-9a37-418c01dda084",
    "067ee281-7202-7287-8b42-5130abc903b1",
    "067ee281-7212-7030-83ab-21cfcd246b7a",
    "067ee281-71f2-71ef-bd0b-1c8013f662ae",
    "067feecb-34c7-75b8-a610-a6924041dddd",
    "067feecb-34dc-7d75-ae50-db0e4212b0b5",
    "067feecb-3486-7ee5-80d1-511f0f0d11ac",
    "067feecb-351d-7e6e-82b3-39896c0b1798",
    "067fcc93-2610-737a-a422-d500ae44a8a0",
    "067fcc93-261e-78ec-a768-be0fb451002c",
    "067fcc93-262d-7091-8ebe-1d847d9a6b07",
    "067fcc93-25f3-779c-ad4b-e57c887f77f8",
    "067ed16c-ea43-7d2c-8bce-46bbf09bacef",
    "067ed16c-ea13-700e-a2f5-5d10f60bb58f",
    "067ed16c-ea33-775c-ba7f-dc20e848a698",
    "067ed16c-ea23-7001-a660-8c351340f100",
    "067feecb-2d3a-7fd4-a53d-95c2a9bbda5c",
    "067feecb-2e2f-7219-b35a-57434f46cfb2",
    "067feecb-303f-757e-aab4-47bf91b53b9b",
    "067feecb-2ee5-7b31-b059-8dbb5be79bcc",
    "067fcc93-25c5-7ae9-a857-7373a2eaa4db",
    "067fcc93-258a-7d8f-aaac-19e8ffe22060",
    "067fcc93-25d4-7bf0-9d4b-dc50f55ede56",
    "067fcc93-256c-7008-a8ae-bbf369c70de7",
    "067ed16c-ea59-77f2-a88f-bdbd53dcdfdb",
    "067ed16c-ea79-76fe-b171-df7ba5c26c11",
    "067ed16c-ea49-740a-bef0-fc45cf38359a",
    "067ed16c-ea69-7852-ba1c-c8327b682e77",
    "067ee281-7200-752d-b68e-5965c850fb26",
    "067ee281-7220-729b-94d3-4cf50d993443",
    "067ee281-7210-73d5-81be-be9a2967ba54",
    "067ee281-71f0-74d0-975b-32b42fee09c1",
    "067feecb-3552-7944-8340-6266c6e0b929",
    "067feecb-3568-71d1-8ae5-c4d2e8c04848",
    "067feecb-34ba-7e6c-a1c9-b97d83a3536e",
    "067feecb-33cd-73bb-b260-cfa7c08dc343",
    "067fcc93-2637-7c27-b2ea-4fa91a76f6f0",
    "067fcc93-2646-765b-9cf2-5a9265b414e1",
    "067fcc93-260c-7a76-8476-b2b3e486cf5f",
    "067fcc93-261a-7f93-b187-99117bb7c68c",
    "067feecb-36d8-70b8-8bcb-6e0882532285",
    "067feecb-3640-74ef-80e3-5c2d32f4654b",
    "067feecb-3697-752f-a90d-54dbd0771225",
    "067feecb-3784-717a-bfac-07df852816ff",
    "067ee281-71d5-7c8c-87ea-5dc8b0cf2c0f",
    "067ee281-71c6-7180-97d6-82a97e56fbf4",
    "067ee281-71a6-7852-a8f2-c9b5748c727a",
    "067ee281-71b6-7632-9b28-3e5645ec0da4",
    "067ee281-716d-7400-84f9-4c1385c65645",
    "067ee281-717d-718b-a1ef-d5fa2e380243",
    "067ee281-719d-7b75-831b-2bd76a2e9bf0",
    "067ee281-718d-7a18-b361-ddccdf5e8706",
    "067ee281-71bb-79cd-85a7-85a0f32487ee",
    "067ee281-71cb-7451-9489-21676fa4d789",
    "067ee281-71ab-7c7b-8d9a-b6a0dc35cb07",
    "067ee281-71db-739e-8dd3-4aab55dcdb1a",
    "067f8931-1b43-7f04-b9f7-1a738d130b2a",
    "067f8931-1b51-761a-ad45-c63bcb6e670a",
    "067f8931-1b36-7b70-982a-7760c5f51c12",
    "067f8931-1b29-75d0-91b8-b00ff331bc5f",
    "067feecb-30da-723b-8151-c03ab48cf7c1",
    "067feecb-3099-7516-804d-b22059b02f32",
    "067feecb-30ef-78ec-af0f-a450735c91e4",
    "067feecb-321f-7a44-bca3-c238d25e7020",
    "067ed16c-eaa4-778f-a4a2-8fa48622ab25",
    "067ed16c-eac4-79d4-9787-30908cb605a5",
    "067ed16c-ea94-706d-a1b5-b8438412e035",
    "067ed16c-eab4-790c-a949-759d67681815",
    "067f8931-1b4d-7966-b040-a2e9a981696f",
    "067f8931-1b40-72eb-b776-24a3b4c2fc9a",
    "067f8931-1b25-786e-93dc-5c4759ba3c77",
    "067f8931-1b32-7e93-9b28-82aa38719a60",
    "067fcc93-2728-7c73-93ff-23528befbecf",
    "067fcc93-271a-7465-b5c9-c87840fdc8cd",
    "067fcc93-2737-74fb-9c88-56b0012b8d32",
    "067fcc93-26ee-75de-9aaa-9374ef89a1a8",
    "067fcc93-2617-7589-a97a-21787eef7d9e",
    "067fcc93-25dd-7fdc-8362-8f876d244f05",
    "067fcc93-2642-7a72-89a8-2f32785629d7",
    "067fcc93-2609-714b-99c0-7d9389f1a7dc",
    "067feecb-31e1-7031-9fd3-031972de65ca",
    "067feecb-315d-78bc-a647-6fe5d9ba521b",
    "067feecb-32a1-7743-9165-948be100207d",
    "067feecb-328b-7ad8-8515-77ff84b298d2",
    "067ed16c-eab1-7024-9691-f8741857e26b",
    "067ed16c-eac1-716d-8038-3e9b5fa958a7",
    "067ed16c-ea90-77e1-9965-737a4d88d53f",
    "067ed16c-eaa0-7c2a-8411-1cff31437c11",
    "067feecb-2d39-72c5-9199-d5c4b7789e91",
    "067feecb-303d-78c3-8e7d-6d7da713a1e7",
    "067feecb-2de9-7f7d-b1d4-56bc88c72427",
    "067feecb-2cca-7892-851b-29fe7f9ec0d1",
    "067ee281-71d7-7aa5-897e-bc5fd09fd54a",
    "067ee281-71a8-74c7-8bed-a26a33e90277",
    "067ee281-71b8-71d9-8d29-6b71851d6dbc",
    "067ee281-71c7-7d52-8bb3-8eea5be7f77b",
    "067f8931-1b8b-70f3-8c2a-2d786edb51ce",
    "067f8931-1b62-7b3c-afc3-c03f76bfcde0",
    "067f8931-1b70-7196-92b7-138b5f79b1cd",
    "067f8931-1b7d-7939-998c-f72328737596",
    "067ed16c-ea8b-72c3-8e76-1bdf452ef3d2",
    "067ed16c-eabb-7c4e-9888-34fc328fa7f0",
    "067ed16c-ea9b-74a7-87d7-0afb05c463ca",
    "067ed16c-eaab-79d4-87c9-3a734f539ffe",
    "067ee281-7154-74a8-9638-0383bb041404",
    "067ee281-7133-74ce-ad0a-4e9e983a4225",
    "067ee281-7121-776f-8eca-d0b8d29cf62e",
    "067ee281-7143-7dcd-8928-8269cb950945",
    "067feecb-2cac-7ee3-ae34-da1c79706fee",
    "067feecb-2dcc-7840-852c-4f13799687ab",
    "067feecb-2e9b-7de7-90c0-4a07ab300efe",
    "067feecb-2cd9-78dd-8bd8-ab820346073f",
    "067ed16c-e9da-7a30-a20e-eb45716e0986",
    "067ed16c-e9c9-7325-ac8b-1a76abbbef8c",
    "067ed16c-e9eb-740d-a6c3-7dca22327807",
    "067ed16c-e9fb-7d96-801d-74c333d10a1c",
    "067fcc93-26b5-79c1-bd7b-4a7ae059c0eb",
    "067fcc93-2698-775a-825a-d953099d2848",
    "067fcc93-2689-7fc8-823b-6d51602adeae",
    "067fcc93-264f-7904-837c-3d40f397e1f9",
    "067fcc93-25ee-7214-b3af-ee77c791341c",
    "067fcc93-25fc-786e-a7c4-0d8abaed9135",
    "067fcc93-260a-7da9-b9b4-cb0b653e1dbb",
    "067fcc93-2627-7a8e-a843-7540cf2b7a38",
    "067feecb-3123-7adb-ac46-663db25fca89",
    "067feecb-3164-7b46-acc2-ae6db56cc860",
    "067feecb-31e8-7235-8b7d-e3aa95079774",
    "067feecb-3268-73b2-9cf1-e32cbe07f23e",
    "067ed16c-ea2d-7eb2-97c8-386b768e92a0",
    "067ed16c-ea1d-7a4b-b370-2e14802abc66",
    "067ed16c-ea3e-76ba-ba33-064daef61c54",
    "067ed16c-ea0d-79a7-b060-e6df980d78d5",
    "067ee281-7125-7d21-8d23-2495b56ec715",
    "067ee281-7157-7de6-8694-d9fc9768ca48",
    "067ee281-7137-7110-9af2-37ca216af610",
    "067ee281-7147-76a5-a092-7fb087328450",
    "067ed16c-ea5b-7467-a4e3-ed285f6de507",
    "067ed16c-ea6b-7485-8200-2dc36f28ada9",
    "067ed16c-ea7b-7308-aa5f-443e76cfbdd9",
    "067ed16c-ea4b-7110-a055-1d3ce7ace419",
    "067feecb-2c8b-7acb-8889-dac7d53ef52d",
    "067feecb-2ebc-7854-9a86-3bc2d6f74e37",
    "067feecb-2e90-7e45-833e-ec2a33437103",
    "067feecb-2cb7-7fa3-97b2-0e17b87a66ca",
    "067feecb-3194-7e5a-9091-da46cba687cf",
    "067feecb-32c3-7427-a208-e2cdf8ba8ee6",
    "067feecb-3092-7206-864d-ff8934ee8a42",
    "067feecb-3051-7455-bedf-4398f985e922",
    "067ed16c-ea03-7066-b009-ed3dd06bed0e",
    "067ed16c-e9d1-7375-a691-1c46a65fb2d8",
    "067ed16c-e9e2-7240-84eb-48fa2fa111ed",
    "067ed16c-e9f2-771f-a62a-5d13c0f64676",
    "067ed16c-ea2c-7134-b4bd-5ff61c5891de",
    "067ed16c-ea1b-7d54-a181-2c97ce8bd981",
    "067ed16c-ea0b-7d31-b995-a6590f52afe5",
    "067ed16c-ea3c-7a23-9a1e-0a1e9b3b5621",
    "067ed16c-eaa2-7a1c-b4ba-9ce56e2fd758",
    "067ed16c-eab2-7cd1-b1f9-32b7a81595e7",
    "067ed16c-eac2-7d7b-88b0-fc030a85e06b",
    "067ed16c-ea92-73c4-b2c7-d65b779ae6cb",
    "067fcc93-26a5-72b8-aed5-3ca629bf4e8d",
    "067fcc93-265c-72fb-a7b5-f23d9a458637",
    "067fcc93-2696-7a4b-9602-bd9e30c0931a",
    "067fcc93-266a-7c85-8198-192eac2092db",
    "067feecb-384c-7adc-8cfd-4e373e60589a",
    "067feecb-36c4-7713-ba6a-7753faf17e51",
    "067feecb-380b-7ac4-a665-6e42a8981b99",
    "067feecb-3699-7248-97de-0e2b0b2cf511",
    "067ee281-718b-7bf3-af71-3813446d4cc5",
    "067ee281-719b-7fa7-9ea6-24a79b791913",
    "067ee281-717b-7411-8193-bc140ad3147a",
    "067ee281-716b-77bd-8783-abe629c011de",
    "067fcc93-2573-7ec2-865b-f4a34fa0ece7",
    "067fcc93-2592-74e4-b014-553596500a0b",
    "067fcc93-25af-7970-8ec1-6d4820216902",
    "067fcc93-25be-7586-8bbd-930c59a9a38a",
    "067ee281-720e-77cb-9641-1346496215cb",
    "067ee281-71ee-7899-803f-315fcd8a98af",
    "067ee281-71fe-7936-9f67-328ad77f9a20",
    "067ee281-721e-762a-b6f8-5cb6823f6ce5",
    "067feecb-2ca7-7531-8ceb-4147d1fc14e0",
    "067feecb-2e20-76d8-930f-7d09ff496414",
    "067feecb-2d00-7112-be07-d52f490b619e",
    "067feecb-2d2c-73d0-b9f6-81f426323d60",
    "067f8931-1b3e-74c8-beff-ac64220a8f20",
    "067f8931-1b31-7076-bd94-85c7da2abc15",
    "067f8931-1b4b-78f1-a1a1-5870acb4fff6",
    "067f8931-1b59-7048-91cf-48018a52eed2",
    "067feecb-305c-704e-97bf-0cecad964ced",
    "067feecb-32a3-741a-ae03-9d1978ab4cc3",
    "067feecb-3133-7e7a-ad60-eb8ecb4b8ab9",
    "067feecb-32e3-7268-ba09-add06db27a7b",
    "067ee281-71ad-788f-b58b-3ca1ddcd0bcd",
    "067ee281-71dc-7f94-b2b9-cf198548a94c",
    "067ee281-71bd-75df-a1ab-70390c5221bc",
    "067ee281-71cd-704c-a5e9-6cf67200b0ac",
    "067feecb-3823-7c36-92a1-4b1c01912ae4",
    "067feecb-36c6-7368-ac2f-228a59c63fa7",
    "067feecb-379d-7174-957f-bae69db4d7b7",
    "067feecb-37b2-7af0-b778-9d4ed4318493",
    "067f8931-1b0e-7522-b05a-0ac23b50f13e",
    "067f8931-1b00-7651-a465-e2088b23ac8f",
    "067f8931-1b1b-7dd2-8f0a-4afc58009312",
    "067f8931-1af1-7943-b54d-a748f8a425a1",
    "067feecb-33e0-7351-ac28-48dceb54c662",
    "067feecb-34a1-7ebe-8857-e3b45894df79",
    "067feecb-358f-79aa-b3ae-c2146c59db5d",
    "067feecb-3461-73ce-b386-baae1cd3b361",
    "067ed16c-ea5d-7135-84b0-bfc0cf5e2d4d",
    "067ed16c-ea4c-7dd3-b5d5-aab18cafe903",
    "067ed16c-ea7c-7f9c-b988-85eb1e14ece6",
    "067ed16c-ea6d-7128-9c1e-89a49cdd4261",
    "067feecb-2cc1-74a9-baad-0da3bea529b5",
    "067feecb-2c7e-79e6-98a1-59a38dcaf7df",
    "067feecb-2ced-7eba-9865-bae741ac5c18",
    "067feecb-2e24-7203-8aa0-0c75a41ec1ba",
    "067feecb-2c84-730f-97fd-a9903330ab53",
    "067feecb-2e9f-7b36-8847-3425b0c8ec05",
    "067feecb-2eb5-7695-afb1-6c07a27b30c6",
    "067feecb-2dba-754c-8629-1bde82f43e23",
    "067feecb-33de-7566-8c28-c063434b02df",
    "067feecb-345f-7709-8c11-239596965966",
    "067feecb-34a0-724e-87ec-7f090c6e0195",
    "067feecb-3475-7091-9367-47caeffcce23",
    "067f8931-1b89-728d-819f-d27b5be38170",
    "067f8931-1b60-7bcb-9b1b-b7d2833cdb11",
    "067f8931-1b6e-7360-93b8-701da8da4cc9",
    "067f8931-1b7b-7aac-904c-cdd924441f38",
    "067fcc93-25e8-7bc2-ae86-1920e46467a7",
    "067fcc93-2605-7856-9ada-037b42de7027",
    "067fcc93-25f7-718a-bf2b-83b557448855",
    "067fcc93-2613-7c9e-aa5f-d8f980f571f7",
    "067fcc93-2714-7e1b-9a97-789d2c8a3381",
    "067fcc93-26d9-7eaa-919f-23b117b421c3",
    "067fcc93-2706-70ea-ac7e-ba813d4588dd",
    "067fcc93-26e8-7e51-b25c-2be793bb64c3",
    "067feecb-304a-7229-9c1b-745bce5b1b7b",
    "067feecb-3178-7635-89d8-8f4de822c34d",
    "067feecb-32d1-76fc-b406-8bb7c9bbad54",
    "067feecb-3266-778f-ae9f-8acd46a646ed",
    "067feecb-38a6-75b8-a933-8f6dcf7ef0a3",
    "067feecb-3645-7a6e-817d-f6a97556c5c6",
    "067feecb-3850-739c-a8ce-ead7cacfdf07",
    "067feecb-35ee-79a0-857e-7617a7288e89",
    "067fcc93-2632-7585-aef1-8ca1cbb1d446",
    "067fcc93-2640-7d04-8a51-1a9bb7fe7803",
    "067fcc93-25ea-78b6-9f78-8c91852cb009",
    "067fcc93-2615-795b-aad3-8a1a2779d28d",
    "067ee281-7213-7c10-a48c-bd954f2714c6",
    "067ee281-71e3-7f33-9495-07f548fc6ef2",
    "067ee281-7203-7e3f-948e-3135e50b3457",
    "067ee281-71f3-7e88-a9da-33921106f5c0",
    "067feecb-350f-7807-a0c8-0aef546da2d0",
    "067feecb-33f7-7c3a-a1d2-ed7b1183ec39",
    "067feecb-34f9-7e2f-87d7-6a487e813a2b",
    "067feecb-34a3-7afd-8a4f-117697e3f887",
    "067ee281-7169-7b25-8db2-41a7409d5a69",
    "067ee281-7189-7e76-ae15-7c08da5714e0",
    "067ee281-7179-7802-b379-4565c0d81a01",
    "067ee281-719a-72b6-ab42-3e83d3633e9c",
    "067ee281-71e5-7b01-bf5d-f8892dd963d7",
    "067ee281-71f5-7b2e-8153-528250504f6d",
    "067ee281-7205-7a86-b38c-ce29fdce0837",
    "067ee281-7215-7833-b050-f4682b347e3f",
    "067feecb-32ba-7586-b1b3-b04441536713",
    "067feecb-320f-7707-afee-635408925d8f",
    "067feecb-32a5-7070-9d47-00f40d55f754",
    "067feecb-31b8-7594-985b-d2b3da560dec",
    "067ed16c-e9bf-74ce-b666-598215ac9294",
    "067ed16c-e9f8-72ce-a346-4e0460065215",
    "067ed16c-e9d6-7eec-97d3-6a5e930dbfa6",
    "067ed16c-e9e7-79a8-b901-102d29efc50d",
    "067ee281-7191-75db-b0e0-a2aa20841a58",
    "067ee281-71a1-73c8-85de-b10f3f292547",
    "067ee281-7180-7a2d-8680-389965c6b530",
    "067ee281-7170-7bd9-b493-d3321bfc6ee2",
    "067ed16c-eac6-7689-9c9b-bf312265682d",
    "067ed16c-eaa6-7532-aa38-30ee2cc718f7",
    "067ed16c-ea95-7c65-9a40-a2653bcd42e4",
    "067ed16c-eab6-7654-83b2-23b9f166143a",
    "067ed16c-ea26-7968-bdbf-0e31d638a521",
    "067ed16c-ea47-7795-9137-48a6877c8b16",
    "067ed16c-ea16-7881-8014-ea715be8744f",
    "067ed16c-ea37-70d1-a564-0622e7e8620a",
    "067fcc93-271d-7e21-9acb-f46be3228e85",
    "067fcc93-2700-7799-b56f-47290dc269c5",
    "067fcc93-26c5-7eb4-959b-176bba8e54f3",
    "067fcc93-26d4-7874-b4a4-41297b07559e",
    "067fcc93-269c-7188-94f6-a731b864ed87",
    "067fcc93-2670-77f7-a3ca-cfb63b3cb0df",
    "067fcc93-267f-71aa-b9b9-3d30359369f3",
    "067fcc93-268d-79ac-9337-97f08ca19aa1",
    "067ed16c-e9cd-7480-b9ce-de886cb31d0c",
    "067ed16c-e9ff-76f0-bfd2-911063a908bf",
    "067ed16c-e9ee-7d96-b0ae-5f2ec282cefc",
    "067ed16c-e9de-762d-a0db-60a913a3aa59",
    "067fcc93-2588-7fec-bc59-54ef4fca89b2",
    "067fcc93-2597-7f41-abb8-1c9445d616fb",
    "067fcc93-25d2-7e33-88b0-69594d2bd375",
    "067fcc93-2579-7b79-9f0c-b6531ee894f8",
    "067feecb-2dd2-7019-ae49-358f604dec19",
    "067feecb-2d0b-7025-83e0-9479314bf438",
    "067feecb-2c58-75c4-8cb1-3ee4dcf5f8de",
    "067feecb-2cb2-76ad-9910-dee44508130e",
    "067f8931-1b16-715c-b790-c75b5051fe0e",
    "067f8931-1afa-73cc-8ab7-a3fc53ed3b6d",
    "067f8931-1b23-7975-9faa-6cab5504c5c8",
    "067f8931-1b08-77cf-9efe-8c2bd91076c9",
    "067feecb-342e-7fcc-b9ca-f3a052d45e4c",
    "067feecb-3547-7d3a-8942-bc422b91e6e9",
    "067feecb-332c-721d-be4f-588bee308024",
    "067feecb-349a-7a7b-97f7-5037bc21b0e4",
    "067ed16c-ea60-7a1b-9579-aa9c50393371",
    "067ed16c-ea50-778c-8f57-d771de72dd02",
    "067ed16c-ea80-78b5-a2fd-c281808660a0",
    "067ed16c-ea70-79bb-ba82-479a0a691150",
    "067ee281-7184-7276-9b71-a797666242f2",
    "067ee281-7194-7e4d-9d50-5f366dbd0227",
    "067ee281-7164-744d-95af-37451eeb3aa4",
    "067ee281-7174-740f-b96c-c03115c2c404",
    "067f8931-1b12-7345-a15b-1ffd65d3027e",
    "067f8931-1af5-7f5a-8af8-cd8a5cd1c94c",
    "067f8931-1b04-77fe-a232-20a6a4c17281",
    "067f8931-1b1f-7b66-b026-f776b144a67d",
    "067f8931-1ba9-7862-8e84-bda43e089598",
    "067f8931-1b9c-7034-891f-80e664a19afa",
    "067f8931-1bc4-799d-be28-37319eb09142",
    "067f8931-1bb7-7590-beac-695ab6ac8c84",
    "067ee281-71e7-776d-bbeb-d9388f8af968",
    "067ee281-71f7-777c-b768-12c90721c6e9",
    "067ee281-7217-7427-be6d-8e2f634d779f",
    "067ee281-7207-76ad-9bb1-a2b32c2ed049",
    "067f8931-1b42-7089-9227-46e5dd0fefc2",
    "067f8931-1b34-7d11-b7e4-4fd272fe51b1",
    "067f8931-1b27-7700-a716-f712019c6a45",
    "067f8931-1b4f-77c6-bf1b-4ad79af5cb81",
    "067ed16c-eaba-7036-a325-65e748f506d0",
    "067ed16c-ea99-7753-a057-08ae7aa40c48",
    "067ed16c-eaa9-7dba-a479-a9a791468e59",
    "067ed16c-ea89-7684-88d2-29b1a948a727",
    "067feecb-2db2-7f15-83d2-2eacefd3b48b",
    "067feecb-2ddf-7167-89bc-f8887eb6a62a",
    "067feecb-2e81-7740-a7dd-4da15c794a37",
    "067feecb-2cbf-7768-b448-d6f65ba45026",
    "067ee281-714c-7ced-9a54-b9b084eae835",
    "067ee281-713c-7a1b-bc7c-38572885613b",
    "067ee281-712b-7c2e-a149-a39825246c39",
    "067ee281-715d-72de-aae3-ec50aa2c022c",
    "067ed16c-e9e3-7f3b-a58c-8183085d6551",
    "067ed16c-e9d3-71bd-9ee3-4ff5be326b99",
    "067ed16c-e9f4-74cb-93d0-642ff75a6195",
    "067ed16c-ea04-7c85-89c5-c4f855091903",
    "067feecb-31f1-709f-9550-151506c5313b",
    "067feecb-3276-7cb8-ab86-93fb7ceef1fe",
    "067feecb-3145-7c32-a68a-b2c36f31ad4f",
    "067feecb-31bb-7dce-8bc5-c2ccfe6edf1f",
    "067fcc93-2710-7b20-a42e-7bb87b0a31ba",
    "067fcc93-2733-7f84-8b36-efef1f3c37e9",
    "067fcc93-26f2-77b1-9f9d-3dcba2db8a13",
    "067fcc93-26ce-7fb8-96f8-f8b7e9b0eaad",
]
