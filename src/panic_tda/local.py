from sqlmodel import Session, select

from panic_tda.export import export_video
from panic_tda.schemas import Run


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

    # Import Run model
    from panic_tda.schemas import Run

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
                .limit(3)
                .all()
            )

            # Extract run IDs and add to selected_ids
            run_ids = [str(run.id) for run in matching_runs]
            selected_ids.extend(run_ids)


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
