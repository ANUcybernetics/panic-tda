"""Main module for paper charts - coordinates generation of charts for publications."""

from sqlmodel import Session

from panic_tda.data_prep import cache_dfs

# Import specific chart modules - hardcode the current one as needed
from panic_tda.local_modules.cybernetics_26 import cybernetics_26_charts


def paper_charts(session: Session) -> None:
    """
    Generate charts for paper publications.
    """

    # cache_dfs(
    #     session,
    #     runs=True,
    #     embeddings=True,
    #     invocations=True,
    #     persistence_diagrams=True,
    #     clusters=True,
    # )

    artificial_futures_slides_charts(session)
