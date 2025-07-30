"""Main module for paper charts - coordinates generation of charts for publications."""

from sqlmodel import Session

from panic_tda.data_prep import cache_dfs

# Import specific chart modules - hardcode the current one as needed
from panic_tda.local_modules.artificial_futures import artificial_futures_slides_charts


def paper_charts(session: Session) -> None:
    """
    Generate charts for paper publications.
    """

    cache_dfs(
        session,
        runs=False,
        embeddings=False,
        invocations=False,
        persistence_diagrams=False,
        clusters=True,
    )

    artificial_futures_slides_charts(session)
