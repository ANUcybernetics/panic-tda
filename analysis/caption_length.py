#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["polars>=1.0", "numpy>=2.0"]
# ///
"""Caption length by image-to-text model.

Reproduces the tables in `backlog/docs/caption-length-by-i2t-model.md`: words
per caption per captioner, the eta-squared for model identity, p10-p90 overlap
between captioner pairs, and the share of captions cut off mid-sentence (no
terminal punctuation).

    ./analysis/caption_length.py [parquet_dir ...]

Defaults to the `balanced_panel_5x5` dump. Pass several dumps to pool them, so
the same table can be recomputed once natural-length runs exist alongside the
truncated ones.
"""

import itertools
import pathlib
import re
import sys

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).parent.parent
DIRS = [pathlib.Path(d) for d in sys.argv[1:]] or [ROOT / "019f3645_parquet"]
TERMINAL = r'[.!?]["\')\]]*\s*$'


def captions(export_dir: pathlib.Path) -> pl.DataFrame:
    """Model captions from one dump, excluding the synthetic initial prompts."""
    return (
        pl.read_parquet(
            export_dir / "invocations.parquet",
            columns=["model", "type", "sequence_number", "output_text"],
        )
        .filter((pl.col("type") == "text") & (pl.col("sequence_number") > 0))
        .with_columns(
            words=pl.col("output_text").str.extract_all(r"\S+").list.len(),
            complete=pl.col("output_text").str.strip_chars().str.contains(TERMINAL),
        )
    )


frame = pl.concat([captions(d) for d in DIRS])
print(f"{frame.height:,} captions from {', '.join(str(d) for d in DIRS)}\n")

summary = (
    frame.group_by("model")
    .agg(
        n=pl.len(),
        median=pl.col("words").median(),
        mean=pl.col("words").mean(),
        p10=pl.col("words").quantile(0.10),
        p90=pl.col("words").quantile(0.90),
        p99=pl.col("words").quantile(0.99),
        min=pl.col("words").min(),
        max=pl.col("words").max(),
        truncated=1 - pl.col("complete").mean(),
    )
    .sort("median")
)

print("| model | n | median | mean | p10-p90 | min-max | % truncated |")
print("|---|---|---|---|---|---|---|")
for row in summary.iter_rows(named=True):
    print(
        f"| {row['model']} | {row['n']:,} | {row['median']:.0f} | {row['mean']:.1f} | "
        f"{row['p10']:.0f}-{row['p90']:.0f} | {row['min']}-{row['max']} | "
        f"{row['truncated']:.1%} |"
    )

# eta-squared: the share of variance in caption length explained by model
# identity alone. At 0.9-plus, length and captioner are effectively one
# variable and no regression can attribute an effect to either separately.
words = frame["words"].to_numpy().astype(float)
grand = words.mean()
between = sum(
    len(g) * (g["words"].to_numpy().mean() - grand) ** 2
    for _, g in frame.group_by("model")
)
total = ((words - grand) ** 2).sum()
print(f"\neta-squared (model identity -> caption length) = {between / total:.3f}")
print(f"share of captions over 50 words = {(words > 50).mean():.1%}")

print("\np10-p90 overlap (0 = disjoint, 1 = identical):")
ranges = {r["model"]: (r["p10"], r["p90"]) for r in summary.iter_rows(named=True)}
for a, b in itertools.combinations(ranges, 2):
    (lo_a, hi_a), (lo_b, hi_b) = ranges[a], ranges[b]
    inter = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
    union = max(hi_a, hi_b) - min(lo_a, lo_b)
    print(f"  {a:15} vs {b:15} {inter / union if union else 1.0:.2f}")
