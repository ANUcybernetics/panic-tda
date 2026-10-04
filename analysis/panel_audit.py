#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars>=1.0",
#   "numpy>=2.0",
#   "pillow>=11.3",
# ]
# ///
"""Is an experiment's stored data what the experiment was meant to produce?

A pass over every image, caption, embedding and timestamp of one experiment,
looking for artefacts of the machinery rather than of the loop: images that
failed to generate, captions cut at a ceiling or stuck in a repetition loop,
template leakage, steps split by a crash, embeddings that came back empty.
Nothing here judges the dynamics; a run that drifts into an all-black image by
way of darker and darker captions is the loop behaving, and is counted so it
can be told apart from an image that went black under an ordinary caption.

With `--log`, the run's own log is read for what the database cannot show:
resumes, retried steps and anything logged above debug level.

Every image is decoded, which takes a few minutes on all cores, so the
per-image table is cached in the export directory (untracked) and reused.

    ./analysis/panel_audit.py 01a09e21 --cache 01a09e21_parquet --log logs/long-run.log

Results -> analysis/panel_audit.json, summary to stdout.
"""

import argparse
import hashlib
import io
import json
import multiprocessing
import pathlib
import re
import sqlite3
import zlib
from collections import Counter

import numpy as np
import polars as pl
from PIL import Image

HERE = pathlib.Path(__file__).parent
OUT = pathlib.Path(__file__).with_suffix(".json")
DB = (HERE.parent / "priv" / "panic_tda_dev.db").resolve()

# An image whose luma barely varies is a flat field: all black, all white, one
# colour. Pure black (every pixel zero) is also what a NaN in the latents
# decodes to, so those are checked against the caption that produced them.
FLAT_STD = 2.0
DARK_MEAN = 8.0
BRIGHT_MEAN = 247.0
# What a caption says when it is asking for a flat or near-empty image.
FLAT_WORDS = re.compile(
    r"\b(black|dark|darkness|blank|solid|empty|white|plain|uniform|void|"
    r"featureless|monochrom\w*|nothing|fog|foggy|mist|misty|haze|hazy|"
    r"minimalist|gradient|colou?r field|single colou?r)\b",
    re.IGNORECASE,
)

# Closing punctuation a finished caption can end on, allowing for markdown
# emphasis and quotes closing after the full stop.
TERMINATED = re.compile(r"[.!?][\s*_\"'”’)\]`]*$")
SPECIAL = re.compile(
    r"<\|[^|>\n]{0,40}\|>|</?s>|<eos>|<bos>|<pad>|<unk>|</?start_of_turn>|"
    r"</?end_of_turn>|\[/?INST\]|<image>|<img>|addCriterion|�"
)
REFUSAL = re.compile(
    r"\b(I'm sorry|I am sorry|I cannot|I can't|I can not|I'm unable|"
    r"I am unable|As an AI|cannot assist|can't assist)\b",
    re.IGNORECASE,
)
NON_LATIN = re.compile(
    r"[Ͱ-ϿЀ-ӿ֐-ۿऀ-෿฀-໿"
    r"぀-ヿ㐀-鿿가-힯]"
)
WORD = re.compile(r"[A-Za-z']+")
STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "on",
        "in",
        "at",
        "to",
        "with",
        "and",
        "or",
        "is",
        "are",
        "by",
        "from",
        "under",
        "beside",
        "through",
        "into",
        "small",
        "filled",
        "slowly",
        "turning",
        "moving",
        "carrying",
        "displaying",
        "responding",
        "approaching",
        "sitting",
        "standing",
        "leaning",
        "reading",
        "talking",
        "two",
    }
)
# A caption this repetitive is a decoding loop, not a description.
LOOP_MIN_WORDS = 60
LOOP_UNIQUE_4GRAMS = 0.5
STALL_FACTOR = 5.0


def connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def image_rows(run_id: str) -> list[tuple]:
    """Decode every image of one run and measure it."""
    rows = []
    with connect() as db:
        cursor = db.execute(
            "select id, sequence_number, output_image from invocations "
            "where run_id = ? and type = 'image' order by sequence_number",
            (run_id,),
        )
        for inv_id, sn, blob in cursor:
            digest = hashlib.sha1(blob).hexdigest()
            try:
                image = Image.open(io.BytesIO(blob))
                mode = image.mode
                rgb = np.asarray(image.convert("RGB"))
            except Exception as error:  # noqa: BLE001 -- a decode failure is a finding
                rows.append(
                    (inv_id, run_id, sn, len(blob), digest, repr(error)) + (None,) * 15
                )
                continue
            luma = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
            spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2)
            thumb = image.convert("L").resize((16, 16), Image.Resampling.BOX)
            rows.append(
                (
                    inv_id,
                    run_id,
                    sn,
                    len(blob),
                    digest,
                    None,
                    rgb.shape[1],
                    rgb.shape[0],
                    mode,
                    float(luma.mean()),
                    float(luma.std()),
                    float(rgb[..., 0].mean()),
                    float(rgb[..., 1].mean()),
                    float(rgb[..., 2].mean()),
                    int(rgb.min()),
                    int(rgb.max()),
                    float((luma < 16).mean()),
                    float((luma > 239).mean()),
                    # mean absolute difference between neighbouring pixels: high for
                    # an image that is noise, near zero for a flat one
                    float(
                        (
                            np.abs(np.diff(luma, axis=0)).mean()
                            + np.abs(np.diff(luma, axis=1)).mean()
                        )
                        / 2
                    ),
                    float(spread.mean()),
                    thumb.tobytes(),
                )
            )
    return rows


IMAGE_SCHEMA = {
    "id": pl.String,
    "run_id": pl.String,
    "sn": pl.Int64,
    "bytes": pl.Int64,
    "sha1": pl.String,
    "decode_error": pl.String,
    "width": pl.Int64,
    "height": pl.Int64,
    "mode": pl.String,
    "mean": pl.Float64,
    "std": pl.Float64,
    "r": pl.Float64,
    "g": pl.Float64,
    "b": pl.Float64,
    "min": pl.Int64,
    "max": pl.Int64,
    "dark_share": pl.Float64,
    "bright_share": pl.Float64,
    "roughness": pl.Float64,
    "saturation": pl.Float64,
    "thumb": pl.Binary,
}


def image_table(run_ids: list[str], cache: pathlib.Path | None) -> pl.DataFrame:
    path = cache / "image_stats.parquet" if cache else None
    if path and path.exists():
        return pl.read_parquet(path)
    with multiprocessing.Pool() as pool:
        rows = [r for chunk in pool.imap_unordered(image_rows, run_ids) for r in chunk]
    table = pl.DataFrame(rows, schema=IMAGE_SCHEMA, orient="row").sort("run_id", "sn")
    if path:
        table.write_parquet(path)
    return table


def load(experiment: str) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, np.ndarray]]:
    with connect() as db:
        runs = pl.DataFrame(
            db.execute(
                "select id, network, initial_prompt, run_number from runs "
                "where experiment_id like ? order by id",
                (experiment + "%",),
            ).fetchall(),
            schema=["run_id", "network", "prompt", "run_number"],
            orient="row",
        )
        invocations = pl.DataFrame(
            db.execute(
                "select i.id, i.run_id, i.sequence_number, i.type, i.model, i.seed, "
                "i.started_at, i.completed_at, i.output_text from invocations i "
                "join runs r on r.id = i.run_id where r.experiment_id like ?",
                (experiment + "%",),
            ).fetchall(),
            schema=[
                "id",
                "run_id",
                "sn",
                "type",
                "model",
                "seed",
                "started_at",
                "completed_at",
                "text",
            ],
            orient="row",
        ).with_columns(
            pl.col("started_at").str.to_datetime(time_zone="UTC"),
            pl.col("completed_at").str.to_datetime(time_zone="UTC"),
        )
        vectors = {
            inv_id: np.frombuffer(blob, dtype=np.float32)
            for inv_id, blob in db.execute(
                "select e.invocation_id, e.vector from embeddings e "
                "join invocations i on i.id = e.invocation_id "
                "join runs r on r.id = i.run_id where r.experiment_id like ?",
                (experiment + "%",),
            )
        }
    return runs, invocations.join(runs, on="run_id"), vectors


def audit_images(images: pl.DataFrame, inv: pl.DataFrame) -> dict:
    captions = inv.filter(pl.col("type") == "text").select("run_id", "sn", "text")
    images = (
        images.join(inv.select("id", "network", "model", "prompt"), on="id")
        .join(
            captions.with_columns(pl.col("sn") + 1).rename({"text": "caption_in"}),
            on=["run_id", "sn"],
            how="left",
        )
        .with_columns(pl.coalesce("caption_in", "prompt").alias("caption_in"))
    )
    ok = images.filter(pl.col("decode_error").is_null())

    flat = ok.filter(pl.col("std") < FLAT_STD).with_columns(
        pl.when(pl.col("mean") < DARK_MEAN)
        .then(pl.lit("black"))
        .when(pl.col("mean") > BRIGHT_MEAN)
        .then(pl.lit("white"))
        .otherwise(pl.lit("colour"))
        .alias("kind"),
        pl.col("caption_in")
        .map_elements(lambda c: bool(FLAT_WORDS.search(c)), return_dtype=pl.Boolean)
        .alias("caption_asks_for_it"),
    )
    pure_black = ok.filter(pl.col("max") == 0)
    # The first flat image of each run, with the caption that drew it: where a
    # numerical fault would show up as a flat image under an ordinary caption.
    entries = (
        flat.sort("run_id", "sn")
        .group_by("run_id", maintain_order=True)
        .first()
        .select("network", "prompt", "run_id", "id", "sn", "kind", "mean", "caption_in")
        .sort("network", "sn")
    )
    duplicates = (
        ok.group_by("sha1")
        .agg(
            pl.len().alias("n"),
            pl.col("std").max().alias("std"),
            pl.col("run_id").n_unique().alias("runs"),
        )
        .filter(pl.col("n") > 1)
    )
    by_model = (
        ok.group_by("model")
        .agg(
            pl.len().alias("images"),
            pl.col("mean").median().alias("luma_median"),
            (pl.col("mean") < 20).mean().mul(100).alias("dark_pct"),
            (pl.col("std") < FLAT_STD).mean().mul(100).alias("flat_pct"),
            pl.col("roughness").median().alias("roughness_median"),
            pl.col("roughness").quantile(0.999).alias("roughness_p999"),
            pl.col("roughness").max().alias("roughness_max"),
            pl.col("bytes").median().alias("bytes_median"),
        )
        .sort("model")
    )
    thirds = (
        ok.with_columns((pl.col("sn") // 100).alias("third"))
        .group_by("network", "third")
        .agg(
            pl.col("mean").mean().alias("luma"),
            pl.col("saturation").mean().alias("saturation"),
        )
        .sort("network", "third")
    )
    return {
        "images": images.height,
        "decode_failures": images.height - ok.height,
        "dimensions": {
            f"{w}x{h}": n
            for w, h, n in ok.group_by("width", "height").len().iter_rows()
        },
        "modes": dict(ok.group_by("mode").len().iter_rows()),
        "flat": {
            "threshold_luma_std": FLAT_STD,
            "total": flat.height,
            "runs": flat["run_id"].n_unique(),
            "by_kind": dict(flat.group_by("kind").len().iter_rows()),
            "by_network": dict(
                flat.group_by("network").len().sort("network").iter_rows()
            ),
            "caption_does_not_ask_for_it": flat.filter(~pl.col("caption_asks_for_it"))
            .select("network", "id", "sn", "kind", "caption_in")
            .with_columns(pl.col("caption_in").str.slice(0, 300))
            .to_dicts(),
        },
        "pure_black": {
            "total": pure_black.height,
            "runs": pure_black["run_id"].n_unique(),
            "by_network": dict(
                pure_black.group_by("network").len().sort("network").iter_rows()
            ),
        },
        "first_flat_image_per_run": entries.with_columns(
            pl.col("caption_in").str.slice(0, 300)
        ).to_dicts(),
        "byte_identical": {
            "groups": duplicates.height,
            "images": int(duplicates["n"].sum()),
            "groups_that_are_not_flat": duplicates.filter(
                pl.col("std") >= FLAT_STD
            ).height,
            "groups_spanning_runs": duplicates.filter(pl.col("runs") > 1).height,
        },
        "by_model": by_model.to_dicts(),
        "luma_and_saturation_by_hundred_steps": thirds.to_dicts(),
    }


def loopiness(text: str) -> float:
    """Share of a caption's word 4-grams that are distinct: low means it is looping."""
    words = WORD.findall(text.lower())
    grams = list(zip(words, words[1:], words[2:], words[3:]))
    return len(set(grams)) / len(grams) if grams else 1.0


def audit_captions(inv: pl.DataFrame) -> dict:
    text = inv.filter(pl.col("type") == "text").with_columns(
        pl.col("text").str.len_chars().alias("chars"),
        pl.col("text").str.split(" ").list.len().alias("words"),
        pl.col("text")
        .map_elements(lambda c: bool(TERMINATED.search(c)), return_dtype=pl.Boolean)
        .alias("terminated"),
        pl.col("text")
        .map_elements(loopiness, return_dtype=pl.Float64)
        .alias("unique_4grams"),
        pl.col("text")
        .map_elements(
            lambda c: len(zlib.compress(c.encode())) / max(len(c.encode()), 1),
            return_dtype=pl.Float64,
        )
        .alias("compression"),
        pl.col("text")
        .map_elements(lambda c: SPECIAL.findall(c), return_dtype=pl.List(pl.String))
        .alias("special"),
        pl.col("text")
        .map_elements(lambda c: bool(REFUSAL.search(c)), return_dtype=pl.Boolean)
        .alias("refusal"),
        pl.col("text")
        .map_elements(lambda c: len(NON_LATIN.findall(c)), return_dtype=pl.Int64)
        .alias("non_latin"),
    )
    looping = text.filter(
        (pl.col("words") >= LOOP_MIN_WORDS)
        & (pl.col("unique_4grams") < LOOP_UNIQUE_4GRAMS)
    )

    def sample(frame: pl.DataFrame, n: int = 6, tail: bool = False) -> list[dict]:
        cut = (
            pl.col("text").str.slice(-160) if tail else pl.col("text").str.slice(0, 220)
        )
        return (
            frame.sort("id")
            .head(n)
            .select("network", "id", "sn", "words", cut.alias("text"))
            .to_dicts()
        )

    # The first caption of a run should still be about its prompt: a check that
    # prompts, images and captions are paired the way the database says.
    first = text.filter(pl.col("sn") == 1)
    keyed = [
        (
            prompt,
            {w for w in WORD.findall(prompt.lower()) if w not in STOP and len(w) > 3},
            caption.lower(),
        )
        for prompt, caption in first.select("prompt", "text").iter_rows()
    ]
    on_prompt = Counter()
    for prompt, keys, caption in keyed:
        on_prompt[prompt, any(k[:5] in caption for k in keys)] += 1

    repeated = (
        text.group_by("network", "text")
        .agg(pl.len().alias("n"), pl.col("run_id").n_unique().alias("runs"))
        .filter(pl.col("runs") > 1)
    )
    by_model = (
        text.group_by("model")
        .agg(
            pl.len().alias("captions"),
            pl.col("words").median().alias("words_median"),
            pl.col("words").quantile(0.99).alias("words_p99"),
            pl.col("words").max().alias("words_max"),
            pl.col("chars").max().alias("chars_max"),
            pl.col("chars").min().alias("chars_min"),
            (~pl.col("terminated")).sum().alias("unterminated"),
            (pl.col("unique_4grams") < LOOP_UNIQUE_4GRAMS)
            .filter(pl.col("words") >= LOOP_MIN_WORDS)
            .sum()
            .alias("looping"),
            (pl.col("special").list.len() > 0).sum().alias("special_tokens"),
            pl.col("refusal").sum().alias("refusals"),
            (pl.col("non_latin") > 0).sum().alias("with_non_latin_script"),
            pl.col("text")
            .str.starts_with("The image you provided")
            .sum()
            .alias("addresses_the_user"),
            pl.col("text").str.contains(r"\*\*").sum().alias("with_markdown_bold"),
        )
        .sort("model")
    )
    return {
        "captions": text.height,
        "empty_placeholder": text.filter(pl.col("text") == "[empty]").height,
        "by_model": by_model.to_dicts(),
        "unterminated_examples": sample(text.filter(~pl.col("terminated")), tail=True),
        "longest": sample(text.sort("words", descending=True).head(6), tail=True),
        "looping": {
            "rule": f">= {LOOP_MIN_WORDS} words and < {LOOP_UNIQUE_4GRAMS} distinct 4-grams",
            "total": looping.height,
            "runs": looping["run_id"].n_unique(),
            "by_network": dict(
                looping.group_by("network").len().sort("network").iter_rows()
            ),
            "examples": sample(looping, tail=True),
        },
        "special_tokens": dict(Counter(t for row in text["special"] for t in row)),
        "special_token_examples": sample(text.filter(pl.col("special").list.len() > 0)),
        "refusal_examples": sample(text.filter(pl.col("refusal"))),
        "non_latin_examples": sample(
            text.filter(pl.col("non_latin") > 0).sort("non_latin", descending=True)
        ),
        "same_caption_in_different_runs": {
            "distinct_captions": repeated.height,
            "occurrences": int(repeated["n"].sum()),
            "by_network": dict(
                repeated.group_by("network")
                .agg(pl.col("n").sum())
                .sort("network")
                .iter_rows()
            ),
            "most_common": repeated.sort("n", descending=True)
            .head(8)
            .with_columns(pl.col("text").str.slice(0, 160))
            .to_dicts(),
        },
        "first_caption_mentions_its_prompt": [
            {
                "prompt": prompt,
                "mentions": on_prompt[prompt, True],
                "does_not": on_prompt[prompt, False],
            }
            for prompt in sorted({p for p, _, _ in keyed})
        ],
    }


def audit_process(inv: pl.DataFrame) -> dict:
    steps = (
        inv.group_by("network", "sn", "model")
        .agg(
            pl.len().alias("items"),
            pl.col("started_at").n_unique().alias("batches"),
            pl.col("started_at").min().alias("started_at"),
            pl.col("completed_at").max().alias("completed_at"),
        )
        .with_columns(
            (pl.col("completed_at") - pl.col("started_at"))
            .dt.total_seconds()
            .alias("seconds")
        )
    )
    typical = steps.group_by("model").agg(pl.col("seconds").median().alias("median"))
    stalls = steps.join(typical, on="model").filter(
        pl.col("seconds") > STALL_FACTOR * pl.col("median")
    )
    order = inv.sort("run_id", "sn").with_columns(
        pl.col("completed_at").shift(1).over("run_id").alias("previous_completed")
    )
    seeds = inv.filter(pl.col("type") == "image")["seed"]
    repeated_seeds = seeds.value_counts().filter(pl.col("count") > 1)
    bins = np.bincount((seeds.to_numpy() >> 28).astype(np.int64), minlength=16)
    return {
        "steps": steps.height,
        "steps_not_a_single_batch_of_40": steps.filter(
            (pl.col("items") != 40) | (pl.col("batches") != 1)
        )
        .select("network", "sn", "items", "batches")
        .to_dicts(),
        "invocations_started_before_their_input_completed": order.filter(
            pl.col("started_at") < pl.col("previous_completed")
        ).height,
        "step_seconds_by_model": steps.group_by("model")
        .agg(
            pl.col("seconds").median().alias("median"),
            pl.col("seconds").quantile(0.99).alias("p99"),
            pl.col("seconds").max().alias("max"),
        )
        .sort("model")
        .to_dicts(),
        "stalled_steps": stalls.sort("started_at")
        .select(
            "network",
            "sn",
            pl.col("started_at").dt.strftime("%Y-%m-%d %H:%M"),
            "seconds",
            "median",
        )
        .to_dicts(),
        "seeds": {
            "images": seeds.len(),
            "missing": seeds.null_count(),
            "min": int(seeds.min()),
            "max": int(seeds.max()),
            "values_used_more_than_once": repeated_seeds.height,
            "expected_repeats_for_32_bit_seeds": seeds.len() ** 2 / 2 / 2**32,
            "sixteenths_min_max": [int(bins.min()), int(bins.max())],
        },
    }


def audit_embeddings(inv: pl.DataFrame, vectors: dict[str, np.ndarray]) -> dict:
    text = inv.filter(pl.col("type") == "text")
    ids = text["id"].to_list()
    missing = [i for i in ids if i not in vectors]
    matrix = np.stack([vectors[i] for i in ids if i in vectors])
    norms = np.linalg.norm(matrix, axis=1)
    # The same caption embedded twice should give the same vector; how far apart
    # the copies sit is the embedding stage's own noise.
    index = {inv_id: n for n, inv_id in enumerate(i for i in ids if i in vectors)}
    worst, pairs = 1.0, 0
    for group in (
        text.group_by("text")
        .agg(pl.col("id"))
        .filter(pl.col("id").list.len() > 1)["id"]
    ):
        rows = matrix[[index[i] for i in group if i in index]]
        cosines = rows @ rows.T
        worst = min(worst, float(cosines.min()))
        pairs += len(rows) * (len(rows) - 1) // 2
    return {
        "captions": len(ids),
        "without_embedding": len(missing),
        "dimension": int(matrix.shape[1]),
        "non_finite_values": int((~np.isfinite(matrix)).sum()),
        "zero_vectors": int((norms == 0).sum()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "identical_caption_pairs": pairs,
        "identical_caption_min_cosine": worst,
    }


LOG_LEVEL = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d+ \[(warning|error|notice)\] (.*)")
LOG_DATE = re.compile(r"~U\[(\d{4}-\d{2}-\d{2}) ")


def audit_log(path: pathlib.Path) -> dict:
    """What the run logged about itself: resumes, retries, allocator pressure."""
    messages, resumes, tracebacks = [], 0, 0
    allocator: Counter[str] = Counter()
    date = None
    with path.open(errors="replace") as lines:
        for line in lines:
            if found := LOG_DATE.search(line):
                date = found.group(1)
            if "Resuming experiment" in line:
                resumes += 1
            elif "Traceback (most recent call last)" in line:
                tracebacks += 1
            elif "CUDACachingAllocator" in line:
                # an allocation that failed, was retried after freeing the cache
                # and succeeded: memory pressure, not an error
                allocator[date] += 1
            elif found := LOG_LEVEL.match(line):
                messages.append(
                    {
                        "near": date,
                        "level": found.group(1),
                        "message": found.group(2)[:700],
                    }
                )
    return {
        "resumes": resumes,
        "tracebacks": tracebacks,
        "above_debug": messages,
        "retried_steps": [m for m in messages if "[retry]" in m["message"]],
        "allocator_warnings_by_date": dict(sorted(allocator.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", help="experiment id prefix")
    parser.add_argument(
        "--cache", type=pathlib.Path, help="directory for the per-image table"
    )
    parser.add_argument("--log", type=pathlib.Path, help="the run's log file")
    args = parser.parse_args()

    runs, inv, vectors = load(args.experiment)
    images = image_table(runs["run_id"].to_list(), args.cache)
    results = {
        "experiment": args.experiment,
        "runs": runs.height,
        "images": audit_images(images, inv),
        "captions": audit_captions(inv),
        "process": audit_process(inv),
        "embeddings": audit_embeddings(inv, vectors),
    }
    if args.log:
        results["log"] = audit_log(args.log)
    OUT.write_text(json.dumps(results, indent=2, default=str))

    i, c, p, e = (results[k] for k in ("images", "captions", "process", "embeddings"))
    print(
        f"images: {i['images']}, {i['decode_failures']} undecodable, sizes {i['dimensions']}"
    )
    print(
        f"  flat: {i['flat']['total']} in {i['flat']['runs']} runs {i['flat']['by_kind']}; "
        f"{len(i['flat']['caption_does_not_ask_for_it'])} under a caption that does not ask for one"
    )
    print(f"  pure black: {i['pure_black']['total']} in {i['pure_black']['runs']} runs")
    print(
        f"  byte-identical groups: {i['byte_identical']['groups']}, "
        f"{i['byte_identical']['groups_that_are_not_flat']} of them not flat"
    )
    print(
        f"captions: {c['captions']}, {c['empty_placeholder']} empty, {c['looping']['total']} looping"
    )
    for row in c["by_model"]:
        print(
            f"  {row['model']:<11} median {row['words_median']:.0f} words, max {row['words_max']}, "
            f"unterminated {row['unterminated']}, looping {row['looping']}, "
            f"special {row['special_tokens']}, refusals {row['refusals']}, non-latin {row['with_non_latin_script']}"
        )
    print(
        f"process: {len(p['steps_not_a_single_batch_of_40'])} split steps, "
        f"{p['invocations_started_before_their_input_completed']} out of order, "
        f"{len(p['stalled_steps'])} stalled, seeds repeated {p['seeds']['values_used_more_than_once']}"
    )
    print(
        f"embeddings: {e['without_embedding']} missing, {e['zero_vectors']} zero, "
        f"norms {e['norm_min']:.4f}-{e['norm_max']:.4f}, "
        f"identical captions min cosine {e['identical_caption_min_cosine']:.5f} over {e['identical_caption_pairs']} pairs"
    )


if __name__ == "__main__":
    main()
