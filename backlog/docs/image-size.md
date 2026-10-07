# Image compressed size as a second instrument

Written 2026-10-07 for TASK-108, on the first finished cell of the long
follow-up (experiment `01a10613`, Flux2Klein + Gemma4: five prompts, eight runs
per prompt, 1,000 images per run). Measured with `analysis/image_size.py`; the
numbers are in `analysis/image_size.json`.

Every image is stored as AVIF, so its byte length is already in the database.
It says how much detail the encoder had to keep, and it says so without passing
through the captioner. Everything else we measure is a caption embedding. The
question was whether the byte length is worth keeping, and for what.

## What was measured

All 40,000 images are 1024 by 1024 and were encoded the same way: libvips
`heifsave` to AV1 at quality 50, every other setting left at its default
(`lib/panic_tda/models/image_converter.ex`, unchanged since February 2026).
Sizes run from 1.2 kB to 375 kB with a median of 46 kB. Because the spread is
multiplicative, everything below is about the logarithm of the byte length.

## Whose it is

Over the whole run the variance splits almost evenly three ways: 37% between
prompts, 30% between the runs of one prompt, 33% within a run. The typical image
is 27 kB for the doorway prompt, 30 kB for the apple, 60 kB for the
firefighters, 61 kB for the train station and 77 kB for the city.

The split moves with time. In the first quarter of the run the prompt accounts
for 53% and the runs of a prompt for 29%. By the second quarter those have
swapped, at 35% and 47%, and they stay close to that through the last quarter
(34% and 39%). So runs of one prompt start together in size and separate, which
is the same picture `drift-and-memory.md` draws from the captions.

## How it moves within a run

Size is persistent and it does not settle. Half the mean squared difference
between two images of a run grows with their separation the whole way out: 0.022
at one image apart, 0.046 at ten, 0.101 at a hundred, 0.197 at five hundred. The
last figure is above the within-run variance (0.159), so a run is still
travelling in size after 500 images.

The seed contributes little of this. Read the way `drift_memory.py` reads
captions, the part of a step that the next step gives back is 0.015, a tenth of
the within-run variance. With each run's mean removed, the autocorrelation is
0.86 at one image, 0.69 at ten, 0.31 at a hundred and gone by two hundred.

The first look at this cell, at 330 images per run, found a doorway run sitting
in a white corridor at under 15 kB and called it settled. It was not: over the
full run 60% of its images are that small, but its last quarter is back at a
typical 30 kB. A run that looks absorbed at one horizon can leave at a longer
one.

## What it shares with the captions

Image by image, the two share nothing. Within a run, the size of the step in byte length has
a rank correlation of 0.03 with the distance between the two captions (median
over runs, range -0.10 to 0.20). The largest 5% of caption steps come with
size steps of ordinary size, and the reverse holds too.

Over longer stretches the two agree. Averaged in blocks of 25 images, the step
in size between neighbouring blocks has a rank correlation of 0.45 with the
distance between the blocks' mean captions (median over runs, range 0.01 to
0.63). A run that moves somewhere new in caption space usually changes how much
detail its images carry.

The level is partly readable from the caption. A ridge regression from the
caption embedding to the image's size, scored on runs it was not fitted to,
explains 55% of the variance. It gets the run means nearly right (correlation
0.85) and the movement within a run less so (0.52).

## Recommendation

Keep it, as a cheap second instrument for the slow movement of a run, and not
as a detector of single-step jumps. It costs one SQL column, the seed accounts for a tenth of its
within-run variance, and its picture of prompts and runs separating
matches the captions' picture without using the captioner. About half of it is
predictable from the caption, so it is a partial cross-check and not an
independent coordinate.

Three cautions apply. This is one cell with five prompts, so the shares
above carry no intervals. The comparison across generators has to wait for the
SD35Medium and ZImageTurbo cells, the first of which is due about 11 October
2026; each generator's house style will shift the level. And size measures
texture, so two images of different things can weigh the same.
