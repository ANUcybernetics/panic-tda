# Is the panel's data what the experiment was meant to produce?

Asked 2026-10-04 by Ben, after TASK-90's panel (experiment `01a09e21`)
completed and before any analysis was built on it. Measured with two scripts.
`analysis/panel_audit.py` reads every stored image, caption, timestamp, seed
and embedding, and the run's own log. `analysis/panel_regenerate.py` runs
stored inputs back through the production invoke path on the GPU. The numbers
are in the JSON beside each script.

## Verdict

The data is sound. Every image decodes and every step ran as one batch of 40.
Stored captions and seeds regenerate the stored images in a fresh process,
three weeks and three reboots after some of them were made. Nothing in the
96,000 images or 96,000 captions is a failure of the machinery.

What the audit did turn up belongs in the methods section, since each item is
part of what a generator or captioner read:

- 346 of the 95,360 captions a generator read (0.36%) ran past its 512-token
  encoder and were cut, nearly all of them Gemma4's
- ten Moondream3 captions are repetition loops cut off at the 1,024-token
  generation ceiling
- Qwen25VL switches into Chinese part-way through a caption now and then
- Gemma4's captions are markdown, some of them addressed to the reader, and
  four times it declined to describe an image it called too dark
- the GPU was shared with other jobs on 28 and 29 September, at the cost of
  two retried steps and no change to any output

Three of the 640 runs reach a flat field of one colour. Seven spend a third or
more of their length on images with almost nothing in them. Both are the
loop's own behaviour, described below so that neither is mistaken for a fault.

## Stored outputs reproduce from stored inputs

This is the direct test that the database holds what the pipeline says it
holds. A caption, a seed and an image should be paired as recorded, and the
seed should be what varies the image.

| What was run again | Result |
| --- | --- |
| 38 images in 11 batches, all four generators, from the stored caption and seed | mean pixel difference from the stored image 0.1--4.7 on the 0--255 scale (median 2.2) |
| the same captions at a different seed (30 images) | 23--87 (median 47), apart from one all-black image that is black at either seed |
| 200 stored images in five batches of 40, all four captioners | 200 of 200 captions identical to the stored ones |
| 600 stored captions, one whole run per captioner | every embedding matches the stored vector (cosine 1.000000) |

The few levels of difference at the same seed are the storage format. Images
are kept as AVIF at quality 50, and it is that stored image the captioner
reads. The regenerated image has not been through the encoder. The compression
is a fixed property of the pipeline, the same for every step.

The batches sit either side of each restart, on a retried step, and on the
step where a run first went black. All of them reproduce.

## Images

All 96,000 decode to 1024 by 1024 RGB, and no two are byte-identical. Call an
image flat when its brightness barely varies (a standard deviation under 2
levels). There are 224 of them, in three runs:

| Network | Prompt | Flat from step | Flat images | Colour |
| --- | --- | --- | --- | --- |
| Flux2Dev + Gemma4 | a glass bottle beside a candle | 62 | 119 | black |
| SD35Medium + Gemma4 | a market stall displaying fruit, flowers, and handmade crafts | 62 | 74 | lime green |
| SD35Medium + Gemma4 | a city slowly turning into a forest | 102 | 31 | lime green |

A pure black image is also what a numerical fault in the generator decodes to,
so each of these was traced. All three are gradual and the captions track them.
The bottle-and-candle run darkens over twenty steps, its mean brightness
falling 33, 16, 9, 4, 3 on a scale of 255. A candlelit room becomes a
mantelpiece and then a strip of red indicator lights. At that point Gemma4
writes "I'm sorry, but the image you provided is almost entirely black".
Flux2Dev, given that sentence as a prompt, draws black. The market stall
becomes broccoli, then green spheres, then a green texture, then a field that
Gemma4 describes as "a solid, vibrant lime green".

No flat image follows a caption that does not ask for one. The first black
image regenerates as black from its stored caption, at the stored seed and at
a different one.

Flat images are the extreme case of a wider group. An image that compresses
under 5 kB has almost nothing in it. There are 1,184 of those, and seven runs
have fifty or more:

| Network | Prompt | Low-detail images | What it settles on |
| --- | --- | --- | --- |
| Flux2Dev + Moondream3 | a city slowly turning into a forest | 148 | a white-to-black gradient |
| Flux2Dev + Gemma4 | a city slowly turning into a forest | 140 | a white-to-black gradient |
| Flux2Dev + Gemma4 | a glass bottle beside a candle | 126 | black |
| Flux2Dev + Moondream3 | a bustling Tokyo street at night | 125 | a black half and a red half |
| SD35Medium + Gemma4 | a market stall displaying fruit, flowers, and handmade crafts | 97 | lime green |
| Flux2Dev + Gemma4 | a library with impossible architecture | 84 | a red star on black |
| SD35Medium + Gemma4 | the feeling of nostalgia | 56 | a figure in fog, which it later leaves |

The first four end there: at least nine-tenths of their last fifty images are
low-detail. Five of the seven are Flux2Dev runs and five are Gemma4 runs.

Nothing else looks wrong. The roughest images by pixel-to-pixel difference are
mostly grass, bare trees and forest. Contact sheets of about 300 images show
well-formed pictures throughout. They cover one random run per network at ten
steps, the runs in the two tables above, and the final frame of every
Flux2Dev + Gemma4 run.

## Captions

None is empty. Lengths differ by captioner as decision-01 intended: a median
of 214 words for Gemma4, 177 for JoyCaption, 87 for Qwen25VL and 50 for
Moondream3.

### Cut at the generation ceiling

Counted under each captioner's own tokenizer, no Qwen25VL, Gemma4 or JoyCaption
caption reaches 1,024 tokens (the longest are 431, 566 and 335). Moondream3
does not expose its tokenizer. Ten of its captions, in four runs, stop
mid-phrase, and all ten are repetition loops. The model starts transcribing
text in the image and then repeats it, for up to 773 words, until the ceiling
stops it: "POLICIA POLICIA POLICIA", or "Reading for Life," over and over.
Three more captions repeat a phrase many times before ending on their own. In
every case the run is back to an ordinary caption one or two captions later.

When the run finished, `mix experiment.status` reported 13.1% of Gemma4
captions as truncated. That figure was wrong. Its check wanted a caption to end
in terminal punctuation, and Gemma4's end in `.**`, the markdown bold closing
after the full stop. The check now allows closing marks after the punctuation
and agrees with this audit: one Gemma4 caption in 24,000 lacks terminal
punctuation, and it ends on a bullet point.

### Cut at the generator's encoder

Every generator reads only the first 512 tokens of a caption, counted as its
own pipeline counts them. The rest is cut without a warning (TASK-100).

| Network | Captions read | Over 512 | Median tokens cut |
| --- | --- | --- | --- |
| SD35Medium + Gemma4 | 5,960 | 209 (3.5%) | 30 |
| Flux2Dev + Gemma4 | 5,960 | 125 (2.1%) | 23 |
| Flux2Klein + Moondream3 | 5,960 | 5 (0.1%) | 696 |
| SD35Medium + Moondream3 | 5,960 | 4 (0.1%) | 202 |
| ZImageTurbo + Gemma4 | 5,960 | 3 (0.1%) | 47 |
| the other eleven | 65,560 | 0 | |

What Gemma4 loses is the last twenty to thirty tokens of a caption that runs a
little long, which is typically its closing summary sentence. The Moondream3
rows are the repetition loops. For Gemma4 the cut is a little more common in
the second half of a run than the first: 3.7% against 3.3% with SD35Medium,
2.4% against 1.8% with Flux2Dev.

### What the captioners do that a reader might not expect

All of it follows from giving every captioner the same bare instruction,
"Describe this image.", and none of it is a fault:

- Gemma4 answers in markdown (22,939 of its 24,000 captions contain bold
  markers), so the generators are prompted with headings and bullet points
- Gemma4 opens 117 captions by addressing the reader ("The image you
  provided..."), and in four more, all in the run that went black, it declines
  to describe the image
- Qwen25VL has Chinese in 43 captions, eight of them with ten characters or
  more, where it switches language mid-sentence
- two Moondream3 captions end in a replacement character, where the ceiling
  cut a multi-byte character in half

### Pairing

The first caption of a run should still be about its prompt. For eighteen of
the twenty prompts, at least 29 of the 32 first captions mention a content
word from it. The two exceptions are "the feeling of
nostalgia" and "two friends talking at a cafe". The first has no object to
name. For the second the check is at fault: the captions write "café" with its
accent and call the friends a couple or two people.

## Process

| Check | Result |
| --- | --- |
| steps | 4,800, each a single batch of 40 with one start time |
| invocations starting before their input completed | 0 |
| seeds | 96,000, all recorded, 32-bit, one value used twice (1.07 expected) |
| embeddings | 96,000, unit length, none empty; identical captions embed to cosine 0.9996 or better |
| code under `lib/`, `priv/` or `config/` changed after launch | no |
| model environment rebuilt during the run | no (last modified 3 September) |

The run resumed five times and none of them split a step:

- 14 September, stopped to move the database to `/data`
- 15 September, the machine went down at 07:30 with no shutdown logged and
  came back at 09:30
- 22 September, the out-of-memory killer took the unit (TASK-90)
- 23 September and 1 October, reboots after kernel upgrades (6.8.0-139 to 142
  to 146); `dpkg.log` shows no upgrade of the NVIDIA driver packages in the
  window

Two steps were retried, both Flux2Dev in the Gemma4 cell: step 118 on 28
September and step 186 on 29 September. Each failed with CUDA out of memory,
the error listing half a dozen other processes holding about 4.3 GiB each on
the same GPU. Each succeeded on the second attempt. Seeds are drawn before the
retry loop, so the retry used the seeds that were then stored. Step 118 is one
of the batches that regenerates. A Flux2Dev step took a median of 31 minutes;
the two retried ones took 53 and 73.

The log also carries 307 allocator warnings, where an allocation failed, the
cache was freed and the retry succeeded. Almost all come one per step in two
cells: Flux2Klein with Qwen25VL and Flux2Klein with JoyCaption. In those cells
the two models are evidently tight on memory together at a batch of 40. A
batch from each is among those that regenerate.

## What the loop does that is worth knowing

Two things the audit measured are dynamics, recorded here because the analysis
will meet them:

- the Flux2Dev + Gemma4 cell is much darker than any other (mean brightness 81
  falling to 75, against 98--150 elsewhere), and most of its final frames are
  high-contrast scenes in red and black
- SD35Medium's images grow more saturated over a run with every captioner (a
  mean channel spread of 56--69 in the first hundred steps, 78--95 in the last)

## What this does not check

The audit does not check that the models are the right ones (they are pinned,
TASK-84) or that the embedding model measures what the analysis wants. Nor
does it look at the content of an image beyond whether it is flat, low-detail
or noise.
