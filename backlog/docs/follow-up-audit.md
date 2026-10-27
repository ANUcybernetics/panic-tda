# Is the long follow-up's data what the experiment was meant to produce?

The same audit that passed on the panel (`panel-audit.md`), run on the long
follow-up (experiment `01a10613`, TASK-104) after it completed on 24 October
2026: 240 runs of 2,000 invocations, six networks, five prompts, eight runs
each. `analysis/panel_audit.py` read every stored image, caption, timestamp,
seed and embedding, and the run's log, with results in
`analysis/follow_up_audit.json`. `analysis/panel_regenerate.py` ran stored
inputs back through the production invoke path on the GPU, with results in
`analysis/follow_up_regenerate.json`.

## Verdict

The data is sound. Every image decodes, every step ran as one batch of 40, and
no restart split a step or left a gap in a run. Stored captions and seeds
regenerate the stored images, stored images caption to the stored captions, and
stored captions embed to the stored vectors, three weeks after the first of
them were made and on a newer NVIDIA driver than most of them were made with.
Nothing in the 240,000 images or 240,000 captions is a failure of the
machinery.

What belongs in the methods section, since each is part of what a model read:

- 1,090 of the 239,760 captions a generator read (0.45%) ran past its 512-token
  encoder and were cut, 863 of them Gemma4's under SD35Medium's T5
- 83 Moondream3 captions are repetition loops, 73 of them in SD35Medium +
  Moondream3
- the run was interrupted six times (one deliberate restart, three
  out-of-memory kills, one stop, one reboot), and its last 306 steps ran on a
  newer NVIDIA driver
- one commit to `lib/` was taken up mid-run (`d2143d5`, the reads in the stages
  that follow a cell; nothing in generation or captioning)

Ten runs reach flat fields of one colour, nine of them in SD35Medium +
Moondream3, and eight spend a third or more of their length on images with
almost nothing in them. Both are the loop's own behaviour, described below.

## Stored outputs reproduce from stored inputs

| What was run again | Result |
| --- | --- |
| 30 images in 9 batches, all three generators, from the stored caption and seed | mean pixel difference from the stored image 0.7--4.8 on the 0--255 scale (median 2.3) |
| the same captions at a different seed (30 images) | 5.5--106 (median 35) |
| 160 stored images in four batches of 40, both captioners | 160 of 160 captions identical to the stored ones |
| 3,000 stored captions, one whole run in each of three cells | every embedding matches the stored vector (cosine 1.000000) |

As on the panel, the few levels of difference at the same seed are the AVIF
encoding the stored image has been through and the regenerated one has not.
The two smallest differences at another seed (5.5 and 9.9) are the two flat
images below, which stay nearly flat whatever the seed.

The targets sit on the steps either side of each interruption, chosen from the
timestamps: every interruption fell between whole steps, so each target is the
last step stored before it or the first redone after it. All but the step-1388
ZImageTurbo + Gemma4 batch were first made on NVIDIA driver 580.178 and
regenerated on 580.196, which is the direct check that the driver upgrade of
23 October changed nothing that matters.

## Images

All 240,000 decode to 1024 by 1024 RGB AVIF, and no two are byte-identical. No
image is pure black.

3,667 images in ten runs are flat (luma standard deviation under 2 levels).
3,652 of them are in SD35Medium + Moondream3, in nine runs of the apple and
city prompts, which slide into saturated yellow, green or orange fields. The
tenth run is Flux2Klein + Gemma4 on the doorway prompt, which reaches a dark
maroon field at step 1,970. In SD35Medium + Moondream3 the slide is a run of
apples or shapes on an ever plainer coloured ground, which the captions track
until they describe the ground alone.

All but nine flat images follow a caption the audit reads as asking for a
plain field. The nine ask for apples, lines, dots or stripes on a yellow, green
or orange ground, all in SD35Medium + Moondream3. They are saturated fields (red channel at 255, blue near 0), and
the first of them shows two faint ghost apples. SD35Medium washes the subject
out of a caption whose ground is a saturated colour, rather than failing to
generate. Two of the nine (steps 234 and 406, in batches with three ordinary
images each) regenerate flat from their stored caption and seed, and stay
nearly flat at another seed, so it is the caption that does it, not the draw.

Low-detail images (under 5 kB) number 5,796, and eight runs have a third or
more of their images below that, all of them SD35Medium + Moondream3. One ends
there by the audit's rule (nine-tenths of its last fifty images low-detail),
six more have two-thirds or more of their last fifty low-detail, and one has
left by its end.

## Captions

None is empty and none is a refusal. Gemma4's median is 209 words (max 621),
Moondream3's 51 (max 736). There is no mis-decoded text: 26,027 captions carry
correctly encoded em dashes or curly quotes, and none carries the `â€` that
UTF-8 read as Latin-1 would leave.

- Moondream3's 83 repetition loops are the same failure the panel showed:
  transcribed text in the image repeated until the 1,024-token ceiling ("LOVE
  IS OF THE BEATLE ...", a row of one Chinese character). Eight Moondream3
  captions contain a replacement character, and 116 Moondream3 and four Gemma4
  captions lack terminal punctuation. One Gemma4 caption in 120,000 reaches the
  1,024-token generation ceiling (1,023 tokens under its own tokenizer)
- 129 Gemma4 captions and 30 Moondream3 captions carry ten or more non-Latin
  characters, for instance Gemma4 reading the names off character cards in the
  game screenshots some Flux2Klein + Gemma4 runs drift into
- 809 distinct captions occur in more than one run, 8,228 times in all, nearly
  all in the Moondream3 cells. The commonest, "A solid yellow background fills
  the frame. ...", appears 2,040 times across nine SD35Medium + Moondream3 runs
- the first caption of a run names a content word from its prompt in 233 of 240
  runs; the seven that do not are four of the city prompt and three of the
  doorway prompt

### Cut at the generator's encoder

| Network | Captions read | Median tokens | Over 512 | Median tokens cut |
| --- | --- | --- | --- | --- |
| SD35Medium + Gemma4 | 39,960 | 357 | 863 (2.2%) | 29 |
| Flux2Klein + Gemma4 | 39,960 | 300 | 114 (0.3%) | 67 |
| SD35Medium + Moondream3 | 39,960 | 66 | 96 (0.2%) | 698 |
| Flux2Klein + Moondream3 | 39,960 | 77 | 12 (0.03%) | 531 |
| ZImageTurbo + Gemma4 | 39,960 | 267 | 4 (0.01%) | 8 |
| ZImageTurbo + Moondream3 | 39,960 | 66 | 1 | 521 |

As on the panel, what Gemma4 loses is the end of a caption that runs a little
long, and the Moondream3 rows are its repetition loops. With Flux2Klein the cut
doubles in the second half of a run (0.57% against 0.29% overall).

## Process

| Check | Result |
| --- | --- |
| steps | 12,000, each a single batch of 40 |
| invocations starting before their input completed | 0 |
| stalled or retried steps | 0 |
| seeds | 240,000, all recorded, 32-bit; three values used twice, each pair in different runs (6.7 expected) |
| embeddings | 240,000, 256 dimensions, unit length, none empty; identical captions embed to cosine 0.9994 or better |
| code under `lib/`, `priv/` or `config/` changed after launch | `d2143d5` on 4 October, taken up at step 30 (TASK-104) |
| model environment rebuilt during the run | no (site-packages last modified 5 September) |

The run resumed six times and none split a step:

| When (AEDT) | Why | Cell | Last step stored | Redone from |
| --- | --- | --- | --- | --- |
| 4 Oct 20:30 | killed to take up `d2143d5` | Flux2Klein + Gemma4 | 29 | 30 |
| 13 Oct 21:12 | out-of-memory kill | ZImageTurbo + Moondream3 | 1,437 | 1,438 |
| 15 Oct 13:57 | out-of-memory kill | Flux2Klein + Moondream3 | 398 | 399 |
| 20 Oct 13:35 | stopped by hand, restarted at 15:25 | SD35Medium + Gemma4 | 1,555 | 1,556 |
| 21 Oct 22:34 | out-of-memory kill | ZImageTurbo + Gemma4 | 334 | 335 |
| 23 Oct 16:49 | stopped for a reboot | ZImageTurbo + Gemma4 | 1,387 | 1,388 |

The out-of-memory kills were of host memory, not the GPU. Each time the kernel
killed the model process (Python, holding 46--60 GB resident), while the
allocation that tipped the machine over came from another job: a worker thread,
a Claude session, ffmpeg. The model process is the largest thing on the machine
and so the one chosen. The log has five allocator warnings, all on 13 October
before that day's kill.

The NVIDIA driver was upgraded from 580.178 to 580.196 on 23 October at 14:22
and loaded at the reboot that afternoon, so ZImageTurbo + Gemma4 steps 1,388 to
1,999 ran on the newer driver. The regenerations above, nearly all of steps
first made on the older driver, reproduce on the newer one as closely as the
panel's did on a single driver. The kernel moved from 6.8.0-146 to 147 at the
same reboot.

## What this does not check

As for the panel: not that the models are the right ones (pinned, TASK-84), not
that the embedding model measures what the analysis wants, and nothing about an
image's content beyond whether it is flat or low-detail.
