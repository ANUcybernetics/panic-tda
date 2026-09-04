---
id: TASK-94
title: GLM-Image renders parts of the caption as literal text in the image
status: To Do
assignee: []
created_date: '2026-09-04 07:15'
labels:
  - instrument
  - models
  - paper
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found 2026-09-04 while checking what max_sequence_length does per pipeline for TASK-89.

GlmImagePipeline splits the prompt in two. The caption goes to a vision-language encoder through processor.apply_chat_template with no length ceiling. Separately, get_glyph_texts extracts substrings that GLM should render as literal text inside the image, encodes them with ByT5, and passes them to the transformer as prompt_embeds. The extraction is four regexes over the prompt, and one of them is

    re.findall(r"'([^']*)'", p)

which matches on APOSTROPHES, not quotes. Any caption containing two or more possessives hands GLM the entire span between them as text to render. On the eight-caption TASK-89 set, four captions produce a glyph fragment and two of those are long runs of ordinary prose:

  c02   92w  18 ByT5 tokens   "Krakow Glowny,"                  (a real sign in the image)
  c05  222w  19 ByT5 tokens   "E Krakow Glowny"                 (ditto)
  c06  245w  103 ByT5 tokens  "s eye toward him. The overall mood is positive, ..."
  c07  313w  268 ByT5 tokens  "s left. He is wearing a gray t-shirt with green trim ..."

The two long ones are pure artefact: the text between "the man's" and "the boy's".

WHY IT MATTERS. Possessive frequency is a property of captioner style, and the captioner is a panel factor. GLM-Image networks therefore get a caption-dependent extra instruction that the other four generators never see, and it is largest for the verbose prose captioners (JoyCaption, Gemma4, Qwen3VL) and absent for terse ones. Any GLMImage-versus-other or captioner-within-GLMImage comparison is confounded by it, which lands on RQ2 directly.

NOT A TRUNCATION BUG. max_sequence_length caps only this glyph branch and truncates without padding, so it is not the padding perturbation it is for the other four; the longest fragment here is 268 tokens against the 2048 upstream default. panic_models keeps GLMImage at 2048 for that reason.

OPTIONS. Leave it and report it as a property of the model; strip or neutralise apostrophes before the GLMImage call, which changes the text the VLM encoder sees too; or pass prompt_embeds explicitly with an empty glyph branch, which is closest to "the caption conditions the image and nothing else" and matches what the other four do. Whichever way, measure how much it changes the image before TASK-90.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The size of the effect measured: images from the same caption with and without the spurious glyph fragment, compared in caption-embedding space against the seed-noise floor from TASK-89
- [ ] #2 Glyph-fragment incidence measured per captioner over real trajectory captions, so the confound with captioner style is quantified rather than assumed
- [ ] #3 Decision recorded on whether GLMImage runs with the glyph branch as-is, neutralised, or empty --- and applied in panic_models before TASK-90
- [ ] #4 If the glyph branch stays, the paper's methods state that GLMImage receives an extra text-rendering instruction the other four generators do not
<!-- AC:END -->
