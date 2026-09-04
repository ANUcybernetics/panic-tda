# Screening text-to-image models for the panel

Method and results from the 2026-09-04 search for a fifth generator, after
GLMImage was removed (TASK-94). Kept because the same screen applies whenever
the lineup changes, and because most of the rejections are non-obvious from a
model card.

## The method

Enumerate every model with a first-class pipeline in the installed diffusers
(`api.list_models(filter="diffusers", pipeline_tag="text-to-image")` over three
sort orders, then group by the `_class_name` in each repo's `model_index.json`),
rather than searching for notable releases. A model without a diffusers pipeline
cannot be measured without a spike, so it is not a candidate; and the pipeline's
own `__call__` signature answers two of the four screens for free, before any
download.

## The four screens

1. **bfloat16 in 48 GB, no quantisation.** GLMImage was the only quantised
   generator in the panel, which confounded architecture with quantisation for
   every result about it. A candidate that only ships fp8 or NF4 weights, or
   needs sequential offload to fit, repeats that.
2. **Caption context at least 512 tokens.** The v2 captioners run to 466 T5
   tokens (TASK-87, decision-01). A shorter ceiling truncates the verbose
   captioners and reintroduces the per-model asymmetry TASK-82 removed.
3. **No prompt-modifying default.** Read the `__call__` signature for anything
   matching instruction, template, system_prompt, rewrite or enhance. This is
   the screen that matters most and is the least visible: **five of the fifteen
   candidates rewrite or augment the prompt by default.** GLM-Image's glyph
   branch was not a one-off.
4. **Architecturally distinct from what the panel already has.** The panel's
   four generators are three architectures: Flux2Klein and Flux2Dev share the
   Flux2 transformer and its Mistral3 encoder and differ only in distillation.
   Text encoder family is the axis worth varying, since it is how the generator
   reads the caption --- which is the paper's subject.

## Results

| candidate                          | bf16 fits    | context | prompt path                                         | verdict                                                          |
| ---------------------------------- | ------------ | ------- | --------------------------------------------------- | ---------------------------------------------------------------- |
| THUDM/CogView4-6B                  | 29.0 GB      | 1024    | clean                                               | **adopted, TASK-95**                                             |
| lodestones/Chroma1-HD              | ~26 GB       | 512     | clean                                               | fallback; FLUX.1-derived, overlaps Flux2                         |
| Efficient-Large-Model/SANA1.5_4.8B | 14.8 GB      | 300     | `complex_human_instruction`                         | rejected: context                                                |
| briaai/FIBO                        | 23.8 GB      | 3000    | clean                                               | gated repo, non-permissive licence, JSON-structured prompts      |
| baidu/ERNIE-Image                  | 29.4 GB      | —       | ships a `pe` prompt-encoder (Ministral3ForCausalLM) | rejected; also Mistral3 + AutoencoderKLFlux2, i.e. Flux2-derived |
| ideogram-ai/ideogram-4             | fp8/NF4 only | 2048    | clean                                               | rejected: no unquantised release                                 |
| meituan-longcat/LongCat-Image      | 27.3 GB      | —       | `enable_prompt_rewrite=True`                        | rejected                                                         |
| HiDream-ai/HiDream-I1              | 43.9 GB      | 128     | clean                                               | rejected: context                                                |
| Alpha-VLLM/Lumina-Image-2.0        | 19.8 GB      | 256     | `system_prompt`                                     | rejected                                                         |
| Qwen/Qwen-Image                    | 53.7 GB      | 512     | clean                                               | rejected: size                                                   |
| krea/Krea-2-Raw                    | 57.7 GB      | 512     | clean                                               | rejected: size                                                   |
| BestWishYsh/Helios-Base            | 128.3 GB     | 512     | clean                                               | rejected: size                                                   |
| ATH-MaaS/Ovis-Image-7B             | 52.9 GB      | 256     | —                                                   | rejected: size and context                                       |
| stepfun-ai/NextStep-1.1            | 55.7 GB      | —       | —                                                   | rejected: size                                                   |
| nvidia/Cosmos3-Super-Text2Image    | 122.4 GB     | —       | system prompt and templates                         | rejected                                                         |
| Photoroom/prx-1024-t2i-beta        | 19.3 GB      | —       | clean                                               | beta, 55 downloads --- too little to rely on                     |

Context defaults are the pipeline's `max_sequence_length`; sizes are the sum of
`.safetensors` in the repo (Chroma's 42.2 GB total counts a duplicate
single-file checkpoint alongside the sharded one).

## Why CogView4-6B

It is the only candidate that clears every screen with headroom. 1024-token
context against a 466-token worst case, an apache-2.0 licence, 29 GB at
bfloat16, and no instruction, template or rewrite parameter anywhere in its call
signature. Its GLM-4 decoder-only text encoder (`GlmModel`, 18.3 GB of the 29)
gives the panel four encoder families: T5+CLIP, Mistral3, Qwen3, GLM-4.

Its weaknesses are age --- March 2025, the oldest model in the panel --- and
that its distinctness is the encoder rather than the sampler, where SANA's
linear attention and 32x deep-compression autoencoder would have been more novel
had its context been usable. Speed is the open risk: 50 default steps on a 6B
transformer is unmeasured, and at 20-25 s/image it would add roughly 20 GPU-days
rather than the 6 a fast generator costs. TASK-83's step-count method applies.
