"""PANIC-TDA model registry: setup, loading, invoke, and embedding functions.

All functions are called from Elixir via Snex.pyeval(). The module is imported
once during PythonBridge.ensure_setup/1 and then used as panic_models.*.
"""

from __future__ import annotations

import base64
import functools
import gc
import io
import os
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

IMAGE_SIZE = 256
EMBEDDING_DIM = 256

_T2I_IMAGE_SIZES: dict[str, int] = {
    "SD35Medium": 1024,
    "Flux2Dev": 1024,
    "GLMImage": 1024,
    "ZImageTurbo": 1024,
    "Flux2Klein": 1024,
}

# Upstream revisions, pinned 2026-09-02 (TASK-84). Each was the repo's HEAD at
# that date. Pinning matters because an unpinned repo resolves to whatever was
# cached at first download, which differs between machines and drifts silently.
_REVISIONS: dict[str, str] = {
    "stabilityai/stable-diffusion-3.5-medium": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
    "Tongyi-MAI/Z-Image-Turbo": "f332072aa78be7aecdf3ee76d5c247082da564a6",
    "black-forest-labs/FLUX.2-klein-9B": "92196c8e11f7b6cf2b7493e037d8c5345c559216",
    "black-forest-labs/FLUX.2-dev": "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
    "zai-org/GLM-Image": "2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4",
    "vikhyatk/moondream2": "6b714b26eea5cbd9f31e4edb2541c170afa935ba",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "google/gemma-3n-E2B-it": "5e092ebca197cdcd8d8b195040accf22693501bc",
    "mistral-community/pixtral-12b": "c2756cbbb9422eba9f6c5c439a214b0392dfc998",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "9eb2daaa8597bf192a8b0e73f848f3a102794df5",
    "Qwen/Qwen3-Embedding-4B": "5cf2132abc99cad020ac570b19d031efec650f2b",
    # TASK-87 lineup
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "internlm/CapRL-Qwen3VL-4B": "1db1c1dd241e2df95b59846a94cdee5300de9ef9",
    "fancyfeast/llama-joycaption-beta-one-hf-llava": "ebf414ea497a020da0f82df3913e5b6cb8e9663a",
    "google/gemma-4-26B-A4B-it": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    "moondream/moondream3-preview": "5112966d1a723413b1c9a1e8bea272b72e647b35",
}


def _rev(repo: str) -> str:
    """The pinned revision for `repo`, for a from_pretrained `revision=` kwarg."""
    return _REVISIONS[repo]


_models: dict[str, Any] = {}
_models_offload_only: set[str] = set()


def _bnb_4bit_config() -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup() -> None:
    """One-time environment init: suppress warnings, patch libraries."""
    warnings.filterwarnings("ignore")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        import transformers

        transformers.logging.set_verbosity_error()

        # huggingface_hub strict dataclass validation rejects float↔int
        # mismatches in model configs. Some upstream config.json files store
        # integer-valued fields as floats (e.g. nomic-bert "n_inner": 2048.0).
        # Patch _validate_simple_type to accept float-for-int and int-for-float
        # when the value is losslessly convertible.
        import huggingface_hub.dataclasses as _hhd

        _orig_validate_simple = _hhd._validate_simple_type

        def _lenient_validate_simple(
            name: str, value: Any, expected_type: type
        ) -> None:
            try:
                _orig_validate_simple(name, value, expected_type)
            except TypeError:
                if (
                    expected_type is int
                    and isinstance(value, float)
                    and not isinstance(value, bool)
                    and value == int(value)
                ):
                    return
                if (
                    expected_type is float
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                ):
                    return
                raise

        _hhd._validate_simple_type = _lenient_validate_simple

        import transformers.modeling_utils as _tmu

        # transformers 5.x passes code_revision through from_config → _from_config
        # → cls(config, **kwargs), but remote-code model classes (Moondream's,
        # for one) don't accept it. Strip it — code_revision is only meaningful
        # for from_pretrained, not from_config.
        _orig_from_config = _tmu.PreTrainedModel._from_config.__func__

        @classmethod  # type: ignore[misc]
        def _safe_from_config(cls, config, **kwargs):  # type: ignore[no-untyped-def]
            kwargs.pop("code_revision", None)
            return _orig_from_config(cls, config, **kwargs)

        _tmu.PreTrainedModel._from_config = _safe_from_config

        _orig_ptm_init = _tmu.PreTrainedModel.__init__

        def _patched_ptm_init(self: Any, *args: Any, **kwargs: Any) -> None:
            _orig_ptm_init(self, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}

        _tmu.PreTrainedModel.__init__ = _patched_ptm_init

        if hasattr(_tmu.PreTrainedModel, "get_expanded_tied_weights_keys"):
            _orig_getwk = _tmu.PreTrainedModel.get_expanded_tied_weights_keys

            def _safe_getwk(self: Any, all_submodels: bool = False) -> dict[str, str]:
                twk = getattr(self, "_tied_weights_keys", None)
                if isinstance(twk, list):
                    self._tied_weights_keys = None
                    try:
                        return _orig_getwk(self, all_submodels)
                    finally:
                        self._tied_weights_keys = twk
                return _orig_getwk(self, all_submodels)

            _tmu.PreTrainedModel.get_expanded_tied_weights_keys = _safe_getwk
    except Exception:
        pass

    try:
        import transformers.cache_utils as _cu

        if not hasattr(_cu.DynamicCache, "get_usable_length"):

            def _get_usable_length(
                self: Any, new_seq_length: int = 0, layer_idx: int = 0
            ) -> int:
                return self.get_seq_length(layer_idx)

            _cu.DynamicCache.get_usable_length = _get_usable_length

    except Exception:
        pass

    try:
        import diffusers

        diffusers.logging.set_verbosity_error()

        from diffusers.pipelines.glm_image.pipeline_glm_image import GlmImagePipeline

        _orig_generate_prior = GlmImagePipeline.generate_prior_tokens

        _GLM_PRIOR_MAX_RETRIES = 5

        def _retrying_generate_prior_tokens(self, *args, **kwargs):
            for attempt in range(_GLM_PRIOR_MAX_RETRIES):
                try:
                    return _orig_generate_prior(self, *args, **kwargs)
                except RuntimeError as e:
                    if "invalid for input of size" not in str(e):
                        raise
                    if attempt == _GLM_PRIOR_MAX_RETRIES - 1:
                        raise
                    print(
                        f"[panic_models] GLM prior token count mismatch "
                        f"(attempt {attempt + 1}), regenerating"
                    )

        GlmImagePipeline.generate_prior_tokens = _retrying_generate_prior_tokens
    except Exception:
        pass

    try:
        # transformers 5.5.x has a bug in _convert_peft_config_moe: it runs
        # for any model that has a checkpoint-conversion mapping, then does a
        # bracket lookup in _MOE_TARGET_MODULE_MAPPING which only contains
        # real MoE architectures. A qwen2_5_vl checkpoint (Qwen25VL) maps to
        # qwen2_vl and crashes with KeyError: 'qwen2_vl'. Patch it to no-op
        # when the mapped base type isn't a known MoE.
        import transformers.integrations.peft as _tip

        if hasattr(_tip, "_convert_peft_config_moe") and not getattr(
            _tip._convert_peft_config_moe, "_panic_patched", False
        ):
            _orig_moe = _tip._convert_peft_config_moe

            def _safe_convert_peft_config_moe(peft_config, model_type):
                try:
                    return _orig_moe(peft_config, model_type)
                except KeyError:
                    return peft_config

            _safe_convert_peft_config_moe._panic_patched = True
            _tip._convert_peft_config_moe = _safe_convert_peft_config_moe
    except Exception:
        pass

    try:
        from functools import partialmethod

        from tqdm import tqdm

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# NoSortingSentenceTransformer (preserves input order)
# ---------------------------------------------------------------------------


class NoSortingSentenceTransformer(SentenceTransformer):
    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | None = None,
        normalize_embeddings: bool = False,
        precision: str = "float32",
        **kwargs: Any,
    ) -> Any:
        self.eval()

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self.device

        all_embeddings: list[Any] = []

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                features = self.preprocess(batch)
                features = batch_to_device(features, device)
                out_features = self.forward(features)
                embeddings = out_features["sentence_embedding"]

                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                if convert_to_numpy:
                    all_embeddings.extend(embeddings.cpu().float().numpy())
                else:
                    all_embeddings.extend(embeddings)

        if convert_to_tensor and not convert_to_numpy and len(all_embeddings) > 0:
            all_embeddings = torch.stack(all_embeddings)

        if input_was_string:
            return all_embeddings[0]

        return all_embeddings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_sentence_transformer(name: str, model_path: str, **kwargs: Any) -> None:
    try:
        m = NoSortingSentenceTransformer(model_path, **kwargs)
    except FileNotFoundError:
        _ = SentenceTransformer(model_path, **kwargs)
        m = NoSortingSentenceTransformer(model_path, **kwargs)
    if torch.cuda.is_available():
        m = m.to("cuda")
    m.eval()
    _models[name] = m


def _encode_image_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", lossless=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_image_b64(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _encode_embedding(arr: np.ndarray) -> str:
    f32 = arr.astype(np.float32)
    np.nan_to_num(f32, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    f32 = f32[:EMBEDDING_DIM]
    norm = float(np.linalg.norm(f32))
    if norm > 0.0:
        f32 = f32 / norm
    return base64.b64encode(f32.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Model state queries
# ---------------------------------------------------------------------------


def is_model_loaded(name: str) -> bool:
    return name in _models


# ---------------------------------------------------------------------------
# Model loading: T2I pipeline registry
# ---------------------------------------------------------------------------

_T2I_LOADER_CONFIGS: dict[str, dict[str, Any]] = {
    "SD35Medium": {
        "pipeline_cls": "StableDiffusion3Pipeline",
        "repo": "stabilityai/stable-diffusion-3.5-medium",
        "offload": "model_cpu_offload",
        # T5 is kept (decision-01): without it SD3.5 sees only CLIP's 77 tokens.
        "extra_kwargs": {
            "torch_dtype": "bfloat16",
            "use_fast": True,
        },
    },
    "Flux2Dev": {
        "pipeline_cls": "Flux2Pipeline",
        "repo": "black-forest-labs/FLUX.2-dev",
        "offload": "sequential_cpu_offload",
        "offload_only": True,
        "extra_kwargs": {"torch_dtype": "bfloat16", "token": True},
    },
    "GLMImage": {
        "pipeline_cls": "GlmImagePipeline",
        "repo": "zai-org/GLM-Image",
        "offload": "model_cpu_offload",
        "offload_only": True,
        "quantize": True,
        "extra_kwargs": {"torch_dtype": "bfloat16"},
    },
    "ZImageTurbo": {
        "pipeline_cls": "ZImagePipeline",
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "offload": "model_cpu_offload",
        "extra_kwargs": {"torch_dtype": "bfloat16"},
    },
    "Flux2Klein": {
        "pipeline_cls": "Flux2KleinPipeline",
        "repo": "black-forest-labs/FLUX.2-klein-9B",
        "offload": "model_cpu_offload",
        "extra_kwargs": {"torch_dtype": "bfloat16"},
    },
}


def _load_t2i_pipeline(name: str) -> None:
    import diffusers

    cfg = _T2I_LOADER_CONFIGS[name]
    pipeline_cls = getattr(diffusers, cfg["pipeline_cls"])
    kwargs = {}
    for k, v in cfg["extra_kwargs"].items():
        if v == "bfloat16":
            kwargs[k] = torch.bfloat16
        else:
            kwargs[k] = v
    if cfg.get("quantize"):
        from diffusers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": _bnb_4bit_config(),
            }
        )
    pipe = pipeline_cls.from_pretrained(
        cfg["repo"], revision=_rev(cfg["repo"]), **kwargs
    )
    getattr(pipe, f"enable_{cfg['offload']}")()
    _models[name] = pipe
    if cfg.get("offload_only"):
        _models_offload_only.add(name)


# ---------------------------------------------------------------------------
# Model loading: I2T model registry
# ---------------------------------------------------------------------------


_MOONDREAM_REPO = "vikhyatk/moondream2"
_MOONDREAM_REV = _REVISIONS[_MOONDREAM_REPO]


def _load_moondream() -> None:
    import sys

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig

    snap_dir = snapshot_download(_MOONDREAM_REPO, revision=_MOONDREAM_REV)

    AutoConfig.from_pretrained(
        _MOONDREAM_REPO, revision=_MOONDREAM_REV, trust_remote_code=True
    )

    md_mod = next(
        mod
        for name, mod in sys.modules.items()
        if "transformers_modules" in name
        and "moondream" in name
        and "MoondreamModel" in getattr(mod, "__dict__", {})
        and "MoondreamConfig" in getattr(mod, "__dict__", {})
    )

    import json

    with open(f"{snap_dir}/config.json") as f:
        raw_config = json.load(f)

    config = md_mod.MoondreamConfig.from_dict(raw_config["config"])
    model = md_mod.MoondreamModel(config, setup_caches=True)

    state_dict = load_file(f"{snap_dir}/model.safetensors")
    stripped = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(stripped, strict=False)

    _models["Moondream"] = model.to("cuda").eval()


def _load_pixtral() -> None:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(
        "mistral-community/pixtral-12b",
        revision=_rev("mistral-community/pixtral-12b"),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=_bnb_4bit_config(),
    )
    processor = AutoProcessor.from_pretrained(
        "mistral-community/pixtral-12b", revision=_rev("mistral-community/pixtral-12b")
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    _models["Pixtral"] = {"processor": processor, "model": model}
    _models_offload_only.add("Pixtral")


def _load_llama32vision() -> None:
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    model = MllamaForConditionalGeneration.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        revision=_rev("meta-llama/Llama-3.2-11B-Vision-Instruct"),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=True,
        quantization_config=_bnb_4bit_config(),
    )
    processor = AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        revision=_rev("meta-llama/Llama-3.2-11B-Vision-Instruct"),
        token=True,
    )
    _models["LLaMA32Vision"] = {"processor": processor, "model": model}
    _models_offload_only.add("LLaMA32Vision")


# Qwen-VL family: same chat-template and processor path, different checkpoints.
# CapRL is a Qwen3-VL fine-tune, so it shares everything but the weights --- which
# is the point of pairing it with Qwen3VL in the panel (TASK-87): same backbone,
# different training objective.
_QWEN_VL_CONFIGS: dict[str, dict[str, Any]] = {
    "Qwen25VL": {
        "repo": "Qwen/Qwen2.5-VL-7B-Instruct",
        "cls": "Qwen2_5_VLForConditionalGeneration",
        "quantize": True,
    },
    "Qwen3VL": {
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "cls": "Qwen3VLForConditionalGeneration",
        "quantize": True,
    },
    "CapRL": {
        "repo": "internlm/CapRL-Qwen3VL-4B",
        "cls": "Qwen3VLForConditionalGeneration",
        "quantize": False,
    },
}


def _load_qwen_vl(name: str) -> None:
    import transformers

    cfg = _QWEN_VL_CONFIGS[name]
    repo = cfg["repo"]
    model_cls = getattr(transformers, cfg["cls"])
    kwargs: dict[str, Any] = {
        "revision": _rev(repo),
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "device_map": "auto",
    }
    if cfg["quantize"]:
        kwargs["quantization_config"] = _bnb_4bit_config()

    model = model_cls.from_pretrained(repo, **kwargs)
    processor = transformers.AutoProcessor.from_pretrained(repo, revision=_rev(repo))
    _models[name] = {"processor": processor, "model": model}
    _models_offload_only.add(name)


def _load_joycaption() -> None:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    repo = "fancyfeast/llama-joycaption-beta-one-hf-llava"
    model = LlavaForConditionalGeneration.from_pretrained(
        repo,
        revision=_rev(repo),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=_bnb_4bit_config(),
    )
    processor = AutoProcessor.from_pretrained(repo, revision=_rev(repo))
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    _models["JoyCaption"] = {"processor": processor, "model": model}
    _models_offload_only.add("JoyCaption")


def _load_gemma3n() -> None:
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    model = (
        Gemma3nForConditionalGeneration.from_pretrained(
            "google/gemma-3n-E2B-it",
            revision=_rev("google/gemma-3n-E2B-it"),
            torch_dtype=torch.bfloat16,
        )
        .to("cuda")
        .eval()
    )
    processor = AutoProcessor.from_pretrained(
        "google/gemma-3n-E2B-it", revision=_rev("google/gemma-3n-E2B-it")
    )
    _models["Gemma3n"] = {"processor": processor, "model": model}


# ---------------------------------------------------------------------------
# Model loading: embedding models
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unified model loader dispatch
# ---------------------------------------------------------------------------

_I2T_LOADERS: dict[str, Any] = {
    "Moondream": _load_moondream,
    "Pixtral": _load_pixtral,
    "LLaMA32Vision": _load_llama32vision,
    "Qwen25VL": functools.partial(_load_qwen_vl, "Qwen25VL"),
    "Qwen3VL": functools.partial(_load_qwen_vl, "Qwen3VL"),
    "CapRL": functools.partial(_load_qwen_vl, "CapRL"),
    "JoyCaption": _load_joycaption,
    "Gemma3n": _load_gemma3n,
}

_EMBEDDING_LOADERS: dict[str, tuple[str, dict[str, Any]]] = {
    "Qwen3Embed": (
        "Qwen/Qwen3-Embedding-4B",
        {
            "revision": _REVISIONS["Qwen/Qwen3-Embedding-4B"],
            "model_kwargs": {"attn_implementation": "sdpa"},
            "processor_kwargs": {"padding_side": "left"},
        },
    ),
}

def load_model(name: str) -> None:
    """Load a model by name into the _models registry."""
    if name in _T2I_LOADER_CONFIGS:
        _load_t2i_pipeline(name)
    elif name in _I2T_LOADERS:
        _I2T_LOADERS[name]()
    elif name in _EMBEDDING_LOADERS:
        path, kwargs = _EMBEDDING_LOADERS[name]
        _load_sentence_transformer(name, path, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# GPU management
# ---------------------------------------------------------------------------


def swap_to_gpu(name: str) -> None:
    if name not in _models_offload_only:
        obj = _models[name]
        if hasattr(obj, "remove_all_hooks"):
            obj.remove_all_hooks()
        if isinstance(obj, dict):
            obj["model"].to("cuda").eval()
        elif isinstance(obj, torch.nn.Module):
            obj.to("cuda").eval()
        else:
            obj.to("cuda")


def swap_to_cpu(name: str) -> None:
    if name not in _models:
        return
    if name in _models_offload_only:
        torch.cuda.empty_cache()
    else:
        obj = _models[name]
        if hasattr(obj, "remove_all_hooks"):
            obj.remove_all_hooks()
        if isinstance(obj, dict):
            obj["model"].to("cpu")
        else:
            obj.to("cpu")
        torch.cuda.empty_cache()


def _force_free_model(obj: Any) -> None:
    if hasattr(obj, "remove_all_hooks"):
        obj.remove_all_hooks()
    if isinstance(obj, dict):
        for v in obj.values():
            if hasattr(v, "parameters"):
                for p in v.parameters():
                    p.data = torch.empty(0)
                for b in v.buffers():
                    b.data = torch.empty(0)
            elif hasattr(v, "cpu"):
                try:
                    v.cpu()
                except ValueError:
                    pass
            del v
    elif hasattr(obj, "cpu"):
        try:
            obj.to("cpu")
        except ValueError:
            pass
    del obj


def unload_model(name: str) -> None:
    if name not in _models:
        return
    obj = _models.pop(name)
    _models_offload_only.discard(name)
    _force_free_model(obj)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def unload_all_models() -> None:
    for name in list(_models.keys()):
        _force_free_model(_models.pop(name))
    _models_offload_only.clear()
    gc.collect()
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# T2I invocation: config-driven
# ---------------------------------------------------------------------------

_T2I_INVOKE_CONFIGS: dict[str, dict[str, Any]] = {
    # max_sequence_length is the T5 branch's ceiling and defaults to 256 in
    # diffusers; 512 is the pipeline maximum and what decision-01 calls for,
    # since natural captions reach ~300 words. CLIP's 77 tokens are separate
    # and architectural.
    "SD35Medium": {
        "num_inference_steps": 20,
        "guidance_scale": 5.0,
        "max_sequence_length": 512,
    },
    # 12 rather than 15 (TASK-83): caption cosine against a 25-step reference is
    # flat and non-monotone across 8-15 steps over 12 prompts, so the metric
    # cannot separate them; 12 keeps the best of that band on both caption
    # cosine and pixel fidelity while cutting ~19% off the panel's dearest model.
    "Flux2Dev": {"num_inference_steps": 12, "guidance_scale": 3.5},
    "GLMImage": {"num_inference_steps": 25, "guidance_scale": 7.5},
    "ZImageTurbo": {"num_inference_steps": 8, "guidance_scale": 0.0},
    "Flux2Klein": {"num_inference_steps": 4, "guidance_scale": 1.0},
}

_T2I_BATCH_CAPABLE: set[str] = {
    "SD35Medium",
    "ZImageTurbo",
    "Flux2Klein",
    "Flux2Dev",
    "GLMImage",
}
_I2T_BATCH_CAPABLE: set[str] = {"Pixtral", "LLaMA32Vision", "JoyCaption"}


_T2I_MAX_RETRIES = 3
# Batch caps tuned on the RTX 6000 Ada (48 GB). Probed at 4/8/16 in TASK-78:
# Flux2Dev is flat from 4 to 16 (57.7 / 58.4 / 57.8 s/item), so it is already
# compute-saturated and stays at 4. GLMImage gains 7.3% at 8 (45.8 -> 42.4)
# and OOMs at 16, so 8 is its knee.
_T2I_MAX_BATCH: dict[str, int] = {
    "SD35Medium": 4,
    "ZImageTurbo": 4,
    "Flux2Klein": 2,
    "Flux2Dev": 4,
    "GLMImage": 8,
}


def _invoke_t2i_single(name: str, prompt: str) -> str:
    cfg = _T2I_INVOKE_CONFIGS[name]
    size = _T2I_IMAGE_SIZES.get(name, IMAGE_SIZE)
    for attempt in range(_T2I_MAX_RETRIES):
        try:
            img = _models[name](
                prompt=prompt,
                height=size,
                width=size,
                generator=None,
                **cfg,
            ).images[0]
            return _encode_image_b64(img)
        except RuntimeError as e:
            if attempt == _T2I_MAX_RETRIES - 1:
                raise
            print(f"[panic_models] {name} attempt {attempt + 1} failed: {e}, retrying")
    raise RuntimeError("unreachable")


def invoke_t2i(name: str, prompt: str) -> str:
    """Run a single T2I inference. Returns base64-encoded WEBP."""
    return _invoke_t2i_single(name, prompt)


def invoke_t2i_batch(name: str, prompts: list[str]) -> list[str]:
    """Run batch T2I inference. Returns list of base64-encoded WEBP."""
    cfg = _T2I_INVOKE_CONFIGS[name]
    size = _T2I_IMAGE_SIZES.get(name, IMAGE_SIZE)
    if name not in _T2I_BATCH_CAPABLE:
        return [_invoke_t2i_single(name, p) for p in prompts]
    max_batch = _T2I_MAX_BATCH[name]
    if len(prompts) <= max_batch:
        imgs = _models[name](
            prompt=prompts,
            height=size,
            width=size,
            generator=None,
            **cfg,
        ).images
        return [_encode_image_b64(img) for img in imgs]
    results: list[str] = []
    for i in range(0, len(prompts), max_batch):
        chunk = prompts[i : i + max_batch]
        imgs = _models[name](
            prompt=chunk,
            height=size,
            width=size,
            generator=None,
            **cfg,
        ).images
        results.extend(_encode_image_b64(img) for img in imgs)
    return results


# ---------------------------------------------------------------------------
# I2T invocation: strategy-based dispatch
# ---------------------------------------------------------------------------


def invoke_i2t(name: str, image_b64: str) -> str:
    """Run a single I2T inference. Returns caption text."""
    img = _decode_image_b64(image_b64)
    return _I2T_STRATEGIES[name](name, img)


_I2T_MAX_BATCH = 8

# Generation ceiling for every captioner. The default is deliberately far
# above any natural caption length so models terminate on their own (the
# panel's captioners stop at ~100-300 words); the balanced_panel_5x5 run used
# per-model ceilings of 100-128 tokens that cut most captions off mid-sentence.
# An experiment can override it via its i2t_max_new_tokens config key.
_I2T_MAX_NEW_TOKENS_DEFAULT = 1024
_I2T_MAX_NEW_TOKENS_OVERRIDE: int | None = None


def set_i2t_max_new_tokens(value: int | None) -> None:
    global _I2T_MAX_NEW_TOKENS_OVERRIDE
    _I2T_MAX_NEW_TOKENS_OVERRIDE = value


def _i2t_max_new_tokens() -> int:
    if _I2T_MAX_NEW_TOKENS_OVERRIDE is None:
        return _I2T_MAX_NEW_TOKENS_DEFAULT
    return _I2T_MAX_NEW_TOKENS_OVERRIDE


def invoke_i2t_batch(name: str, b64_list: list[str]) -> list[str]:
    """Run batch I2T inference. Returns list of caption texts."""
    if len(b64_list) <= _I2T_MAX_BATCH:
        images = [_decode_image_b64(b) for b in b64_list]
        return _I2T_BATCH_STRATEGIES[name](name, images)
    results: list[str] = []
    for i in range(0, len(b64_list), _I2T_MAX_BATCH):
        chunk = b64_list[i : i + _I2T_MAX_BATCH]
        images = [_decode_image_b64(b) for b in chunk]
        results.extend(_I2T_BATCH_STRATEGIES[name](name, images))
    return results


# --- Moondream ---


# caption() is left at its default length="normal" (decision-01); the SMC 2025
# runs used length="short", which is a brevity instruction, not a ceiling.
def _moondream_settings() -> dict[str, Any]:
    return {"temperature": 0.0, "max_tokens": _i2t_max_new_tokens()}


def _invoke_moondream(_name: str, img: Image.Image) -> str:
    model = _models["Moondream"]
    with torch.inference_mode():
        encoded = model.encode_image(img)
        cap = model.caption(encoded, settings=_moondream_settings())
    return cap["caption"].strip()


def _invoke_moondream_batch(_name: str, images: list[Image.Image]) -> list[str]:
    model = _models["Moondream"]
    results = []
    with torch.inference_mode():
        for img in images:
            encoded = model.encode_image(img)
            cap = model.caption(encoded, settings=_moondream_settings())
            results.append(cap["caption"].strip())
    return results


# --- Chat-template models (Pixtral, LLaMA32Vision) ---

_CHAT_TEMPLATE_CONFIGS: dict[str, dict[str, Any]] = {
    "Pixtral": {
        "message_fn": lambda img: [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        "processor_call": lambda proc, text, img: proc(
            text=[text], images=[img], padding=True, return_tensors="pt"
        ),
        "batch_processor_call": lambda proc, texts, images: proc(
            text=texts, images=images, padding=True, return_tensors="pt"
        ),
        "dtype_cast": torch.bfloat16,
        "extra_generate_kwargs": {},
        "batch_images_fn": lambda img: img,
    },
    # JoyCaption's chat template takes plain-string content, not the structured
    # parts list the others use; the image reaches it through the processor.
    # The instruction is deliberately identical to every other captioner's, so
    # that captioner and prompt stay unconfounded across the panel.
    "JoyCaption": {
        "message_fn": lambda _img: [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": "Describe this image."},
        ],
        "processor_call": lambda proc, text, img: proc(
            text=[text], images=[img], padding=True, return_tensors="pt"
        ),
        "batch_processor_call": lambda proc, texts, images: proc(
            text=texts, images=images, padding=True, return_tensors="pt"
        ),
        "dtype_cast": torch.bfloat16,
        "extra_generate_kwargs": {},
        "batch_images_fn": lambda img: img,
    },
    "LLaMA32Vision": {
        "message_fn": lambda img: [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        "processor_call": lambda proc, text, img: proc(img, text, return_tensors="pt"),
        "batch_processor_call": lambda proc, texts, images: proc(
            images, texts, padding=True, return_tensors="pt"
        ),
        "dtype_cast": None,
        "extra_generate_kwargs": {},
        "batch_images_fn": lambda img: [img],
    },
}


def _invoke_chat_template(name: str, img: Image.Image) -> str:
    cfg = _CHAT_TEMPLATE_CONFIGS[name]
    model_dict = _models[name]
    messages = cfg["message_fn"](img)
    text = model_dict["processor"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = cfg["processor_call"](model_dict["processor"], text, img)
    target = model_dict["model"].device
    if cfg["dtype_cast"] is not None:
        inputs = inputs.to(target, dtype=cfg["dtype_cast"])
    else:
        inputs = inputs.to(target)
    with torch.no_grad():
        gen_ids = model_dict["model"].generate(
            **inputs,
            max_new_tokens=_i2t_max_new_tokens(),
            **cfg["extra_generate_kwargs"],
        )
        gen_ids = gen_ids[:, inputs["input_ids"].shape[1] :]
    return (
        model_dict["processor"]
        .batch_decode(gen_ids, skip_special_tokens=True)[0]
        .strip()
    )


def _invoke_chat_template_batch(name: str, images: list[Image.Image]) -> list[str]:
    cfg = _CHAT_TEMPLATE_CONFIGS[name]
    model_dict = _models[name]
    all_texts = []
    all_images = []
    for img in images:
        messages = cfg["message_fn"](img)
        all_texts.append(
            model_dict["processor"].apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        all_images.append(cfg["batch_images_fn"](img))
    inputs = cfg["batch_processor_call"](model_dict["processor"], all_texts, all_images)
    target = model_dict["model"].device
    if cfg["dtype_cast"] is not None:
        inputs = inputs.to(target, dtype=cfg["dtype_cast"])
    else:
        inputs = inputs.to(target)
    with torch.no_grad():
        gen_ids = model_dict["model"].generate(
            **inputs,
            max_new_tokens=_i2t_max_new_tokens(),
            **cfg["extra_generate_kwargs"],
        )
        gen_ids = gen_ids[:, inputs["input_ids"].shape[1] :]
    return [
        s.strip()
        for s in model_dict["processor"].batch_decode(gen_ids, skip_special_tokens=True)
    ]


# --- Qwen-VL family (Qwen25VL, Qwen3VL, CapRL) ---


def _invoke_qwen_vl(name: str, img: Image.Image) -> str:
    from qwen_vl_utils import process_vision_info

    qwen_vl = _models[name]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text = qwen_vl["processor"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_vl["processor"](
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(qwen_vl["model"].device)
    with torch.no_grad():
        gen_ids = qwen_vl["model"].generate(
            **inputs, max_new_tokens=_i2t_max_new_tokens()
        )
        gen_ids = gen_ids[:, inputs["input_ids"].shape[1] :]
    return (
        qwen_vl["processor"].batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    )


def _invoke_qwen_vl_batch(name: str, images: list[Image.Image]) -> list[str]:
    from qwen_vl_utils import process_vision_info

    qwen_vl = _models[name]
    all_texts = []
    all_images: list[Any] = []
    for img in images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        all_texts.append(
            qwen_vl["processor"].apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        image_inputs, _ = process_vision_info(messages)
        all_images.extend(image_inputs)
    inputs = qwen_vl["processor"](
        text=all_texts,
        images=all_images,
        padding=True,
        return_tensors="pt",
    ).to(qwen_vl["model"].device)
    with torch.no_grad():
        gen_ids = qwen_vl["model"].generate(
            **inputs, max_new_tokens=_i2t_max_new_tokens()
        )
        gen_ids = gen_ids[:, inputs["input_ids"].shape[1] :]
    return [
        s.strip()
        for s in qwen_vl["processor"].batch_decode(gen_ids, skip_special_tokens=True)
    ]


# --- Gemma3n ---


def _invoke_gemma3n(_name: str, img: Image.Image) -> str:
    gemma3n = _models["Gemma3n"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    inputs = (
        gemma3n["processor"]
        .apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        .to(gemma3n["model"].device, dtype=torch.bfloat16)
    )
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen_ids = gemma3n["model"].generate(
            **inputs, max_new_tokens=_i2t_max_new_tokens(), do_sample=False
        )
    return (
        gemma3n["processor"]
        .decode(gen_ids[0][input_len:], skip_special_tokens=True)
        .strip()
    )


def _invoke_gemma3n_batch(_name: str, images: list[Image.Image]) -> list[str]:
    gemma3n = _models["Gemma3n"]
    all_messages = []
    for img in images:
        all_messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
        )
    inputs = (
        gemma3n["processor"]
        .apply_chat_template(
            all_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            padding=True,
        )
        .to(gemma3n["model"].device, dtype=torch.bfloat16)
    )
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen_ids = gemma3n["model"].generate(
            **inputs, max_new_tokens=_i2t_max_new_tokens(), do_sample=False
        )
    return [
        s.strip()
        for s in gemma3n["processor"].batch_decode(
            gen_ids[:, input_len:], skip_special_tokens=True
        )
    ]


# Strategy dispatch tables

_I2T_STRATEGIES: dict[str, Any] = {
    "Moondream": _invoke_moondream,
    "Pixtral": _invoke_chat_template,
    "LLaMA32Vision": _invoke_chat_template,
    "Qwen25VL": _invoke_qwen_vl,
    "Qwen3VL": _invoke_qwen_vl,
    "CapRL": _invoke_qwen_vl,
    "JoyCaption": _invoke_chat_template,
    "Gemma3n": _invoke_gemma3n,
}

_I2T_BATCH_STRATEGIES: dict[str, Any] = {
    "Moondream": _invoke_moondream_batch,
    "Pixtral": _invoke_chat_template_batch,
    "LLaMA32Vision": _invoke_chat_template_batch,
    "Qwen25VL": _invoke_qwen_vl_batch,
    "Qwen3VL": _invoke_qwen_vl_batch,
    "CapRL": _invoke_qwen_vl_batch,
    "JoyCaption": _invoke_chat_template_batch,
    "Gemma3n": _invoke_gemma3n_batch,
}


# ---------------------------------------------------------------------------
# Embeddings: text
# ---------------------------------------------------------------------------


def embed_text(name: str, texts: list[str]) -> list[str]:
    """Embed texts. Returns list of base64-encoded float32 vectors."""
    if name != "Qwen3Embed":
        raise ValueError(f"Unknown text embedding model: {name}")
    with torch.no_grad():
        embs = _models[name].encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
    return [_encode_embedding(e) for e in embs]


# ---------------------------------------------------------------------------
# Batch size probing
# ---------------------------------------------------------------------------


def probe_max_batch(
    name: str, test_inputs: list[str], sizes: list[int]
) -> dict[int, str]:
    """Try increasing batch sizes for a model, return {size: "ok"/"oom"/error}.

    For T2I models, test_inputs should be a list of prompt strings.
    For I2T models, test_inputs should be a list of base64-encoded images.
    """
    results: dict[int, str] = {}
    for n in sizes:
        batch = test_inputs[:n]
        torch.cuda.empty_cache()
        gc.collect()
        try:
            if name in _T2I_BATCH_CAPABLE:
                invoke_t2i_batch(name, batch)
            elif name in _I2T_BATCH_CAPABLE:
                invoke_i2t_batch(name, batch)
            else:
                results[n] = f"model {name} is not truly batched"
                break
            results[n] = "ok"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            results[n] = "oom"
            break
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            results[n] = str(e)
            break
    return results


# ---------------------------------------------------------------------------
# Benchmark harness (TASK-74): seeded T2I generation + per-item timing/parity.
# Benchmark-only — does NOT touch the production invoke_t2i* path, which uses
# generator=None (random). Seeds exist solely so batched-vs-serial output can be
# compared at matched noise to prove batching introduces no systematic change.
# ---------------------------------------------------------------------------


def _t2i_generate_seeded(name: str, prompts: list[str], seeds: list[int]) -> list:
    """Generate images for `prompts` with matched per-item `seeds`.

    Returns a list of PIL images. Passing a list of per-item generators makes the
    i-th image depend only on seeds[i], so a batch of N reproduces N single calls
    at the same seeds up to GPU kernel nondeterminism.
    """
    cfg = _T2I_INVOKE_CONFIGS[name]
    size = _T2I_IMAGE_SIZES.get(name, IMAGE_SIZE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gens = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    return _models[name](
        prompt=list(prompts),
        height=size,
        width=size,
        generator=gens if len(gens) > 1 else gens[0],
        **cfg,
    ).images


def benchmark_t2i(
    name: str,
    prompts: list[str],
    seeds: list[int],
    batch_sizes: list[int],
    dump_dir: str = "",
) -> dict[str, Any]:
    """Measure per-item wall-clock and batched-vs-serial pixel parity for a T2I model.

    Returns {n, single_per_item_s, batches: {bs: {per_item_s, parity_mean_abs_delta,
    parity_max_abs_delta, status}}}. per-item time is wall-clock / n; parity is the
    mean abs pixel delta (0-255) between each batched image and its seed-matched
    serial reference.
    """
    import time

    n = len(prompts)
    seeds = [int(s) for s in seeds]

    # Serial references (batch size 1), timed.
    torch.cuda.empty_cache()
    gc.collect()
    refs = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p, s in zip(prompts, seeds):
        refs.extend(_t2i_generate_seeded(name, [p], [s]))
    torch.cuda.synchronize()
    single_per_item = (time.perf_counter() - t0) / n

    ref_arrs = [np.asarray(im, dtype=np.float32) for im in refs]
    result: dict[str, Any] = {
        "n": n,
        "single_per_item_s": single_per_item,
        "batches": {},
    }

    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        for i, im in enumerate(refs):
            im.save(os.path.join(dump_dir, f"{name}_{i}_serial.png"))

    for bs in batch_sizes:
        if bs <= 1:
            continue
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            batched = []
            for i in range(0, n, bs):
                batched.extend(
                    _t2i_generate_seeded(name, prompts[i : i + bs], seeds[i : i + bs])
                )
            torch.cuda.synchronize()
            per_item = (time.perf_counter() - t0) / n
            deltas = [
                float(np.abs(np.asarray(b, dtype=np.float32) - r).mean())
                for b, r in zip(batched, ref_arrs)
            ]
            if dump_dir:
                for i, im in enumerate(batched):
                    im.save(os.path.join(dump_dir, f"{name}_{i}_b{bs}.png"))
            result["batches"][bs] = {
                "per_item_s": per_item,
                "parity_mean_abs_delta": float(np.mean(deltas)),
                "parity_max_abs_delta": float(np.max(deltas)),
                "status": "ok",
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            result["batches"][bs] = {"status": "oom"}
            break
        except Exception as e:  # noqa: BLE001 - surface the failure in the report
            torch.cuda.empty_cache()
            gc.collect()
            result["batches"][bs] = {"status": f"error: {e}"}
            break

    return result
