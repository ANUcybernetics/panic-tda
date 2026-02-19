"""PANIC-TDA model registry: setup, loading, invoke, and embedding functions.

All functions are called from Elixir via Snex.pyeval(). The module is imported
once during PythonBridge.ensure_setup/1 and then used as panic_models.*.
"""

from __future__ import annotations

import base64
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
EMBEDDING_DIM = 768

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
        import transformers.modeling_utils as _tmu

        _orig_ptm_init = _tmu.PreTrainedModel.__init__

        def _patched_ptm_init(self: Any, *args: Any, **kwargs: Any) -> None:
            _orig_ptm_init(self, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}

        _tmu.PreTrainedModel.__init__ = _patched_ptm_init
    except Exception:
        pass

    try:
        import diffusers

        diffusers.logging.set_verbosity_error()
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
                features = self.tokenize(batch)
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


def _load_remote_code_model(
    model_path: str, model_cls: Any = None, **kwargs: Any
) -> Any:
    import contextlib

    import transformers.modeling_utils as _tmu

    if model_cls is None:
        from transformers import AutoModel

        model_cls = AutoModel
    _orig_ctx = _tmu.PreTrainedModel.get_init_context

    @classmethod  # type: ignore[misc]
    def _cpu_init_ctx(cls: Any, *a: Any, **kw: Any) -> list[Any]:
        return [contextlib.nullcontext()]

    _tmu.PreTrainedModel.get_init_context = _cpu_init_ctx
    try:
        model = model_cls.from_pretrained(model_path, trust_remote_code=True, **kwargs)
        model.tie_weights()
        return model
    finally:
        _tmu.PreTrainedModel.get_init_context = _orig_ctx


def _encode_image_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", lossless=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_image_b64(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _encode_embedding(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


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
        "extra_kwargs": {
            "text_encoder_3": None,
            "tokenizer_3": None,
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
    "HunyuanImage": {
        "pipeline_cls": "HunyuanImagePipeline",
        "repo": "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
        "offload": "sequential_cpu_offload",
        "offload_only": True,
        "extra_kwargs": {"torch_dtype": "bfloat16"},
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
    "QwenImage": {
        "pipeline_cls": "QwenImagePipeline",
        "repo": "Qwen/Qwen-Image-2512",
        "offload": "model_cpu_offload",
        "offload_only": True,
        "quantize": True,
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
    pipe = pipeline_cls.from_pretrained(cfg["repo"], **kwargs)
    getattr(pipe, f"enable_{cfg['offload']}")()
    _models[name] = pipe
    if cfg.get("offload_only"):
        _models_offload_only.add(name)


# ---------------------------------------------------------------------------
# Model loading: I2T model registry
# ---------------------------------------------------------------------------


def _load_moondream() -> None:
    from transformers import AutoModelForCausalLM

    _models["Moondream"] = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision="2025-06-21", trust_remote_code=True
    ).to("cuda")


def _load_pixtral() -> None:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(
        "mistral-community/pixtral-12b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=_bnb_4bit_config(),
    )
    processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    _models["Pixtral"] = {"processor": processor, "model": model}
    _models_offload_only.add("Pixtral")


def _load_llama32vision() -> None:
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    model = MllamaForConditionalGeneration.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=True,
        quantization_config=_bnb_4bit_config(),
    )
    processor = AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct", token=True
    )
    _models["LLaMA32Vision"] = {"processor": processor, "model": model}
    _models_offload_only.add("LLaMA32Vision")


def _load_phi4vision() -> None:
    from transformers import AutoProcessor, Phi4MultimodalForCausalLM

    model = (
        Phi4MultimodalForCausalLM.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct",
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
        .to("cuda")
        .eval()
    )
    processor = AutoProcessor.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    _models["Phi4Vision"] = {"processor": processor, "model": model}


def _load_qwen25vl() -> None:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        quantization_config=_bnb_4bit_config(),
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    _models["Qwen25VL"] = {"processor": processor, "model": model}
    _models_offload_only.add("Qwen25VL")


def _load_gemma3n() -> None:
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    model = (
        Gemma3nForConditionalGeneration.from_pretrained(
            "google/gemma-3n-E2B-it", torch_dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    processor = AutoProcessor.from_pretrained("google/gemma-3n-E2B-it")
    _models["Gemma3n"] = {"processor": processor, "model": model}


# ---------------------------------------------------------------------------
# Model loading: embedding models
# ---------------------------------------------------------------------------


def _load_nomic_vision() -> None:
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
    model = _load_remote_code_model(
        "nomic-ai/nomic-embed-vision-v1.5", torch_dtype=torch.float32
    )
    model = model.to("cuda").eval()
    _models["NomicVision"] = {"processor": processor, "model": model}


def _load_jina_clip_vision() -> None:
    m = _load_remote_code_model("jinaai/jina-clip-v2")
    m = m.to("cuda").eval()
    _models["JinaClipVision"] = m


def _load_jina_clip() -> None:
    m = _load_remote_code_model("jinaai/jina-clip-v2")
    m = m.to("cuda").eval()
    _models["JinaClip"] = m


# ---------------------------------------------------------------------------
# Unified model loader dispatch
# ---------------------------------------------------------------------------

_I2T_LOADERS: dict[str, Any] = {
    "Moondream": _load_moondream,
    "Pixtral": _load_pixtral,
    "LLaMA32Vision": _load_llama32vision,
    "Phi4Vision": _load_phi4vision,
    "Qwen25VL": _load_qwen25vl,
    "Gemma3n": _load_gemma3n,
}

_EMBEDDING_LOADERS: dict[str, tuple[str, dict[str, Any]]] = {
    "STSBMpnet": ("sentence-transformers/stsb-mpnet-base-v2", {}),
    "STSBRoberta": ("sentence-transformers/stsb-roberta-base-v2", {}),
    "STSBDistilRoberta": ("sentence-transformers/stsb-distilroberta-base-v2", {}),
    "Nomic": ("nomic-ai/nomic-embed-text-v2-moe", {"trust_remote_code": True}),
    "Qwen3Embed": (
        "Qwen/Qwen3-Embedding-4B",
        {
            "model_kwargs": {"attn_implementation": "sdpa"},
            "tokenizer_kwargs": {"padding_side": "left"},
        },
    ),
}

_VISION_EMBED_LOADERS: dict[str, Any] = {
    "NomicVision": _load_nomic_vision,
    "JinaClipVision": _load_jina_clip_vision,
    "JinaClip": _load_jina_clip,
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
    elif name in _VISION_EMBED_LOADERS:
        _VISION_EMBED_LOADERS[name]()
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


def unload_model(name: str) -> None:
    if name not in _models:
        return
    obj = _models.pop(name)
    _models_offload_only.discard(name)
    if hasattr(obj, "remove_all_hooks"):
        obj.remove_all_hooks()
    if isinstance(obj, dict):
        for v in obj.values():
            if hasattr(v, "cpu"):
                try:
                    v.cpu()
                except ValueError:
                    pass
            del v
    elif hasattr(obj, "cpu"):
        try:
            obj.cpu()
        except ValueError:
            pass
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_all_models() -> None:
    for name in list(_models.keys()):
        obj = _models.pop(name)
        if hasattr(obj, "remove_all_hooks"):
            obj.remove_all_hooks()
        if isinstance(obj, dict):
            for v in obj.values():
                if hasattr(v, "cpu"):
                    try:
                        v.cpu()
                    except ValueError:
                        pass
                del v
        elif hasattr(obj, "cpu"):
            try:
                obj.cpu()
            except ValueError:
                pass
        del obj
    _models_offload_only.clear()
    gc.collect()
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# T2I invocation: config-driven
# ---------------------------------------------------------------------------

_T2I_INVOKE_CONFIGS: dict[str, dict[str, Any]] = {
    "SD35Medium": {"num_inference_steps": 20, "guidance_scale": 5.0},
    "Flux2Dev": {"num_inference_steps": 15, "guidance_scale": 3.5},
    "HunyuanImage": {"num_inference_steps": 25},
    "GLMImage": {"num_inference_steps": 25, "guidance_scale": 7.5},
    "ZImageTurbo": {"num_inference_steps": 8},
    "Flux2Klein": {"num_inference_steps": 4, "guidance_scale": 1.0},
    "QwenImage": {"num_inference_steps": 25, "true_cfg_scale": 4.0},
}

_T2I_BATCH_CAPABLE: set[str] = {"SD35Medium", "ZImageTurbo", "Flux2Klein"}


def invoke_t2i(name: str, prompt: str) -> str:
    """Run a single T2I inference. Returns base64-encoded WEBP."""
    cfg = _T2I_INVOKE_CONFIGS[name]
    img = _models[name](
        prompt=prompt,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        generator=None,
        **cfg,
    ).images[0]
    return _encode_image_b64(img)


def invoke_t2i_batch(name: str, prompts: list[str]) -> list[str]:
    """Run batch T2I inference. Returns list of base64-encoded WEBP."""
    cfg = _T2I_INVOKE_CONFIGS[name]
    if name in _T2I_BATCH_CAPABLE:
        imgs = _models[name](
            prompt=prompts,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            generator=None,
            **cfg,
        ).images
        return [_encode_image_b64(img) for img in imgs]
    else:
        results = []
        for p in prompts:
            img = _models[name](
                prompt=p,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                generator=None,
                **cfg,
            ).images[0]
            results.append(_encode_image_b64(img))
        return results


# ---------------------------------------------------------------------------
# I2T invocation: strategy-based dispatch
# ---------------------------------------------------------------------------


def invoke_i2t(name: str, image_b64: str) -> str:
    """Run a single I2T inference. Returns caption text."""
    img = _decode_image_b64(image_b64)
    return _I2T_STRATEGIES[name](name, img)


def invoke_i2t_batch(name: str, b64_list: list[str]) -> list[str]:
    """Run batch I2T inference. Returns list of caption texts."""
    images = [_decode_image_b64(b) for b in b64_list]
    return _I2T_BATCH_STRATEGIES[name](name, images)


# --- Moondream ---


_MOONDREAM_SETTINGS: dict[str, Any] = {
    "temperature": 0.0,
    "max_tokens": 256,
    "top_p": 1.0,
    "variant": None,
}


def _invoke_moondream(_name: str, img: Image.Image) -> str:
    with torch.inference_mode():
        cap = _models["Moondream"].caption(
            img, length="short", settings=_MOONDREAM_SETTINGS
        )
    return cap["caption"].strip()


def _invoke_moondream_batch(_name: str, images: list[Image.Image]) -> list[str]:
    results = []
    with torch.inference_mode():
        for img in images:
            cap = _models["Moondream"].caption(
                img, length="short", settings=_MOONDREAM_SETTINGS
            )
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
        "batch_images_fn": lambda img: img,
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
            **inputs, max_new_tokens=128, **cfg["extra_generate_kwargs"]
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
            **inputs, max_new_tokens=128, **cfg["extra_generate_kwargs"]
        )
        gen_ids = gen_ids[:, inputs["input_ids"].shape[1] :]
    return [
        s.strip()
        for s in model_dict["processor"].batch_decode(gen_ids, skip_special_tokens=True)
    ]


# --- Phi4Vision ---


def _invoke_phi4vision(_name: str, img: Image.Image) -> str:
    phi4 = _models["Phi4Vision"]
    messages = [{"role": "user", "content": "<|image_1|>\nDescribe this image."}]
    text = phi4["processor"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = phi4["processor"](text=text, images=[img], return_tensors="pt")
    inputs = inputs.to(phi4["model"].device)
    with torch.no_grad():
        gen_ids = phi4["model"].generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_image_embeds=inputs.get("input_image_embeds"),
            input_audio_embeds=inputs.get("input_audio_embeds"),
            input_mode=inputs.get("input_mode"),
            max_new_tokens=128,
            do_sample=False,
        )
        gen_ids = gen_ids[:, inputs["input_ids"].shape[1] :]
    return phi4["processor"].batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def _invoke_phi4vision_batch(_name: str, images: list[Image.Image]) -> list[str]:
    phi4 = _models["Phi4Vision"]
    all_texts = []
    for img in images:
        messages = [{"role": "user", "content": "<|image_1|>\nDescribe this image."}]
        all_texts.append(
            phi4["processor"].apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    inputs = phi4["processor"](
        text=all_texts, images=images, padding=True, return_tensors="pt"
    )
    inputs = inputs.to(phi4["model"].device)
    with torch.no_grad():
        gen_ids = phi4["model"].generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_image_embeds=inputs.get("input_image_embeds"),
            input_audio_embeds=inputs.get("input_audio_embeds"),
            input_mode=inputs.get("input_mode"),
            max_new_tokens=128,
            do_sample=False,
        )
        gen_ids = gen_ids[:, inputs["input_ids"].shape[1] :]
    return [
        s.strip()
        for s in phi4["processor"].batch_decode(gen_ids, skip_special_tokens=True)
    ]


# --- Qwen25VL ---


def _invoke_qwen25vl(_name: str, img: Image.Image) -> str:
    from qwen_vl_utils import process_vision_info

    qwen_vl = _models["Qwen25VL"]
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
        gen_ids = qwen_vl["model"].generate(**inputs, max_new_tokens=128)
        gen_ids = gen_ids[:, inputs["input_ids"].shape[1] :]
    return (
        qwen_vl["processor"].batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    )


def _invoke_qwen25vl_batch(_name: str, images: list[Image.Image]) -> list[str]:
    from qwen_vl_utils import process_vision_info

    qwen_vl = _models["Qwen25VL"]
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
        gen_ids = qwen_vl["model"].generate(**inputs, max_new_tokens=128)
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
            **inputs, max_new_tokens=100, do_sample=False
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
            **inputs, max_new_tokens=100, do_sample=False
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
    "Phi4Vision": _invoke_phi4vision,
    "Qwen25VL": _invoke_qwen25vl,
    "Gemma3n": _invoke_gemma3n,
}

_I2T_BATCH_STRATEGIES: dict[str, Any] = {
    "Moondream": _invoke_moondream_batch,
    "Pixtral": _invoke_chat_template_batch,
    "LLaMA32Vision": _invoke_chat_template_batch,
    "Phi4Vision": _invoke_phi4vision_batch,
    "Qwen25VL": _invoke_qwen25vl_batch,
    "Gemma3n": _invoke_gemma3n_batch,
}


# ---------------------------------------------------------------------------
# Embeddings: text
# ---------------------------------------------------------------------------


def embed_text(name: str, texts: list[str]) -> list[str]:
    """Embed texts. Returns list of base64-encoded float32 vectors."""
    with torch.no_grad():
        if name in ("STSBMpnet", "STSBRoberta", "STSBDistilRoberta"):
            embs = _models[name].encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
        elif name == "Nomic":
            embs = _models[name].encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                prompt_name="passage",
            )
        elif name == "JinaClip":
            embs = _models[name].encode_text(
                texts, truncate_dim=EMBEDDING_DIM, task="retrieval.query"
            )
            return [_encode_embedding(np.array(e)) for e in embs]
        elif name == "Qwen3Embed":
            embs = _models[name].encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
            return [_encode_embedding(e[:EMBEDDING_DIM]) for e in embs]
        else:
            raise ValueError(f"Unknown text embedding model: {name}")

        if isinstance(embs, np.ndarray) and embs.ndim > 1:
            return [_encode_embedding(e) for e in embs]
        else:
            return [_encode_embedding(np.array(e)) for e in embs]


# ---------------------------------------------------------------------------
# Embeddings: images
# ---------------------------------------------------------------------------


def embed_images(name: str, b64_list: list[str]) -> list[str]:
    """Embed images. Returns list of base64-encoded float32 vectors."""
    images = [_decode_image_b64(b).convert("RGB") for b in b64_list]

    if name == "NomicVision":
        return _embed_nomic_vision(images)
    elif name == "JinaClipVision":
        return _embed_jina_clip_vision(images)
    else:
        raise ValueError(f"Unknown image embedding model: {name}")


def _embed_nomic_vision(images: list[Image.Image]) -> list[str]:
    nomic_vis = _models["NomicVision"]
    embeddings: list[Any] = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = images[i : i + 32]
            inputs = nomic_vis["processor"](batch, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = nomic_vis["model"](**inputs)
            batch_embs = outputs.last_hidden_state[:, 0]
            norms = torch.norm(batch_embs, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            batch_embs = batch_embs / norms
            if torch.isnan(batch_embs).any():
                raise ValueError(f"NomicVision produced NaN embeddings for batch {i}")
            batch_embs = batch_embs.cpu().numpy()
            embeddings.extend(list(batch_embs))
    return [_encode_embedding(e) for e in embeddings]


def _embed_jina_clip_vision(images: list[Image.Image]) -> list[str]:
    with torch.no_grad():
        embs = _models["JinaClipVision"].encode_image(
            images, truncate_dim=EMBEDDING_DIM
        )
        embs_np = [np.array(e).astype(np.float32) for e in embs]
        for idx, e in enumerate(embs_np):
            if np.isnan(e).any():
                raise ValueError(
                    f"JinaClipVision produced NaN embeddings for image {idx}"
                )
        return [base64.b64encode(e.tobytes()).decode("ascii") for e in embs_np]
