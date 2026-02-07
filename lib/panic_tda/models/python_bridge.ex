defmodule PanicTda.Models.PythonBridge do
  @moduledoc """
  Python bridge for real ML model execution via Snex.
  Manages one-time setup and lazy model loading into a persistent Python environment.
  """

  @setup_code """
  import warnings
  warnings.filterwarnings("ignore")

  import os
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

  import io
  import base64
  import hashlib
  import random

  try:
      import transformers
      transformers.logging.set_verbosity_error()
  except Exception:
      pass

  try:
      import diffusers
      diffusers.logging.set_verbosity_error()
  except Exception:
      pass

  try:
      from tqdm import tqdm
      from functools import partialmethod
      tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
  except Exception:
      pass

  import torch
  import numpy as np
  from PIL import Image
  import torch.nn.functional as F
  from sentence_transformers import SentenceTransformer
  from sentence_transformers.util import batch_to_device

  IMAGE_SIZE = 256
  EMBEDDING_DIM = 768

  class NoSortingSentenceTransformer(SentenceTransformer):
      def encode(
          self,
          sentences,
          batch_size=32,
          show_progress_bar=None,
          output_value="sentence_embedding",
          convert_to_numpy=True,
          convert_to_tensor=False,
          device=None,
          normalize_embeddings=False,
          precision="float32",
          **kwargs,
      ):
          self.eval()

          input_was_string = False
          if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
              sentences = [sentences]
              input_was_string = True

          if device is None:
              device = self.device

          all_embeddings = []

          with torch.no_grad():
              for i in range(0, len(sentences), batch_size):
                  batch = sentences[i:i+batch_size]
                  features = self.tokenize(batch)
                  features = batch_to_device(features, device)
                  out_features = self.forward(features)
                  embeddings = out_features["sentence_embedding"]

                  if normalize_embeddings:
                      embeddings = F.normalize(embeddings, p=2, dim=1)

                  if convert_to_numpy:
                      all_embeddings.extend(embeddings.cpu().numpy())
                  else:
                      all_embeddings.extend(embeddings)

          if convert_to_tensor and not convert_to_numpy and len(all_embeddings) > 0:
              all_embeddings = torch.stack(all_embeddings)

          if input_was_string:
              return all_embeddings[0]

          return all_embeddings

  def _load_sentence_transformer(name, model_path, **kwargs):
      try:
          _m = NoSortingSentenceTransformer(model_path, **kwargs)
      except FileNotFoundError:
          _ = SentenceTransformer(model_path, **kwargs)
          _m = NoSortingSentenceTransformer(model_path, **kwargs)
      if torch.cuda.is_available():
          _m = _m.to("cuda")
      _m.eval()
      _models[name] = _m

  _models = {}
  _panic_setup_done = True
  """

  @model_loaders %{
    "SD35Medium" => """
    from diffusers import StableDiffusion3Pipeline
    _pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        text_encoder_3=None, tokenizer_3=None,
        torch_dtype=torch.bfloat16, use_fast=True,
    )
    _pipe.enable_model_cpu_offload()
    _models["SD35Medium"] = _pipe
    """,
    "FluxDev" => """
    from diffusers import FluxPipeline
    _pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, use_fast=True
    )
    _pipe.enable_model_cpu_offload()
    _models["FluxDev"] = _pipe
    """,
    "FluxSchnell" => """
    from diffusers import FluxPipeline
    _pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, use_fast=True
    )
    _pipe.enable_model_cpu_offload()
    _models["FluxSchnell"] = _pipe
    """,
    "ZImageTurbo" => """
    from diffusers import ZImagePipeline
    _pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16
    )
    _pipe.enable_model_cpu_offload()
    _models["ZImageTurbo"] = _pipe
    """,
    "Flux2Klein" => """
    from diffusers import Flux2KleinPipeline
    _pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16,
    )
    _pipe.enable_model_cpu_offload()
    _models["Flux2Klein"] = _pipe
    """,
    "Moondream" => """
    from transformers import AutoModelForCausalLM
    _models["Moondream"] = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision="2025-06-21", trust_remote_code=True
    ).to("cuda")
    """,
    "InstructBLIP" => """
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    _iblip_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    _iblip_model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl", torch_dtype=torch.float16
    ).to("cuda")
    _models["InstructBLIP"] = {"processor": _iblip_processor, "model": _iblip_model}
    """,
    "Qwen25VL" => """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    _qwen_vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
    )
    _qwen_vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    _models["Qwen25VL"] = {"processor": _qwen_vl_processor, "model": _qwen_vl_model}
    """,
    "Gemma3n" => """
    from transformers import Gemma3nForConditionalGeneration, AutoProcessor
    _gemma3n_model = Gemma3nForConditionalGeneration.from_pretrained(
        "google/gemma-3n-E2B-it", torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    _gemma3n_processor = AutoProcessor.from_pretrained("google/gemma-3n-E2B-it")
    _models["Gemma3n"] = {"processor": _gemma3n_processor, "model": _gemma3n_model}
    """,
    "STSBMpnet" => """
    _load_sentence_transformer("STSBMpnet", "sentence-transformers/stsb-mpnet-base-v2")
    """,
    "STSBRoberta" => """
    _load_sentence_transformer("STSBRoberta", "sentence-transformers/stsb-roberta-base-v2")
    """,
    "STSBDistilRoberta" => """
    _load_sentence_transformer("STSBDistilRoberta", "sentence-transformers/stsb-distilroberta-base-v2")
    """,
    "Nomic" => """
    _load_sentence_transformer("Nomic", "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
    """,
    "JinaClip" => """
    from transformers import AutoModel
    _m = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to("cuda").eval()
    _models["JinaClip"] = _m
    """,
    "Qwen3Embed" => """
    _load_sentence_transformer(
        "Qwen3Embed", "Qwen/Qwen3-Embedding-4B",
        model_kwargs={"attn_implementation": "sdpa"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    """,
    "NomicVision" => """
    from transformers import AutoImageProcessor, AutoModel
    _nomic_vis_processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
    _nomic_vis_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
    ).to("cuda").eval()
    _models["NomicVision"] = {"processor": _nomic_vis_processor, "model": _nomic_vis_model}
    """,
    "JinaClipVision" => """
    from transformers import AutoModel
    _m = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to("cuda").eval()
    _models["JinaClipVision"] = _m
    """
  }

  @load_timeout 600_000

  def ensure_setup(env) do
    case Snex.pyeval(env, "_panic_setup_done", %{}, returning: "_panic_setup_done") do
      {:ok, true} ->
        :ok

      _ ->
        case Snex.pyeval(env, @setup_code, %{}, returning: "_panic_setup_done", timeout: @load_timeout) do
          {:ok, true} -> :ok
          error -> error
        end
    end
  end

  def ensure_model_loaded(env, model_name) do
    check_code = "result = model_name in _models"

    case Snex.pyeval(env, check_code, %{"model_name" => model_name}, returning: "result") do
      {:ok, true} ->
        :ok

      {:ok, false} ->
        load_code = Map.fetch!(@model_loaders, model_name)

        case Snex.pyeval(env, load_code, %{}, returning: "True", timeout: @load_timeout) do
          {:ok, _} -> :ok
          error -> error
        end

      error ->
        error
    end
  end

  @unload_timeout 30_000

  def unload_model(env, model_name) do
    code = """
    if model_name in _models:
        _obj = _models.pop(model_name)
        if hasattr(_obj, 'remove_all_hooks'):
            _obj.remove_all_hooks()
        if isinstance(_obj, dict):
            for _v in _obj.values():
                if hasattr(_v, 'cpu'):
                    _v.cpu()
                del _v
        elif hasattr(_obj, 'cpu'):
            _obj.cpu()
        del _obj
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    """

    case Snex.pyeval(env, code, %{"model_name" => model_name}, returning: "True",
           timeout: @unload_timeout) do
      {:ok, _} -> :ok
      error -> error
    end
  end

  def unload_all_models(env) do
    code = """
    if "_models" in dir() and _models:
        for _name in list(_models.keys()):
            _obj = _models.pop(_name)
            if hasattr(_obj, 'remove_all_hooks'):
                _obj.remove_all_hooks()
            if isinstance(_obj, dict):
                for _v in _obj.values():
                    if hasattr(_v, 'cpu'):
                        _v.cpu()
                    del _v
            elif hasattr(_obj, 'cpu'):
                _obj.cpu()
            del _obj
        import gc
        gc.collect()
        torch.set_default_dtype(torch.float32)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    """

    case Snex.pyeval(env, code, %{}, returning: "True", timeout: @unload_timeout) do
      {:ok, _} -> :ok
      error -> error
    end
  end
end
