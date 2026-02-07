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

          for sentence in sentences:
              with torch.no_grad():
                  features = self.tokenize([sentence])
                  features = batch_to_device(features, device)
                  out_features = self.forward(features)
                  embedding = out_features["sentence_embedding"]

                  if normalize_embeddings:
                      embedding = F.normalize(embedding, p=2, dim=1)

                  if convert_to_numpy:
                      embedding = embedding.cpu().numpy()[0]
                  elif convert_to_tensor:
                      embedding = embedding[0]
                  else:
                      embedding = embedding[0]

                  all_embeddings.append(embedding)

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
    "SDXLTurbo" => """
    from diffusers import AutoPipelineForText2Image
    _models["SDXLTurbo"] = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        use_fast=True,
    ).to("cuda")
    """,
    "FluxDev" => """
    from diffusers import FluxPipeline
    _models["FluxDev"] = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, use_fast=True
    ).to("cuda")
    """,
    "FluxSchnell" => """
    from diffusers import FluxPipeline
    _models["FluxSchnell"] = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, use_fast=True
    ).to("cuda")
    """,
    "Moondream" => """
    from transformers import AutoModelForCausalLM
    _models["Moondream"] = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision="2025-01-09", trust_remote_code=True
    ).to("cuda")
    """,
    "BLIP2" => """
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    torch.set_default_dtype(torch.float16)
    _blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    _blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto"
    )
    def _ensure_half(module):
        for param in module.parameters():
            param.data = param.data.to(torch.float16)
        for buf in module.buffers():
            buf.data = buf.data.to(torch.float16)
    _blip2_model.apply(_ensure_half)
    if not hasattr(_blip2_processor, "num_query_tokens"):
        _blip2_processor.num_query_tokens = _blip2_model.config.num_query_tokens
    _models["BLIP2"] = {"processor": _blip2_processor, "model": _blip2_model}
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
end
