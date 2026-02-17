defmodule PanicTda.Models.Embeddings do
  @moduledoc """
  Embedding model invocation via Python interop.
  Supports text and image embedding models.
  Uses inline Python implementations for dummy models
  and real HuggingFace models via PythonBridge.
  """

  alias PanicTda.Models.PythonBridge

  @dummy_text_models ~w(DummyText DummyText2)
  @dummy_image_models ~w(DummyVision DummyVision2)
  @real_text_models ~w(STSBMpnet STSBRoberta STSBDistilRoberta Nomic JinaClip Qwen3Embed)
  @real_image_models ~w(NomicVision JinaClipVision)
  @text_models @dummy_text_models ++ @real_text_models
  @image_models @dummy_image_models ++ @real_image_models

  @embed_timeout 60_000

  def list_models do
    @text_models ++ @image_models
  end

  def model_type(model_name) when model_name in @text_models, do: :text
  def model_type(model_name) when model_name in @image_models, do: :image
  def model_type(model_name), do: raise("Unknown embedding model: #{model_name}")

  def embed(env, model_name, contents) when model_name in @dummy_text_models do
    embed_dummy_text(env, model_name, contents)
  end

  def embed(env, model_name, contents) when model_name in @dummy_image_models do
    embed_dummy_images(env, model_name, contents)
  end

  def embed(env, model_name, contents) when model_name in @real_text_models do
    embed_real_text(env, model_name, contents)
  end

  def embed(env, model_name, contents) when model_name in @real_image_models do
    embed_real_images(env, model_name, contents)
  end

  defp embed_real_text(env, model_name, texts) when is_list(texts) do
    with :ok <- PythonBridge.ensure_setup(env),
         :ok <- PythonBridge.ensure_model_loaded(env, model_name) do
      case Snex.pyeval(
             env,
             "result = panic_models.embed_text(model_name, texts)",
             %{"model_name" => model_name, "texts" => texts},
             returning: "result",
             timeout: @embed_timeout
           ) do
        {:ok, base64_list} -> {:ok, Enum.map(base64_list, &Base.decode64!/1)}
        error -> error
      end
    end
  end

  defp embed_real_images(env, model_name, image_binaries) when is_list(image_binaries) do
    image_b64_list = Enum.map(image_binaries, &Base.encode64/1)

    with :ok <- PythonBridge.ensure_setup(env),
         :ok <- PythonBridge.ensure_model_loaded(env, model_name) do
      case Snex.pyeval(
             env,
             "result = panic_models.embed_images(model_name, image_b64_list)",
             %{"model_name" => model_name, "image_b64_list" => image_b64_list},
             returning: "result",
             timeout: @embed_timeout
           ) do
        {:ok, base64_list} -> {:ok, Enum.map(base64_list, &Base.decode64!/1)}
        error -> error
      end
    end
  end

  defp embed_dummy_text(env, model_name, texts) when is_list(texts) do
    version = if model_name == "DummyText2", do: 2, else: 1

    case Snex.pyeval(
           env,
           """
           import numpy as np
           import base64

           EMBEDDING_DIM = 768
           embeddings = []

           for content in texts:
               if version == 1:
                   seed = sum(ord(c) for c in content)
                   np.random.seed(seed)
                   vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
               else:
                   chars = [ord(c) for c in (content[:100] if len(content) > 100 else content)]
                   chars = (chars + [0] * EMBEDDING_DIM)[:EMBEDDING_DIM]
                   vector = (np.array(chars) / 255.0).astype(np.float32)
               embeddings.append(base64.b64encode(vector.tobytes()).decode('ascii'))

           np.random.seed(None)
           result = embeddings
           """,
           %{"texts" => texts, "version" => version},
           returning: "result"
         ) do
      {:ok, base64_list} -> {:ok, Enum.map(base64_list, &Base.decode64!/1)}
      error -> error
    end
  end

  defp embed_dummy_images(env, model_name, image_binaries) when is_list(image_binaries) do
    version = if model_name == "DummyVision2", do: 2, else: 1
    image_b64_list = Enum.map(image_binaries, &Base.encode64/1)

    case Snex.pyeval(
           env,
           """
           import numpy as np
           import io
           import base64
           from PIL import Image

           EMBEDDING_DIM = 768
           embeddings = []

           for image_b64 in image_b64_list:
               image_bytes = base64.b64decode(image_b64)
               image = Image.open(io.BytesIO(image_bytes))
               width, height = image.size

               if version == 1:
                   pixels = list(image.getdata())[:100] if image.mode != "P" else []
                   pixel_sum = sum(sum(p) if isinstance(p, tuple) else p for p in pixels)
                   seed = (width + height + pixel_sum) % (2**32)
                   np.random.seed(seed)
                   vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
               else:
                   pixels = list(image.getdata())[:EMBEDDING_DIM]
                   if image.mode == "P":
                       pixel_values = pixels + [0] * (EMBEDDING_DIM - len(pixels))
                   else:
                       pixel_values = []
                       for p in pixels:
                           if isinstance(p, tuple):
                               pixel_values.extend(p)
                           else:
                               pixel_values.append(p)
                       pixel_values = (pixel_values + [0] * EMBEDDING_DIM)[:EMBEDDING_DIM]
                   vector = (np.array(pixel_values) / 255.0).astype(np.float32)

               embeddings.append(base64.b64encode(vector.tobytes()).decode('ascii'))

           np.random.seed(None)
           result = embeddings
           """,
           %{"image_b64_list" => image_b64_list, "version" => version},
           returning: "result"
         ) do
      {:ok, base64_list} -> {:ok, Enum.map(base64_list, &Base.decode64!/1)}
      error -> error
    end
  end
end
