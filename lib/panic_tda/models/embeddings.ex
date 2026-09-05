defmodule PanicTda.Models.Embeddings do
  @moduledoc """
  Embedding model invocation via Python interop.

  Text only. Image embedding was removed once it had proved to add nothing:
  in an alternating text-to-image network every second state is already text,
  and the caption at step n is itself a representation of the image at step
  n-1, so the image states are still observed --- through the captioner, which
  is the thing under study.

  Uses inline Python implementations for dummy models and real HuggingFace
  models via PythonBridge.
  """

  alias PanicTda.Models.PythonBridge

  @dummy_text_models ~w(DummyText DummyText2)
  @real_text_models ~w(Qwen3Embed)
  @text_models @dummy_text_models ++ @real_text_models

  @embed_timeout 60_000

  def list_models, do: @text_models

  @doc """
  Whether this embedding model is still registered.

  Historical experiments name embedders that have since been retired, so read
  paths over old data have to skip them rather than raise the way the compute
  path does.
  """
  def registered?(model_name), do: model_name in @text_models

  def embed(env, model_name, contents) when model_name in @dummy_text_models do
    embed_dummy_text(env, model_name, contents)
  end

  def embed(env, model_name, contents) when model_name in @real_text_models do
    embed_real_text(env, model_name, contents)
  end

  defp embed_real_text(env, model_name, texts) when is_list(texts) do
    with :ok <- PythonBridge.ensure_setup(env),
         :ok <- PythonBridge.ensure_model_loaded(env, model_name) do
      case Snex.pyeval(
             env,
             "return panic_models.embed_text(model_name, texts)",
             %{"model_name" => model_name, "texts" => texts},
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

           EMBEDDING_DIM = 256
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
           return embeddings
           """,
           %{"texts" => texts, "version" => version}
         ) do
      {:ok, base64_list} -> {:ok, Enum.map(base64_list, &Base.decode64!/1)}
      error -> error
    end
  end
end
