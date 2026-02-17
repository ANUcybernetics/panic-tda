defmodule PanicTda.Models.GenAI do
  @moduledoc """
  GenAI model invocation via Python interop.
  Supports text-to-image and image-to-text models.
  Uses inline Python implementations for dummy models
  and real HuggingFace models via PythonBridge.
  """

  alias PanicTda.Models.{ImageConverter, PythonBridge}

  @dummy_t2i_models ~w(DummyT2I DummyT2I2)
  @dummy_i2t_models ~w(DummyI2T DummyI2T2)
  @real_t2i_models ~w(SD35Medium ZImageTurbo Flux2Klein Flux2Dev QwenImage HunyuanImage GLMImage)
  @real_i2t_models ~w(Moondream Qwen25VL Gemma3n Pixtral LLaMA32Vision Phi4Vision)
  @t2i_models @dummy_t2i_models ++ @real_t2i_models
  @i2t_models @dummy_i2t_models ++ @real_i2t_models

  @t2i_timeout 300_000
  @i2t_timeout 60_000

  def list_models do
    @t2i_models ++ @i2t_models
  end

  def output_type(model_name) when model_name in @t2i_models, do: :image
  def output_type(model_name) when model_name in @i2t_models, do: :text
  def output_type(model_name), do: raise("Unknown model: #{model_name}")

  def invoke(env, model_name, input) when model_name in @dummy_t2i_models do
    invoke_dummy_t2i(env, model_name, input)
  end

  def invoke(env, model_name, input) when model_name in @dummy_i2t_models do
    invoke_dummy_i2t(env, model_name, input)
  end

  def invoke(env, model_name, input) when model_name in @real_t2i_models do
    invoke_real_t2i(env, model_name, input)
  end

  def invoke(env, model_name, input) when model_name in @real_i2t_models do
    invoke_real_i2t(env, model_name, input)
  end

  def invoke_batch(env, model_name, inputs) when model_name in @dummy_t2i_models do
    invoke_batch_dummy_t2i(env, model_name, inputs)
  end

  def invoke_batch(env, model_name, inputs) when model_name in @dummy_i2t_models do
    invoke_batch_dummy_i2t(env, model_name, inputs)
  end

  def invoke_batch(env, model_name, inputs) when model_name in @real_t2i_models do
    invoke_batch_real_t2i(env, model_name, inputs)
  end

  def invoke_batch(env, model_name, inputs) when model_name in @real_i2t_models do
    invoke_batch_real_i2t(env, model_name, inputs)
  end

  defp invoke_real_t2i(env, model_name, prompt) when is_binary(prompt) do
    with :ok <- PythonBridge.swap_model_to_gpu(env, model_name) do
      case Snex.pyeval(
             env,
             "result = panic_models.invoke_t2i(model_name, prompt)",
             %{"model_name" => model_name, "prompt" => prompt},
             returning: "result",
             timeout: @t2i_timeout
           ) do
        {:ok, base64_data} -> {:ok, base64_data |> Base.decode64!() |> ImageConverter.to_avif!()}
        error -> error
      end
    end
  end

  defp invoke_real_i2t(env, model_name, image_binary) when is_binary(image_binary) do
    image_b64 = Base.encode64(image_binary)

    with :ok <- PythonBridge.swap_model_to_gpu(env, model_name) do
      case Snex.pyeval(
             env,
             "result = panic_models.invoke_i2t(model_name, image_b64)",
             %{"model_name" => model_name, "image_b64" => image_b64},
             returning: "result",
             timeout: @i2t_timeout
           ) do
        {:ok, text} -> {:ok, ensure_nonempty_text(text)}
        error -> error
      end
    end
  end

  defp invoke_batch_real_t2i(env, model_name, prompts) do
    with :ok <- PythonBridge.swap_model_to_gpu(env, model_name) do
      case Snex.pyeval(
             env,
             "result = panic_models.invoke_t2i_batch(model_name, prompts)",
             %{"model_name" => model_name, "prompts" => prompts},
             returning: "result",
             timeout: @t2i_timeout * length(prompts)
           ) do
        {:ok, base64_list} ->
          {:ok, Enum.map(base64_list, fn b64 -> b64 |> Base.decode64!() |> ImageConverter.to_avif!() end)}

        error ->
          error
      end
    end
  end

  defp invoke_batch_real_i2t(env, model_name, images) do
    image_b64_list = Enum.map(images, &Base.encode64/1)

    with :ok <- PythonBridge.swap_model_to_gpu(env, model_name) do
      case Snex.pyeval(
             env,
             "result = panic_models.invoke_i2t_batch(model_name, image_b64_list)",
             %{"model_name" => model_name, "image_b64_list" => image_b64_list},
             returning: "result",
             timeout: @i2t_timeout * length(images)
           ) do
        {:ok, texts} -> {:ok, Enum.map(texts, &ensure_nonempty_text/1)}
        error -> error
      end
    end
  end

  defp invoke_dummy_t2i(env, model_name, prompt) when is_binary(prompt) do
    color_offset = if model_name == "DummyT2I2", do: 2000, else: 0

    case Snex.pyeval(
           env,
           """
           import hashlib
           import time
           import io
           import base64
           from PIL import Image

           IMAGE_SIZE = 256

           microseconds = int(time.time() * 1000000)
           r = (microseconds + color_offset) % 256
           g = (microseconds // 100 + color_offset) % 256
           b = (microseconds // 10000 + color_offset) % 256

           image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(r, g, b))
           buffer = io.BytesIO()
           image.save(buffer, format="WEBP", lossless=True)
           result = base64.b64encode(buffer.getvalue()).decode('ascii')
           """,
           %{"color_offset" => color_offset},
           returning: "result"
         ) do
      {:ok, base64_data} -> {:ok, base64_data |> Base.decode64!() |> ImageConverter.to_avif!()}
      error -> error
    end
  end

  defp invoke_dummy_i2t(env, model_name, image_binary) when is_binary(image_binary) do
    prefix = if model_name == "DummyI2T2", do: "dummy v2:", else: "dummy caption:"
    image_b64 = Base.encode64(image_binary)

    Snex.pyeval(
      env,
      """
      import uuid
      import base64

      unique_id = str(uuid.uuid4())[:8]
      result = f"{prefix} {unique_id}"
      """,
      %{"image_b64" => image_b64, "prefix" => prefix},
      returning: "result"
    )
  end

  defp invoke_batch_dummy_t2i(env, model_name, prompts) do
    color_offset = if model_name == "DummyT2I2", do: 2000, else: 0

    case Snex.pyeval(
           env,
           """
           import time
           import io
           import base64
           from PIL import Image

           IMAGE_SIZE = 256
           _results = []
           for _i, _prompt in enumerate(prompts):
               microseconds = int(time.time() * 1000000) + _i
               r = (microseconds + color_offset) % 256
               g = (microseconds // 100 + color_offset) % 256
               b = (microseconds // 10000 + color_offset) % 256
               _image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(r, g, b))
               _buffer = io.BytesIO()
               _image.save(_buffer, format="WEBP", lossless=True)
               _results.append(base64.b64encode(_buffer.getvalue()).decode('ascii'))
           result = _results
           """,
           %{"prompts" => prompts, "color_offset" => color_offset},
           returning: "result"
         ) do
      {:ok, base64_list} -> {:ok, Enum.map(base64_list, fn b64 -> b64 |> Base.decode64!() |> ImageConverter.to_avif!() end)}
      error -> error
    end
  end

  defp ensure_nonempty_text(text) when is_binary(text) do
    if String.trim(text) == "", do: "[empty]", else: text
  end

  defp ensure_nonempty_text(_), do: "[empty]"

  defp invoke_batch_dummy_i2t(env, model_name, images) do
    prefix = if model_name == "DummyI2T2", do: "dummy v2:", else: "dummy caption:"
    image_b64_list = Enum.map(images, &Base.encode64/1)

    Snex.pyeval(
      env,
      """
      import uuid

      _results = []
      for _img_b64 in image_b64_list:
          _unique_id = str(uuid.uuid4())[:8]
          _results.append(f"{prefix} {_unique_id}")
      result = _results
      """,
      %{"image_b64_list" => image_b64_list, "prefix" => prefix},
      returning: "result"
    )
  end
end
