defmodule PanicTda.Models.GenAI do
  @moduledoc """
  GenAI model invocation via Python interop.
  Supports text-to-image and image-to-text models.
  Uses inline Python implementations for dummy models.
  """

  @t2i_models ~w(DummyT2I DummyT2I2)
  @i2t_models ~w(DummyI2T DummyI2T2)

  def list_models do
    @t2i_models ++ @i2t_models
  end

  def output_type(model_name) when model_name in @t2i_models, do: :image
  def output_type(model_name) when model_name in @i2t_models, do: :text
  def output_type(model_name), do: raise("Unknown model: #{model_name}")

  def invoke(env, model_name, input, seed) when model_name in @t2i_models do
    invoke_t2i(env, model_name, input, seed)
  end

  def invoke(env, model_name, input, seed) when model_name in @i2t_models do
    invoke_i2t(env, model_name, input, seed)
  end

  defp invoke_t2i(env, model_name, prompt, seed) when is_binary(prompt) do
    color_offset = if model_name == "DummyT2I2", do: 2000, else: 0

    case Snex.pyeval(
           env,
           """
           import hashlib
           import random
           import io
           import base64
           from PIL import Image

           IMAGE_SIZE = 256

           prompt_hash = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)

           if seed == -1:
               import time
               microseconds = int(time.time() * 1000000)
               r = microseconds % 100
               g = 200 + (microseconds % 56)
               b = 200 + ((microseconds // 100) % 56)
           else:
               random.seed(seed + prompt_hash + color_offset)
               if color_offset > 0:
                   r = random.randint(200, 255)
                   g = random.randint(0, 100)
                   b = random.randint(200, 255)
               else:
                   r = random.randint(0, 100)
                   g = random.randint(200, 255)
                   b = random.randint(200, 255)

           image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(r, g, b))
           buffer = io.BytesIO()
           image.save(buffer, format="WEBP", lossless=True)
           result = base64.b64encode(buffer.getvalue()).decode('ascii')
           """,
           %{"prompt" => prompt, "seed" => seed, "color_offset" => color_offset},
           returning: "result"
         ) do
      {:ok, base64_data} -> {:ok, Base.decode64!(base64_data)}
      error -> error
    end
  end

  defp invoke_i2t(env, model_name, image_binary, seed) when is_binary(image_binary) do
    word_offset = if model_name == "DummyI2T2", do: 1000, else: 0
    image_b64 = Base.encode64(image_binary)

    Snex.pyeval(
      env,
      """
      import hashlib
      import random
      import io
      import base64
      from PIL import Image

      image_bytes = base64.b64decode(image_b64)
      image = Image.open(io.BytesIO(image_bytes))

      pixels = list(image.getdata())
      sample_pixels = pixels[:min(10, len(pixels))]
      pixel_bytes = str(sample_pixels).encode()
      input_hash = int(hashlib.sha256(pixel_bytes).hexdigest()[:8], 16)

      if seed == -1:
          import uuid
          unique_id = str(uuid.uuid4())[:8]
          result = f"dummy text caption {unique_id}"
      else:
          random.seed(seed + input_hash + word_offset)
          if word_offset > 0:
              adjectives = ["colorful", "monochrome", "sharp", "blurred", "dynamic", "static", "bold", "subtle"]
              nouns = ["display", "capture", "moment", "frame", "snapshot", "impression", "rendering", "portrayal"]
          else:
              adjectives = ["bright", "dark", "vivid", "muted", "complex", "simple", "detailed", "abstract"]
              nouns = ["scene", "image", "picture", "view", "composition", "artwork", "photo", "visual"]
          adj = adjectives[random.randint(0, len(adjectives) - 1)]
          noun = nouns[random.randint(0, len(nouns) - 1)]
          num = random.randint(100, 999) if word_offset == 0 else random.randint(1000, 9999)
          prefix = "dummy caption:" if word_offset == 0 else "dummy v2:"
          result = f"{prefix} {adj} {noun} \#{num}"
      """,
      %{"image_b64" => image_b64, "seed" => seed, "word_offset" => word_offset},
      returning: "result"
    )
  end
end
