defmodule PanicTda.Models.GenAI do
  @moduledoc """
  GenAI model invocation via Python interop.
  Supports text-to-image and image-to-text models.
  Uses inline Python implementations for dummy models
  and real HuggingFace models via PythonBridge.
  """

  alias PanicTda.Models.PythonBridge

  @dummy_t2i_models ~w(DummyT2I DummyT2I2)
  @dummy_i2t_models ~w(DummyI2T DummyI2T2)
  @real_t2i_models ~w(SDXLTurbo FluxDev FluxSchnell)
  @real_i2t_models ~w(Moondream BLIP2)
  @t2i_models @dummy_t2i_models ++ @real_t2i_models
  @i2t_models @dummy_i2t_models ++ @real_i2t_models

  @t2i_timeout 120_000
  @i2t_timeout 60_000

  def list_models do
    @t2i_models ++ @i2t_models
  end

  def output_type(model_name) when model_name in @t2i_models, do: :image
  def output_type(model_name) when model_name in @i2t_models, do: :text
  def output_type(model_name), do: raise("Unknown model: #{model_name}")

  def invoke(env, model_name, input, seed) when model_name in @dummy_t2i_models do
    invoke_dummy_t2i(env, model_name, input, seed)
  end

  def invoke(env, model_name, input, seed) when model_name in @dummy_i2t_models do
    invoke_dummy_i2t(env, model_name, input, seed)
  end

  def invoke(env, model_name, input, seed) when model_name in @real_t2i_models do
    invoke_real_t2i(env, model_name, input, seed)
  end

  def invoke(env, model_name, input, seed) when model_name in @real_i2t_models do
    invoke_real_i2t(env, model_name, input, seed)
  end

  defp invoke_real_t2i(env, model_name, prompt, seed) when is_binary(prompt) do
    with :ok <- PythonBridge.ensure_setup(env),
         :ok <- PythonBridge.ensure_model_loaded(env, model_name) do
      invoke_code = real_t2i_code(model_name)

      case Snex.pyeval(
             env,
             invoke_code,
             %{"prompt" => prompt, "seed" => seed},
             returning: "result",
             timeout: @t2i_timeout
           ) do
        {:ok, base64_data} -> {:ok, Base.decode64!(base64_data)}
        error -> error
      end
    end
  end

  defp invoke_real_i2t(env, model_name, image_binary, seed) when is_binary(image_binary) do
    image_b64 = Base.encode64(image_binary)

    with :ok <- PythonBridge.ensure_setup(env),
         :ok <- PythonBridge.ensure_model_loaded(env, model_name) do
      invoke_code = real_i2t_code(model_name)

      Snex.pyeval(
        env,
        invoke_code,
        %{"image_b64" => image_b64, "seed" => seed},
        returning: "result",
        timeout: @i2t_timeout
      )
    end
  end

  defp real_t2i_code("SDXLTurbo") do
    """
    _gen = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)
    _img = _models["SDXLTurbo"](
        prompt=prompt, height=IMAGE_SIZE, width=IMAGE_SIZE,
        num_inference_steps=4, guidance_scale=0.0, generator=_gen,
    ).images[0]
    _buf = io.BytesIO()
    _img.save(_buf, format="WEBP", lossless=True)
    result = base64.b64encode(_buf.getvalue()).decode("ascii")
    """
  end

  defp real_t2i_code("FluxDev") do
    """
    _gen = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)
    _img = _models["FluxDev"](
        prompt, height=IMAGE_SIZE, width=IMAGE_SIZE,
        guidance_scale=3.5, num_inference_steps=20, generator=_gen,
    ).images[0]
    _buf = io.BytesIO()
    _img.save(_buf, format="WEBP", lossless=True)
    result = base64.b64encode(_buf.getvalue()).decode("ascii")
    """
  end

  defp real_t2i_code("FluxSchnell") do
    """
    _gen = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)
    _img = _models["FluxSchnell"](
        prompt, height=IMAGE_SIZE, width=IMAGE_SIZE,
        guidance_scale=3.5, num_inference_steps=6, generator=_gen,
    ).images[0]
    _buf = io.BytesIO()
    _img.save(_buf, format="WEBP", lossless=True)
    result = base64.b64encode(_buf.getvalue()).decode("ascii")
    """
  end

  defp real_i2t_code("Moondream") do
    """
    _img_bytes = base64.b64decode(image_b64)
    _img = Image.open(io.BytesIO(_img_bytes))
    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
    _cap = _models["Moondream"].caption(_img, length="short")
    result = _cap["caption"].strip()
    """
  end

  defp real_i2t_code("BLIP2") do
    """
    _img_bytes = base64.b64decode(image_b64)
    _img = Image.open(io.BytesIO(_img_bytes))
    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
    _blip2 = _models["BLIP2"]
    _inputs = _blip2["processor"](images=_img, return_tensors="pt").to("cuda", torch.float16)
    _inputs = {k: v.to(torch.float16) if isinstance(v, torch.Tensor) else v for k, v in _inputs.items()}
    with torch.amp.autocast("cuda", dtype=torch.float16):
        _gen_ids = _blip2["model"].generate(
            **_inputs, max_length=50, do_sample=(seed != -1), num_beams=5, top_p=0.9,
        )
    result = _blip2["processor"].batch_decode(_gen_ids, skip_special_tokens=True)[0].strip()
    """
  end

  defp invoke_dummy_t2i(env, model_name, prompt, seed) when is_binary(prompt) do
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

  defp invoke_dummy_i2t(env, model_name, image_binary, seed) when is_binary(image_binary) do
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
