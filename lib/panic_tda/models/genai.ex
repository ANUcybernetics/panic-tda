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
  @real_t2i_models ~w(SD35Medium FluxSchnell ZImageTurbo Flux2Klein QwenImage)
  @real_i2t_models ~w(Moondream InstructBLIP Qwen25VL Gemma3n)
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
      invoke_code = real_t2i_code(model_name)

      case Snex.pyeval(
             env,
             invoke_code,
             %{"prompt" => prompt},
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
      invoke_code = real_i2t_code(model_name)

      Snex.pyeval(
        env,
        invoke_code,
        %{"image_b64" => image_b64},
        returning: "result",
        timeout: @i2t_timeout
      )
    end
  end

  defp real_t2i_code("SD35Medium") do
    """
    _img = _models["SD35Medium"](
        prompt=prompt, height=IMAGE_SIZE, width=IMAGE_SIZE,
        num_inference_steps=28, guidance_scale=5.0, generator=None,
    ).images[0]
    _buf = io.BytesIO()
    _img.save(_buf, format="WEBP", lossless=True)
    result = base64.b64encode(_buf.getvalue()).decode("ascii")
    """
  end

  defp real_t2i_code("FluxSchnell") do
    """
    _img = _models["FluxSchnell"](
        prompt, height=IMAGE_SIZE, width=IMAGE_SIZE,
        guidance_scale=3.5, num_inference_steps=6, generator=None,
    ).images[0]
    _buf = io.BytesIO()
    _img.save(_buf, format="WEBP", lossless=True)
    result = base64.b64encode(_buf.getvalue()).decode("ascii")
    """
  end

  defp real_t2i_code("ZImageTurbo") do
    """
    _img = _models["ZImageTurbo"](
        prompt=prompt, height=IMAGE_SIZE, width=IMAGE_SIZE,
        num_inference_steps=8, generator=None,
    ).images[0]
    _buf = io.BytesIO()
    _img.save(_buf, format="WEBP", lossless=True)
    result = base64.b64encode(_buf.getvalue()).decode("ascii")
    """
  end

  defp real_t2i_code("Flux2Klein") do
    """
    _img = _models["Flux2Klein"](
        prompt=prompt, height=IMAGE_SIZE, width=IMAGE_SIZE,
        num_inference_steps=4, guidance_scale=1.0, generator=None,
    ).images[0]
    _buf = io.BytesIO()
    _img.save(_buf, format="WEBP", lossless=True)
    result = base64.b64encode(_buf.getvalue()).decode("ascii")
    """
  end

  defp real_t2i_code("QwenImage") do
    """
    _img = _models["QwenImage"](
        prompt=prompt, height=IMAGE_SIZE, width=IMAGE_SIZE,
        num_inference_steps=50, true_cfg_scale=4.0, generator=None,
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
    _cap = _models["Moondream"].caption(_img, length="short")
    result = _cap["caption"].strip()
    """
  end

  defp real_i2t_code("InstructBLIP") do
    """
    _img_bytes = base64.b64decode(image_b64)
    _img = Image.open(io.BytesIO(_img_bytes))
    _iblip = _models["InstructBLIP"]
    _inputs = _iblip["processor"](
        images=_img, text="Describe this image.", return_tensors="pt"
    ).to("cuda", torch.float16)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        _gen_ids = _iblip["model"].generate(
            **_inputs, max_length=256, num_beams=5,
        )
    result = _iblip["processor"].batch_decode(_gen_ids, skip_special_tokens=True)[0].strip()
    """
  end

  defp real_i2t_code("Qwen25VL") do
    """
    from qwen_vl_utils import process_vision_info
    _img_bytes = base64.b64decode(image_b64)
    _img = Image.open(io.BytesIO(_img_bytes))
    _qwen_vl = _models["Qwen25VL"]
    _messages = [{"role": "user", "content": [
        {"type": "image", "image": _img},
        {"type": "text", "text": "Describe this image."},
    ]}]
    _text = _qwen_vl["processor"].apply_chat_template(_messages, tokenize=False, add_generation_prompt=True)
    _image_inputs, _video_inputs = process_vision_info(_messages)
    _inputs = _qwen_vl["processor"](
        text=[_text], images=_image_inputs, videos=_video_inputs,
        padding=True, return_tensors="pt",
    ).to(_qwen_vl["model"].device)
    with torch.no_grad():
        _gen_ids = _qwen_vl["model"].generate(**_inputs, max_new_tokens=128)
        _gen_ids = _gen_ids[:, _inputs["input_ids"].shape[1]:]
    result = _qwen_vl["processor"].batch_decode(_gen_ids, skip_special_tokens=True)[0].strip()
    """
  end

  defp real_i2t_code("Gemma3n") do
    """
    _img_bytes = base64.b64decode(image_b64)
    _img = Image.open(io.BytesIO(_img_bytes))
    _gemma3n = _models["Gemma3n"]
    _messages = [{"role": "user", "content": [
        {"type": "image", "image": _img},
        {"type": "text", "text": "Describe this image."},
    ]}]
    _inputs = _gemma3n["processor"].apply_chat_template(
        _messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
    ).to(_gemma3n["model"].device, dtype=torch.bfloat16)
    _input_len = _inputs["input_ids"].shape[1]
    with torch.no_grad():
        _gen_ids = _gemma3n["model"].generate(**_inputs, max_new_tokens=100, do_sample=False)
    result = _gemma3n["processor"].decode(_gen_ids[0][_input_len:], skip_special_tokens=True).strip()
    """
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

  defp invoke_batch_real_t2i(env, model_name, prompts) do
    with :ok <- PythonBridge.swap_model_to_gpu(env, model_name) do
      batch_code = real_t2i_batch_code(model_name)

      case Snex.pyeval(
             env,
             batch_code,
             %{"prompts" => prompts},
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
      batch_code = real_i2t_batch_code(model_name)

      Snex.pyeval(
        env,
        batch_code,
        %{"image_b64_list" => image_b64_list},
        returning: "result",
        timeout: @i2t_timeout * length(images)
      )
    end
  end

  defp real_t2i_batch_code("SD35Medium") do
    """
    _imgs = _models["SD35Medium"](
        prompt=prompts, height=IMAGE_SIZE, width=IMAGE_SIZE,
        num_inference_steps=28, guidance_scale=5.0, generator=None,
    ).images
    _results = []
    for _img in _imgs:
        _buf = io.BytesIO()
        _img.save(_buf, format="WEBP", lossless=True)
        _results.append(base64.b64encode(_buf.getvalue()).decode("ascii"))
    result = _results
    """
  end

  defp real_t2i_batch_code("FluxSchnell") do
    """
    _imgs = _models["FluxSchnell"](
        prompts, height=IMAGE_SIZE, width=IMAGE_SIZE,
        guidance_scale=3.5, num_inference_steps=6, generator=None,
    ).images
    _results = []
    for _img in _imgs:
        _buf = io.BytesIO()
        _img.save(_buf, format="WEBP", lossless=True)
        _results.append(base64.b64encode(_buf.getvalue()).decode("ascii"))
    result = _results
    """
  end

  defp real_t2i_batch_code("ZImageTurbo") do
    """
    _imgs = _models["ZImageTurbo"](
        prompt=prompts, height=IMAGE_SIZE, width=IMAGE_SIZE,
        num_inference_steps=8, generator=None,
    ).images
    _results = []
    for _img in _imgs:
        _buf = io.BytesIO()
        _img.save(_buf, format="WEBP", lossless=True)
        _results.append(base64.b64encode(_buf.getvalue()).decode("ascii"))
    result = _results
    """
  end

  defp real_t2i_batch_code("Flux2Klein") do
    """
    _imgs = _models["Flux2Klein"](
        prompt=prompts, height=IMAGE_SIZE, width=IMAGE_SIZE,
        num_inference_steps=4, guidance_scale=1.0, generator=None,
    ).images
    _results = []
    for _img in _imgs:
        _buf = io.BytesIO()
        _img.save(_buf, format="WEBP", lossless=True)
        _results.append(base64.b64encode(_buf.getvalue()).decode("ascii"))
    result = _results
    """
  end

  defp real_t2i_batch_code("QwenImage") do
    """
    _imgs = _models["QwenImage"](
        prompt=prompts, height=IMAGE_SIZE, width=IMAGE_SIZE,
        num_inference_steps=50, true_cfg_scale=4.0, generator=None,
    ).images
    _results = []
    for _img in _imgs:
        _buf = io.BytesIO()
        _img.save(_buf, format="WEBP", lossless=True)
        _results.append(base64.b64encode(_buf.getvalue()).decode("ascii"))
    result = _results
    """
  end

  defp real_i2t_batch_code("Moondream") do
    """
    _results = []
    for _img_b64 in image_b64_list:
        _img = Image.open(io.BytesIO(base64.b64decode(_img_b64)))
        _cap = _models["Moondream"].caption(_img, length="short")
        _results.append(_cap["caption"].strip())
    result = _results
    """
  end

  defp real_i2t_batch_code("InstructBLIP") do
    """
    _iblip = _models["InstructBLIP"]
    _images = [Image.open(io.BytesIO(base64.b64decode(b))) for b in image_b64_list]
    _inputs = _iblip["processor"](
        images=_images, text=["Describe this image."] * len(_images),
        return_tensors="pt", padding=True
    ).to("cuda", torch.float16)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        _gen_ids = _iblip["model"].generate(**_inputs, max_length=256, num_beams=5)
    result = [s.strip() for s in _iblip["processor"].batch_decode(
        _gen_ids, skip_special_tokens=True
    )]
    """
  end

  defp real_i2t_batch_code("Qwen25VL") do
    """
    from qwen_vl_utils import process_vision_info
    _qwen_vl = _models["Qwen25VL"]
    _all_texts = []
    _all_images = []
    for _img_b64 in image_b64_list:
        _img = Image.open(io.BytesIO(base64.b64decode(_img_b64)))
        _messages = [{"role": "user", "content": [
            {"type": "image", "image": _img},
            {"type": "text", "text": "Describe this image."},
        ]}]
        _all_texts.append(_qwen_vl["processor"].apply_chat_template(
            _messages, tokenize=False, add_generation_prompt=True
        ))
        _image_inputs, _ = process_vision_info(_messages)
        _all_images.extend(_image_inputs)
    _inputs = _qwen_vl["processor"](
        text=_all_texts, images=_all_images,
        padding=True, return_tensors="pt",
    ).to(_qwen_vl["model"].device)
    with torch.no_grad():
        _gen_ids = _qwen_vl["model"].generate(**_inputs, max_new_tokens=128)
        _gen_ids = _gen_ids[:, _inputs["input_ids"].shape[1]:]
    result = [s.strip() for s in _qwen_vl["processor"].batch_decode(
        _gen_ids, skip_special_tokens=True
    )]
    """
  end

  defp real_i2t_batch_code("Gemma3n") do
    """
    _gemma3n = _models["Gemma3n"]
    _all_messages = []
    for _img_b64 in image_b64_list:
        _img = Image.open(io.BytesIO(base64.b64decode(_img_b64)))
        _all_messages.append([{"role": "user", "content": [
            {"type": "image", "image": _img},
            {"type": "text", "text": "Describe this image."},
        ]}])
    _inputs = _gemma3n["processor"].apply_chat_template(
        _all_messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, padding=True,
    ).to(_gemma3n["model"].device, dtype=torch.bfloat16)
    _input_len = _inputs["input_ids"].shape[1]
    with torch.no_grad():
        _gen_ids = _gemma3n["model"].generate(**_inputs, max_new_tokens=100, do_sample=False)
    result = [s.strip() for s in _gemma3n["processor"].batch_decode(
        _gen_ids[:, _input_len:], skip_special_tokens=True
    )]
    """
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
