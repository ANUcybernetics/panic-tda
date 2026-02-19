defmodule Mix.Tasks.Gpu.MaxBatch do
  @shortdoc "Probe maximum batch sizes for truly-batched GPU models"

  @moduledoc """
  Empirically determines the maximum batch size each truly-batched model can
  handle on this GPU (RTX 6000 Ada, 48 GB VRAM) before running out of memory.

      $ mix gpu.max_batch

  Tests the following models in priority order (largest first):

    - **T2I**: Flux2Klein, SD35Medium, ZImageTurbo
    - **I2T**: Pixtral, LLaMA32Vision

  For I2T models, test images are first generated using SD35Medium (in batches
  of 3), then SD35Medium is unloaded before probing begins.

  Batch sizes tested: 1, 6, 12, 18, 24, 30, 36, 42, 48.
  """

  use Mix.Task

  @probe_sizes [1, 6, 12, 18, 24, 30, 36, 42, 48]
  @num_test_inputs 48
  @image_gen_batch_size 3

  @t2i_models ["Flux2Klein", "SD35Medium", "ZImageTurbo"]
  @i2t_models ["Pixtral", "LLaMA32Vision"]

  @probe_timeout 600_000
  @load_timeout 600_000

  @impl Mix.Task
  def run(_args) do
    Mix.Task.run("app.start")

    {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    :ok = PanicTda.Models.PythonBridge.ensure_setup(env)
    :ok = PanicTda.Models.PythonBridge.unload_all_models(env)

    prompts = for i <- 1..@num_test_inputs, do: "a test image number #{i}"

    info("Generating #{@num_test_inputs} test images with SD35Medium...")
    test_images = generate_test_images(env, prompts)
    info("Generated #{length(test_images)} test images.")

    unload_all(env)

    results = %{}

    results =
      Enum.reduce(@t2i_models, results, fn model, acc ->
        result = probe_model(env, model, :t2i, prompts)
        Map.put(acc, model, result)
      end)

    results =
      Enum.reduce(@i2t_models, results, fn model, acc ->
        result = probe_model(env, model, :i2t, test_images)
        Map.put(acc, model, result)
      end)

    print_summary(results)

    GenServer.stop(interpreter)
  end

  defp generate_test_images(env, prompts) do
    :ok = load_and_swap(env, "SD35Medium")

    prompts
    |> Enum.chunk_every(@image_gen_batch_size)
    |> Enum.flat_map(fn batch ->
      {:ok, images} =
        Snex.pyeval(
          env,
          "result = panic_models.invoke_t2i_batch(name, prompts)",
          %{"name" => "SD35Medium", "prompts" => batch},
          returning: "result",
          timeout: @probe_timeout
        )

      images
    end)
  end

  defp probe_model(env, model, _type, test_inputs) do
    info("\nProbing #{model}...")
    unload_all(env)
    :ok = load_and_swap(env, model)

    inputs = Enum.take(test_inputs, @num_test_inputs)

    {:ok, results} =
      Snex.pyeval(
        env,
        "result = panic_models.probe_max_batch(name, test_inputs, sizes)",
        %{"name" => model, "test_inputs" => inputs, "sizes" => @probe_sizes},
        returning: "result",
        timeout: @probe_timeout
      )

    results = normalise_keys(results)

    for {size, status} <- Enum.sort_by(results, &elem(&1, 0)) do
      info("  batch=#{size} => #{status}")
    end

    max_ok =
      results
      |> Enum.filter(fn {_k, v} -> v == "ok" end)
      |> Enum.map(&elem(&1, 0))
      |> Enum.max(fn -> 0 end)

    info("  max successful batch size: #{max_ok}")
    results
  end

  defp load_and_swap(env, model) do
    :ok = PanicTda.Models.PythonBridge.ensure_model_loaded(env, model)

    {:ok, _} =
      Snex.pyeval(
        env,
        "panic_models.swap_to_gpu(name)",
        %{"name" => model},
        returning: "True",
        timeout: @load_timeout
      )

    :ok
  end

  defp unload_all(env) do
    :ok = PanicTda.Models.PythonBridge.unload_all_models(env)
  end

  defp normalise_keys(map) do
    Map.new(map, fn {k, v} ->
      key = if is_binary(k), do: String.to_integer(k), else: k
      {key, v}
    end)
  end

  defp print_summary(results) do
    info("\n== Max batch size summary ==")
    info(String.pad_trailing("Model", 20) <> "Max batch")
    info(String.duplicate("-", 30))

    all_models = @t2i_models ++ @i2t_models

    for model <- all_models do
      case Map.get(results, model) do
        nil ->
          info(String.pad_trailing(model, 20) <> "not tested")

        model_results ->
          max_ok =
            model_results
            |> Enum.filter(fn {_k, v} -> v == "ok" end)
            |> Enum.map(&elem(&1, 0))
            |> Enum.max(fn -> 0 end)

          suffix =
            if max_ok == List.last(@probe_sizes), do: "+", else: ""

          info(String.pad_trailing(model, 20) <> "#{max_ok}#{suffix}")
      end
    end
  end

  defp info(msg), do: Mix.shell().info(msg)
end
