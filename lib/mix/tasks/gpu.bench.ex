defmodule Mix.Tasks.Gpu.Bench do
  @shortdoc "Benchmark per-item T2I wall-clock and batched-vs-serial pixel parity"

  @moduledoc """
  Measures inference performance for text-to-image models on this GPU, for the
  TASK-74 optimisation loop. For each model it generates a small seed-matched
  reference set serially, then re-generates it at each requested batch size,
  reporting per-item wall-clock and the mean/max absolute pixel delta between the
  batched output and the seed-matched serial reference (the deterministic
  quality-parity metric).

      $ mix gpu.bench Flux2Dev GLMImage --batch-sizes 1,2,4,8 --n 8

  Options:

    - `--batch-sizes` comma-separated batch sizes to probe (default `1,2,4,8`)
    - `--n` number of prompts/seeds per model (default 8)
    - `--seed` base seed; item i uses `seed + i` (default 424242)

  With no model arguments, benchmarks the full panel T2I set
  (SD35Medium, ZImageTurbo, Flux2Klein, GLMImage, Flux2Dev).
  """

  use Mix.Task

  @default_models ["SD35Medium", "ZImageTurbo", "Flux2Klein", "GLMImage", "Flux2Dev"]
  @default_batch_sizes [1, 2, 4, 8]
  @default_n 8
  @default_seed 424_242

  # The benchmark generates n images serially and then n more at every batch
  # size, so its wall-clock scales with both. A flat ceiling silently blew up
  # an hour into a `--n 16 --batch-sizes 4,8,16` run on Flux2Dev; budget per
  # image instead, generously, since the slowest model is ~90 s/image serial.
  @bench_ms_per_image 150_000
  @load_timeout 600_000

  @prompts [
    "a red apple on a wooden table",
    "a cat sitting on a windowsill",
    "a bicycle leaning against a brick wall",
    "a glass bottle beside a candle",
    "a picnic basket under a tree beside a river",
    "a bustling Tokyo street at night",
    "a storm approaching a fishing village",
    "a library with impossible architecture",
    "a machine dreaming of a forest",
    "a city slowly turning into a forest",
    "a market stall displaying fruit and flowers",
    "a train station with travellers carrying luggage"
  ]

  @impl Mix.Task
  def run(args) do
    {opts, models, _} =
      OptionParser.parse(args,
        strict: [batch_sizes: :string, n: :integer, seed: :integer, dump: :string]
      )

    models = if models == [], do: @default_models, else: models
    n = Keyword.get(opts, :n, @default_n)
    seed = Keyword.get(opts, :seed, @default_seed)
    dump = Keyword.get(opts, :dump, "")

    batch_sizes =
      case Keyword.get(opts, :batch_sizes) do
        nil -> @default_batch_sizes
        s -> s |> String.split(",", trim: true) |> Enum.map(&String.to_integer/1)
      end

    prompts = @prompts |> Stream.cycle() |> Enum.take(n)
    seeds = for i <- 0..(n - 1), do: seed + i

    Mix.Task.run("app.start")

    {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    :ok = PanicTda.Models.PythonBridge.ensure_setup(env)
    :ok = PanicTda.Models.PythonBridge.unload_all_models(env)

    info("Benchmark: n=#{n}, seeds #{seed}..#{seed + n - 1}, batch sizes #{inspect(batch_sizes)}")

    results =
      Enum.map(models, fn model ->
        {model, bench_model(env, model, prompts, seeds, batch_sizes, dump)}
      end)

    print_summary(results)

    GenServer.stop(interpreter)
  end

  defp bench_model(env, model, prompts, seeds, batch_sizes, dump) do
    info("\n== #{model} ==")
    :ok = PanicTda.Models.PythonBridge.unload_all_models(env)
    :ok = PanicTda.Models.PythonBridge.ensure_model_loaded(env, model)

    {:ok, _} =
      Snex.pyeval(env, "panic_models.swap_to_gpu(name)\nreturn True", %{"name" => model},
        timeout: @load_timeout
      )

    {:ok, result} =
      Snex.pyeval(
        env,
        "return panic_models.benchmark_t2i(name, prompts, seeds, batch_sizes, dump_dir)",
        %{
          "name" => model,
          "prompts" => prompts,
          "seeds" => seeds,
          "batch_sizes" => batch_sizes,
          "dump_dir" => dump
        },
        timeout: @bench_ms_per_image * length(seeds) * (length(batch_sizes) + 1)
      )

    single = result["single_per_item_s"]
    info("  batch=1   per-item #{fmt(single)}s  (serial reference)")

    for {bs, m} <- Enum.sort_by(result["batches"], fn {k, _} -> to_int(k) end) do
      case m["status"] do
        "ok" ->
          per_item = m["per_item_s"]
          speedup = single / per_item

          info(
            "  batch=#{to_int(bs)}   per-item #{fmt(per_item)}s  " <>
              "(#{fmt(speedup)}x)  parity mean/max #{fmt(m["parity_mean_abs_delta"])}" <>
              "/#{fmt(m["parity_max_abs_delta"])}"
          )

        other ->
          info("  batch=#{to_int(bs)}   #{other}")
      end
    end

    result
  end

  defp print_summary(results) do
    info("\n== Summary: best per-item time & speedup ==")

    for {model, result} <- results do
      single = result["single_per_item_s"]

      best =
        result["batches"]
        |> Enum.filter(fn {_bs, m} -> m["status"] == "ok" end)
        |> Enum.min_by(fn {_bs, m} -> m["per_item_s"] end, fn -> nil end)

      case best do
        nil ->
          info("  #{String.pad_trailing(model, 14)} #{fmt(single)}s serial; no batch win")

        {bs, m} ->
          info(
            "  #{String.pad_trailing(model, 14)} #{fmt(single)}s -> #{fmt(m["per_item_s"])}s " <>
              "@batch=#{to_int(bs)} (#{fmt(single / m["per_item_s"])}x, parity #{fmt(m["parity_mean_abs_delta"])})"
          )
      end
    end
  end

  defp to_int(k) when is_binary(k), do: String.to_integer(k)
  defp to_int(k), do: k

  defp fmt(x) when is_float(x), do: :erlang.float_to_binary(x, decimals: 2)
  defp fmt(x), do: to_string(x)

  defp info(msg), do: Mix.shell().info(msg)
end
