defmodule Mix.Tasks.Embeddings.Recompute do
  @shortdoc "Recompute stored embedding vectors in place"

  @moduledoc """
  Recomputes stored embedding vectors in place. See
  `PanicTda.Embeddings.Recompute` for what it does and why.

      $ mix embeddings.recompute
      $ mix embeddings.recompute --model Qwen3Embed --experiment 01a060b4
      $ mix embeddings.recompute --dry-run
      $ mix embeddings.recompute --after 019f3645-0000-7000-8000-000000000000

  The run prints the last id of each page. `--after ID` resumes from there,
  which is how you pick a crashed run back up without re-embedding everything
  that already landed.
  """

  use Mix.Task

  alias PanicTda.Embeddings.Recompute

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          model: :keep,
          experiment: :keep,
          batch: :integer,
          dry_run: :boolean,
          after: :string
        ]
      )

    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
    Mix.Task.run("app.start")

    {:ok, interpreter} = PanicTda.Models.PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)
    started = System.monotonic_time(:millisecond)

    recompute_opts =
      [
        experiments: Keyword.get_values(opts, :experiment),
        dry_run: Keyword.get(opts, :dry_run, false),
        on_progress: &report(&1, started)
      ]
      |> put_if(:models, Keyword.get_values(opts, :model))
      |> put_if(:batch, Keyword.get(opts, :batch))
      |> put_if(:after, Keyword.get(opts, :after))

    try do
      Recompute.run(env, recompute_opts)
    after
      PanicTda.Models.PythonInterpreter.stop_or_kill(interpreter)
    end
  end

  defp put_if(opts, _key, nil), do: opts
  defp put_if(opts, _key, []), do: opts
  defp put_if(opts, key, value), do: Keyword.put(opts, key, value)

  defp report(%{done: 0, total: 0, model: model}, _started),
    do: Mix.shell().info("#{model}: nothing to do")

  defp report(%{done: 0, total: total, model: model, dry_run: dry_run?}, _started),
    do:
      Mix.shell().info("#{model}: #{total} embeddings#{if dry_run?, do: " (dry run)", else: ""}")

  defp report(%{done: done, total: total, after: after_id}, started) do
    elapsed = System.monotonic_time(:millisecond) - started
    rate = done * 1000 / max(elapsed, 1)

    Mix.shell().info(
      "  #{done}/#{total}  #{Float.round(rate, 1)}/s  " <>
        "eta #{div(round((total - done) / max(rate, 0.001)), 60)}m  after #{after_id}"
    )
  end
end
