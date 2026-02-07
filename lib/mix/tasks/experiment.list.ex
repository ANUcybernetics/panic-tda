defmodule Mix.Tasks.Experiment.List do
  @shortdoc "List all PANIC-TDA experiments"

  @moduledoc """
  Lists all experiments with their ID, status, and creation timestamp.

      $ mix experiment.list
  """

  use Mix.Task

  @impl Mix.Task
  def run(_args) do
    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
    Mix.Task.run("app.start")

    experiments = PanicTda.list_experiments!(query: [sort: [inserted_at: :desc]])

    if experiments == [] do
      Mix.shell().info("No experiments found.")
    else
      header = format_row("ID", "STATUS", "CREATED")
      separator = String.duplicate("-", String.length(header))

      Mix.shell().info(header)
      Mix.shell().info(separator)

      Enum.each(experiments, fn exp ->
        Mix.shell().info(format_row(short_id(exp.id), status(exp), format_time(exp.inserted_at)))
      end)
    end
  end

  defp format_row(id, status, created) do
    String.pad_trailing(id, 12) <> String.pad_trailing(status, 14) <> created
  end

  defp status(%{completed_at: %DateTime{}}), do: "completed"
  defp status(%{started_at: %DateTime{}}), do: "running"
  defp status(_), do: "pending"

  defp format_time(dt), do: Calendar.strftime(dt, "%Y-%m-%d %H:%M:%S")

  defp short_id(id), do: String.slice(id, 0, 8)
end
