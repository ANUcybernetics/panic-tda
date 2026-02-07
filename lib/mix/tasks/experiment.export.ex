defmodule Mix.Tasks.Experiment.Export do
  @shortdoc "Export a mosaic video or individual image from an experiment"

  @moduledoc """
  Exports experiment data as video or individual images.

      $ mix experiment.export <experiment-id> [--output path.mp4] [--fps 10] [--resolution hd]
      $ mix experiment.export --image <invocation-id> [--output image.png]

  ## Video export

  Creates a mosaic video with a prompt×network grid layout. Each cell shows
  the run's images side by side, animated across sequence numbers.

  Options:

    - `--output` - output file path (default: `export.mp4`)
    - `--fps` - frames per second (default: 10)
    - `--resolution` - `hd` (1920×1080) or `4k` (3840×2160, default: `hd`)
    - `--quality` - CRF value for encoding quality (default: 22)

  ## Image export

  Exports a single invocation's image to a file.

    - `--image` - invocation ID to export
    - `--output` - output file path (default: `export.png`)
  """

  use Mix.Task

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
    Mix.Task.run("app.start")

    {opts, positional, _} =
      OptionParser.parse(args,
        strict: [
          output: :string,
          fps: :integer,
          resolution: :string,
          quality: :integer,
          image: :string
        ]
      )

    cond do
      opts[:image] ->
        export_image(opts)

      length(positional) == 1 ->
        export_video(hd(positional), opts)

      true ->
        Mix.raise("Usage: mix experiment.export <experiment-id> [options]")
    end
  end

  defp export_video(id_prefix, opts) do
    experiment = find_experiment(id_prefix)
    output = Keyword.get(opts, :output, "export.mp4")
    fps = Keyword.get(opts, :fps, 10)
    quality = Keyword.get(opts, :quality, 22)

    resolution =
      case Keyword.get(opts, :resolution, "hd") do
        "4k" -> :"4k"
        _ -> :hd
      end

    Mix.shell().info("Exporting video for experiment #{short_id(experiment.id)}...")

    {:ok, path} = PanicTda.Export.video(experiment.id, output, fps: fps, resolution: resolution, quality: quality)
    Mix.shell().info("Video exported to #{path}")
  end

  defp export_image(opts) do
    invocation_id = opts[:image]
    output = Keyword.get(opts, :output, "export.png")

    case PanicTda.Export.image(invocation_id, output) do
      {:ok, path} ->
        Mix.shell().info("Image exported to #{path}")

      {:error, reason} ->
        Mix.raise("Image export failed: #{inspect(reason)}")
    end
  end

  defp find_experiment(id_prefix) do
    experiments = PanicTda.list_experiments!()

    Enum.find(experiments, fn e -> String.starts_with?(e.id, id_prefix) end) ||
      Mix.raise("No experiment found matching '#{id_prefix}'")
  end

  defp short_id(id), do: String.slice(id, 0, 8)
end
