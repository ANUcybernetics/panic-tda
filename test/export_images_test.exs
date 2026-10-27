defmodule ExportImagesTest do
  use ExUnit.Case

  alias PanicTda.Engine

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    shell = Mix.shell()
    Mix.shell(Mix.Shell.Process)
    on_exit(fn -> Mix.shell(shell) end)

    # two runs of six images each, at sequence numbers 0, 2, ..., 10
    experiment =
      PanicTda.create_experiment!(%{
        networks: [["DummyT2I", "DummyI2T"]],
        prompts: ["test prompt"],
        embedding_models: ["DummyText"],
        max_length: 12,
        num_runs: 2
      })

    {:ok, experiment} = Engine.perform_experiment(experiment.id)

    dir =
      Path.join(System.tmp_dir!(), "export_images_test_#{System.unique_integer([:positive])}")

    on_exit(fn -> File.rm_rf!(dir) end)
    %{experiment: experiment, dir: dir}
  end

  defp written(dir) do
    dir
    |> Path.join("**/*.avif")
    |> Path.wildcard()
    |> Enum.map(&Path.relative_to(&1, dir))
    |> Enum.sort()
  end

  test "--every N writes every Nth image of each run, starting from its first", %{
    experiment: experiment,
    dir: dir
  } do
    Mix.Tasks.Experiment.ExportImages.run([experiment.id, "--output", dir, "--every", "4"])

    run_dirs = for n <- [0, 1], do: "DummyT2I_DummyI2T/test_prompt/run_#{n}"

    assert written(dir) ==
             Enum.sort(for d <- run_dirs, seq <- ["000", "008"], do: "#{d}/seq_#{seq}.avif")

    # the bytes are the stored image's, with metadata added after
    {out, 0} =
      System.cmd("exiftool", ["-s3", "-XMP-dc:Title", Path.join(dir, hd(written(dir)))])

    assert out =~ "step 0"
  end

  test "--limit caps what --every keeps", %{experiment: experiment, dir: dir} do
    Mix.Tasks.Experiment.ExportImages.run([
      experiment.id,
      "--output",
      dir,
      "--every",
      "2",
      "--limit",
      "4"
    ])

    # the first run's 1st, 3rd and 5th images, then the second run's 1st
    assert written(dir) == [
             "DummyT2I_DummyI2T/test_prompt/run_0/seq_000.avif",
             "DummyT2I_DummyI2T/test_prompt/run_0/seq_004.avif",
             "DummyT2I_DummyI2T/test_prompt/run_0/seq_008.avif",
             "DummyT2I_DummyI2T/test_prompt/run_1/seq_000.avif"
           ]
  end

  test "without --every every image is written", %{experiment: experiment, dir: dir} do
    Mix.Tasks.Experiment.ExportImages.run([experiment.id, "--output", dir])
    assert length(written(dir)) == 12
  end
end
