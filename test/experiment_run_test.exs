defmodule ExperimentRunTest do
  use ExUnit.Case

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    shell = Mix.shell()
    Mix.shell(Mix.Shell.Process)
    on_exit(fn -> Mix.shell(shell) end)
    :ok
  end

  # The pattern bin/long-run greps the launch output with, so the test breaks if
  # the script and the task stop agreeing on what an id looks like.
  defp long_run_id_pattern do
    [_, class] =
      Regex.run(
        ~r/'Created experiment \\K(\[[^\]]+\]\+)'/,
        File.read!("bin/long-run")
      )

    Regex.compile!("Created experiment (#{class})")
  end

  defp launch(config_path) do
    Mix.Tasks.Experiment.Run.run([config_path])
    output = collect_output()
    [_, id] = Regex.run(long_run_id_pattern(), output)
    id
  end

  defp collect_output(acc \\ []) do
    receive do
      {:mix_shell, :info, [line]} -> collect_output([line | acc])
    after
      0 -> acc |> Enum.reverse() |> Enum.join("\n")
    end
  end

  test "two experiments launched within a minute each resume by the id recorded" do
    config =
      Path.join(
        System.tmp_dir!(),
        "experiment_run_test_#{System.unique_integer([:positive])}.json"
      )

    File.write!(
      config,
      Jason.encode!(%{
        networks: [["DummyT2I", "DummyI2T"]],
        prompts: ["test prompt"],
        embedding_models: ["DummyText"],
        max_length: 2
      })
    )

    on_exit(fn -> File.rm(config) end)

    first = launch(config)
    second = launch(config)

    assert first != second
    # what bin/long-run used to record, which both launches share
    assert String.slice(first, 0, 8) == String.slice(second, 0, 8)
    assert {:error, _} = PanicTda.find_experiment(String.slice(first, 0, 8))

    for id <- [first, second] do
      Mix.Tasks.Experiment.Resume.run([id, "--force"])
      assert {:ok, %{id: ^id, completed_at: %DateTime{}}} = PanicTda.find_experiment(id)
    end
  end
end
