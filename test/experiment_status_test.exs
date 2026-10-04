defmodule ExperimentStatusTest do
  use ExUnit.Case

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    shell = Mix.shell()
    Mix.shell(Mix.Shell.Process)
    on_exit(fn -> Mix.shell(shell) end)
    :ok
  end

  defp create_run do
    experiment =
      PanicTda.create_experiment!(%{
        networks: [["DummyT2I", "DummyI2T"]],
        prompts: ["test prompt"],
        embedding_models: ["DummyText"],
        max_length: 20
      })

    run =
      PanicTda.create_run!(%{
        network: ["DummyT2I", "DummyI2T"],
        run_number: 0,
        max_length: 20,
        initial_prompt: "test prompt",
        experiment_id: experiment.id
      })

    {experiment, run}
  end

  defp create_captions(run, captions) do
    captions
    |> Enum.with_index()
    |> Enum.each(fn {caption, i} ->
      PanicTda.create_invocation!(%{
        model: "DummyI2T",
        type: :text,
        sequence_number: 2 * i + 1,
        output_text: caption,
        started_at: DateTime.utc_now(),
        completed_at: DateTime.utc_now(),
        run_id: run.id
      })
    end)
  end

  defp status(experiment) do
    Mix.Tasks.Experiment.Status.run([experiment.id])
    assert_received {:mix_shell, :info, [output]}
    output
  end

  describe "caption truncation" do
    test "a caption that ends a sentence is not counted, whatever closes after it" do
      {experiment, run} = create_run()

      create_captions(run, [
        "A red apple on a wooden table.",
        "**In summary, a red apple sits on a wooden table.**",
        "The sign above the door reads \"Open.\"",
        "The image shows two candles. 背景是绿色的。",
        "A lit candle beside a bottle.\n"
      ])

      assert status(experiment) =~ "DummyI2T: 0/5 (0.0%)"
    end

    test "a caption cut off mid-sentence is counted, even when it stops on a closing quote" do
      {experiment, run} = create_run()

      create_captions(run, [
        "A red apple on a wooden table.",
        "The spines read \"Reading for Life,\" \"Reading for Life,\" \"Reading",
        "The spines read \"Parenting,\" \"Parenting,\""
      ])

      assert status(experiment) =~ "DummyI2T: 2/3 (66.7%)"
    end
  end
end
