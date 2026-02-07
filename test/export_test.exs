defmodule PanicTda.ExportTest do
  use ExUnit.Case

  alias PanicTda.Engine

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  defp run_experiment(overrides \\ %{}) do
    defaults = %{
      networks: [["DummyT2I", "DummyI2T"]],
      prompts: ["test prompt"],
      embedding_models: ["DummyText"],
      max_length: 4
    }

    experiment = PanicTda.create_experiment!(Map.merge(defaults, overrides))
    {:ok, experiment} = Engine.perform_experiment(experiment.id)
    experiment
  end

  describe "video/3" do
    test "creates a valid MP4 video file" do
      experiment = run_experiment()
      output_path = Path.join(System.tmp_dir!(), "test_export_#{System.unique_integer([:positive])}.mp4")

      on_exit(fn -> File.rm(output_path) end)

      assert {:ok, ^output_path} = PanicTda.Export.video(experiment.id, output_path)
      assert File.exists?(output_path)

      stat = File.stat!(output_path)
      assert stat.size > 0

      bytes = File.read!(output_path)
      assert <<_::binary-size(4), "ftyp", _::binary>> = bytes
    end

    test "creates video with multiple prompts and networks" do
      experiment =
        run_experiment(%{
          networks: [["DummyT2I", "DummyI2T"], ["DummyT2I2", "DummyI2T2"]],
          prompts: ["prompt A", "prompt B"],
          max_length: 4
        })

      output_path = Path.join(System.tmp_dir!(), "test_export_multi_#{System.unique_integer([:positive])}.mp4")

      on_exit(fn -> File.rm(output_path) end)

      assert {:ok, ^output_path} = PanicTda.Export.video(experiment.id, output_path)
      assert File.exists?(output_path)

      stat = File.stat!(output_path)
      assert stat.size > 0
    end

    test "creates video with multiple runs" do
      experiment = run_experiment(%{num_runs: 2, max_length: 4})
      output_path = Path.join(System.tmp_dir!(), "test_export_runs_#{System.unique_integer([:positive])}.mp4")

      on_exit(fn -> File.rm(output_path) end)

      assert {:ok, ^output_path} = PanicTda.Export.video(experiment.id, output_path)
      assert File.exists?(output_path)
    end
  end

  describe "image/2" do
    test "exports an image invocation to a file" do
      experiment = run_experiment()
      experiment = Ash.load!(experiment, runs: [:invocations])
      run = hd(experiment.runs)

      image_inv = Enum.find(run.invocations, &(&1.type == :image))

      output_path = Path.join(System.tmp_dir!(), "test_image_#{System.unique_integer([:positive])}.avif")
      on_exit(fn -> File.rm(output_path) end)

      assert {:ok, ^output_path} = PanicTda.Export.image(image_inv.id, output_path)
      assert File.exists?(output_path)

      stat = File.stat!(output_path)
      assert stat.size > 0
    end

    test "exports image with format conversion" do
      experiment = run_experiment()
      experiment = Ash.load!(experiment, runs: [:invocations])
      run = hd(experiment.runs)

      image_inv = Enum.find(run.invocations, &(&1.type == :image))

      output_path = Path.join(System.tmp_dir!(), "test_image_#{System.unique_integer([:positive])}.png")
      on_exit(fn -> File.rm(output_path) end)

      assert {:ok, ^output_path} = PanicTda.Export.image(image_inv.id, output_path)
      assert File.exists?(output_path)
    end

    test "returns error for non-existent invocation" do
      fake_id = Ash.UUID.generate()
      output_path = Path.join(System.tmp_dir!(), "test_image_nope.png")

      assert {:error, :not_found} = PanicTda.Export.image(fake_id, output_path)
    end

    test "returns error for text-only invocation" do
      experiment = run_experiment()
      experiment = Ash.load!(experiment, runs: [:invocations])
      run = hd(experiment.runs)

      text_inv = Enum.find(run.invocations, &(&1.type == :text))

      output_path = Path.join(System.tmp_dir!(), "test_image_text.png")

      assert {:error, :no_image} = PanicTda.Export.image(text_inv.id, output_path)
    end
  end
end
