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

  describe "compute_layout/5" do
    test "canvas always equals target resolution" do
      for {n_net, n_prom, n_run} <- [{20, 4, 4}, {2, 2, 1}, {1, 1, 1}, {20, 1, 1}, {1, 1, 16}],
          {target_w, target_h} <- [{1920, 1080}, {3840, 2160}] do
        layout = PanicTda.Export.compute_layout(n_net, n_prom, n_run, target_w, target_h)
        assert layout.canvas_w == target_w, "canvas_w for #{n_net}×#{n_prom}×#{n_run} @ #{target_w}×#{target_h}"
        assert layout.canvas_h == target_h, "canvas_h for #{n_net}×#{n_prom}×#{n_run} @ #{target_w}×#{target_h}"
      end
    end

    test "thumb_size is within bounds" do
      for {n_net, n_prom, n_run} <- [{20, 4, 4}, {2, 2, 1}, {1, 1, 1}, {20, 1, 1}, {1, 1, 16}],
          {target_w, target_h} <- [{1920, 1080}, {3840, 2160}] do
        layout = PanicTda.Export.compute_layout(n_net, n_prom, n_run, target_w, target_h)
        assert layout.thumb_size >= 16, "thumb_size too small for #{n_net}×#{n_prom}×#{n_run}"
        assert layout.thumb_size <= 512, "thumb_size too large for #{n_net}×#{n_prom}×#{n_run}"
      end
    end

    test "20×4×4 wraps into multiple bands" do
      layout = PanicTda.Export.compute_layout(20, 4, 4, 1920, 1080)
      assert layout.wrap > 1
      assert layout.eff_cols == ceil(20 / layout.wrap)
      assert layout.sub_cols == 2
      assert layout.sub_rows == 2
    end

    test "small configs don't wrap" do
      for {n_net, n_prom, n_run} <- [{2, 2, 1}, {1, 1, 1}] do
        layout = PanicTda.Export.compute_layout(n_net, n_prom, n_run, 1920, 1080)
        assert layout.wrap == 1, "unexpected wrap for #{n_net}×#{n_prom}×#{n_run}"
      end
    end

    test "content fits within canvas" do
      for {n_net, n_prom, n_run} <- [{20, 4, 4}, {2, 2, 1}, {1, 1, 1}, {20, 1, 1}, {1, 1, 16}],
          {target_w, target_h} <- [{1920, 1080}, {3840, 2160}] do
        layout = PanicTda.Export.compute_layout(n_net, n_prom, n_run, target_w, target_h)

        grid_w = layout.eff_cols * layout.cell_w + max(layout.eff_cols - 1, 0) * layout.gap_px
        band_h = n_prom * layout.cell_h + max(n_prom - 1, 0) * layout.gap_px
        total_h =
          layout.n_bands * (layout.label_h + band_h) +
            max(layout.n_bands - 1, 0) * layout.band_gap_px

        content_w = layout.label_w + grid_w + 2 * layout.offset_x
        content_h = total_h + 2 * layout.offset_y

        assert content_w <= target_w + 1, "content too wide for #{n_net}×#{n_prom}×#{n_run} @ #{target_w}×#{target_h}"
        assert content_h <= target_h + 1, "content too tall for #{n_net}×#{n_prom}×#{n_run} @ #{target_w}×#{target_h}"
      end
    end

    test "4 runs produce 2×2 sub-grid" do
      layout = PanicTda.Export.compute_layout(1, 1, 4, 1920, 1080)
      assert layout.sub_cols == 2
      assert layout.sub_rows == 2
    end

    test "16 runs produce 4×4 sub-grid" do
      layout = PanicTda.Export.compute_layout(1, 1, 16, 1920, 1080)
      assert layout.sub_cols == 4
      assert layout.sub_rows == 4
    end
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
