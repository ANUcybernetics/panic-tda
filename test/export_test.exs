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

  describe "compute_layout/6" do
    test "supported counts produce valid layouts at HD and 4K" do
      combos = [
        {1, 1, 1},
        {2, 2, 4},
        {3, 4, 3},
        {3, 4, 12},
        {1, 1, 9},
        {2, 1, 6},
        {1, 3, 2}
      ]

      for {n_net, n_prom, n_run} <- combos,
          resolution <- [:hd, :"4k"] do
        {frame_w, frame_h} = if resolution == :hd, do: {1920, 1080}, else: {3840, 2160}

        layout =
          PanicTda.Export.compute_layout(n_net, n_prom, n_run, frame_w, frame_h, resolution)

        assert layout.img_size > 0,
               "img_size for #{n_net}×#{n_prom}×#{n_run} @ #{resolution}"

        assert layout.outer_rows * layout.outer_cols == n_net * n_prom
        assert layout.sub_rows * layout.sub_cols >= n_run
      end
    end

    test "unsupported num_runs raises ArgumentError" do
      for bad_n_run <- [5, 7, 10] do
        assert_raise ArgumentError, ~r/unsupported num_runs/, fn ->
          PanicTda.Export.compute_layout(1, 1, bad_n_run, 1920, 1080, :hd)
        end
      end
    end

    test "unsupported outer count raises ArgumentError" do
      for {n_net, n_prom} <- [{5, 1}, {1, 5}, {7, 1}] do
        assert_raise ArgumentError, ~r/num_networks × num_prompts/, fn ->
          PanicTda.Export.compute_layout(n_net, n_prom, 1, 1920, 1080, :hd)
        end
      end
    end

    test "canvas dimensions always match target resolution" do
      for resolution <- [:hd, :"4k"] do
        {frame_w, frame_h} = if resolution == :hd, do: {1920, 1080}, else: {3840, 2160}

        layout = PanicTda.Export.compute_layout(2, 2, 4, frame_w, frame_h, resolution)
        assert layout.canvas_w == frame_w
        assert layout.canvas_h == frame_h
      end
    end

    test "content fits within canvas" do
      combos = [{1, 1, 1}, {2, 2, 4}, {3, 4, 3}, {3, 4, 12}]

      for {n_net, n_prom, n_run} <- combos,
          resolution <- [:hd, :"4k"] do
        {frame_w, frame_h} = if resolution == :hd, do: {1920, 1080}, else: {3840, 2160}
        layout = PanicTda.Export.compute_layout(n_net, n_prom, n_run, frame_w, frame_h, resolution)

        content_w =
          layout.outer_cols * layout.sub_cols * layout.img_size +
            (layout.outer_cols - 1) * layout.outer_gutter +
            layout.outer_cols * (layout.sub_cols - 1) * layout.gutter

        content_h =
          layout.outer_rows * layout.sub_rows * layout.img_size +
            (layout.outer_rows - 1) * layout.outer_gutter +
            layout.outer_rows * (layout.label_h + layout.label_pad) +
            layout.outer_rows * (layout.sub_rows - 1) * layout.gutter

        assert content_w <= frame_w,
               "content too wide for #{n_net}×#{n_prom}×#{n_run} @ #{resolution}: #{content_w} > #{frame_w}"

        assert content_h <= frame_h,
               "content too tall for #{n_net}×#{n_prom}×#{n_run} @ #{resolution}: #{content_h} > #{frame_h}"
      end
    end

    test "minimal dead margin: at least one dimension has <= 5% dead space" do
      combos = [{1, 1, 1}, {2, 2, 4}, {3, 4, 3}, {3, 4, 12}, {1, 1, 9}]

      for {n_net, n_prom, n_run} <- combos,
          resolution <- [:hd, :"4k"] do
        {frame_w, frame_h} = if resolution == :hd, do: {1920, 1080}, else: {3840, 2160}
        layout = PanicTda.Export.compute_layout(n_net, n_prom, n_run, frame_w, frame_h, resolution)

        content_w =
          layout.outer_cols * layout.sub_cols * layout.img_size +
            (layout.outer_cols - 1) * layout.outer_gutter +
            layout.outer_cols * (layout.sub_cols - 1) * layout.gutter

        content_h =
          layout.outer_rows * layout.sub_rows * layout.img_size +
            (layout.outer_rows - 1) * layout.outer_gutter +
            layout.outer_rows * (layout.label_h + layout.label_pad) +
            layout.outer_rows * (layout.sub_rows - 1) * layout.gutter

        margin_w_pct = (frame_w - content_w) / frame_w
        margin_h_pct = (frame_h - content_h) / frame_h

        assert margin_w_pct <= 0.05 or margin_h_pct <= 0.05,
               "too much dead space for #{n_net}×#{n_prom}×#{n_run} @ #{resolution}: " <>
                 "w=#{Float.round(margin_w_pct * 100, 1)}%, h=#{Float.round(margin_h_pct * 100, 1)}%"
      end
    end

    test "both orientations are tried for asymmetric cases" do
      layout_34 = PanicTda.Export.compute_layout(3, 4, 3, 1920, 1080, :hd)
      layout_43 = PanicTda.Export.compute_layout(4, 3, 3, 1920, 1080, :hd)

      assert layout_34.img_size > 0
      assert layout_43.img_size > 0
    end

    test "net_axis is consistent with outer_rows/outer_cols" do
      layout = PanicTda.Export.compute_layout(3, 4, 1, 1920, 1080, :hd)

      case layout.net_axis do
        :row ->
          assert layout.outer_rows == 3
          assert layout.outer_cols == 4

        :col ->
          assert layout.outer_rows == 4
          assert layout.outer_cols == 3
      end
    end

    test "specific combo (1,1,1)" do
      layout = PanicTda.Export.compute_layout(1, 1, 1, 1920, 1080, :hd)
      assert layout.outer_rows == 1
      assert layout.outer_cols == 1
      assert layout.sub_rows == 1
      assert layout.sub_cols == 1
    end

    test "specific combo (2,2,4)" do
      layout = PanicTda.Export.compute_layout(2, 2, 4, 1920, 1080, :hd)
      assert layout.outer_rows * layout.outer_cols == 4
      assert layout.sub_rows * layout.sub_cols == 4
    end

    test "specific combo (3,4,12)" do
      layout = PanicTda.Export.compute_layout(3, 4, 12, 1920, 1080, :hd)
      assert layout.outer_rows * layout.outer_cols == 12
      assert layout.sub_rows * layout.sub_cols == 12
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
