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

  defp content_dimensions(layout) do
    net_tile_w = layout.run_cols * layout.img_size

    prompt_tile_w =
      layout.net_cols * net_tile_w + (layout.net_cols - 1) * layout.net_gutter

    content_w =
      layout.prompt_cols * prompt_tile_w + (layout.prompt_cols - 1) * layout.prompt_gutter

    net_tile_h = layout.net_label_h + layout.run_rows * layout.img_size

    prompt_tile_h =
      layout.prompt_label_h + layout.net_rows * net_tile_h +
        (layout.net_rows - 1) * layout.net_gutter

    content_h =
      layout.prompt_rows * prompt_tile_h + (layout.prompt_rows - 1) * layout.prompt_gutter

    {content_w, content_h}
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
        {1, 3, 2},
        {7, 1, 1},
        {1, 5, 3}
      ]

      for {n_prom, n_net, n_run} <- combos,
          resolution <- [:hd, :"4k"] do
        {frame_w, frame_h} = if resolution == :hd, do: {1920, 1080}, else: {3840, 2160}

        layout =
          PanicTda.Export.compute_layout(n_prom, n_net, n_run, frame_w, frame_h, resolution)

        assert layout.img_size > 0,
               "img_size for #{n_prom}p×#{n_net}n×#{n_run}r @ #{resolution}"

        assert layout.prompt_rows * layout.prompt_cols >= n_prom
        assert layout.net_rows * layout.net_cols >= n_net
        assert layout.run_rows * layout.run_cols >= n_run
      end
    end

    test "arbitrary grid counts produce valid layouts" do
      for n_run <- [5, 7, 10] do
        layout = PanicTda.Export.compute_layout(1, 1, n_run, 1920, 1080, :hd)
        assert layout.img_size > 0
        assert layout.run_rows * layout.run_cols >= n_run
      end

      for {n_prom, n_net} <- [{1, 5}, {5, 1}, {1, 7}, {7, 1}] do
        layout = PanicTda.Export.compute_layout(n_prom, n_net, 1, 1920, 1080, :hd)
        assert layout.img_size > 0
        assert layout.prompt_rows * layout.prompt_cols >= n_prom
        assert layout.net_rows * layout.net_cols >= n_net
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
      combos = [{1, 1, 1}, {2, 2, 4}, {3, 4, 3}, {3, 4, 12}, {7, 1, 1}, {1, 5, 3}]

      for {n_prom, n_net, n_run} <- combos,
          resolution <- [:hd, :"4k"] do
        {frame_w, frame_h} = if resolution == :hd, do: {1920, 1080}, else: {3840, 2160}

        layout =
          PanicTda.Export.compute_layout(n_prom, n_net, n_run, frame_w, frame_h, resolution)

        {content_w, content_h} = content_dimensions(layout)

        assert content_w <= frame_w,
               "content too wide for #{n_prom}p×#{n_net}n×#{n_run}r @ #{resolution}: #{content_w} > #{frame_w}"

        assert content_h <= frame_h,
               "content too tall for #{n_prom}p×#{n_net}n×#{n_run}r @ #{resolution}: #{content_h} > #{frame_h}"
      end
    end

    test "minimal dead margin: at least one dimension has <= 5% dead space" do
      combos = [{1, 1, 1}, {2, 2, 4}, {3, 4, 3}, {3, 4, 12}, {1, 1, 9}]

      for {n_prom, n_net, n_run} <- combos,
          resolution <- [:hd, :"4k"] do
        {frame_w, frame_h} = if resolution == :hd, do: {1920, 1080}, else: {3840, 2160}

        layout =
          PanicTda.Export.compute_layout(n_prom, n_net, n_run, frame_w, frame_h, resolution)

        {content_w, content_h} = content_dimensions(layout)

        margin_w_pct = (frame_w - content_w) / frame_w
        margin_h_pct = (frame_h - content_h) / frame_h

        assert margin_w_pct <= 0.05 or margin_h_pct <= 0.05,
               "too much dead space for #{n_prom}p×#{n_net}n×#{n_run}r @ #{resolution}: " <>
                 "w=#{Float.round(margin_w_pct * 100, 1)}%, h=#{Float.round(margin_h_pct * 100, 1)}%"
      end
    end

    test "asymmetric counts produce valid layouts" do
      layout_34 = PanicTda.Export.compute_layout(3, 4, 3, 1920, 1080, :hd)
      layout_43 = PanicTda.Export.compute_layout(4, 3, 3, 1920, 1080, :hd)

      assert layout_34.img_size > 0
      assert layout_43.img_size > 0
    end

    test "specific combo (1,1,1)" do
      layout = PanicTda.Export.compute_layout(1, 1, 1, 1920, 1080, :hd)
      assert layout.prompt_rows == 1
      assert layout.prompt_cols == 1
      assert layout.net_rows == 1
      assert layout.net_cols == 1
      assert layout.run_rows == 1
      assert layout.run_cols == 1
    end

    test "specific combo (2,2,4)" do
      layout = PanicTda.Export.compute_layout(2, 2, 4, 1920, 1080, :hd)
      assert layout.prompt_rows * layout.prompt_cols >= 2
      assert layout.net_rows * layout.net_cols >= 2
      assert layout.run_rows * layout.run_cols >= 4
    end

    test "specific combo (3,4,12)" do
      layout = PanicTda.Export.compute_layout(3, 4, 12, 1920, 1080, :hd)
      assert layout.prompt_rows * layout.prompt_cols >= 3
      assert layout.net_rows * layout.net_cols >= 4
      assert layout.run_rows * layout.run_cols >= 12
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
