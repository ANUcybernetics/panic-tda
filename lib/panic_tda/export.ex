defmodule PanicTda.Export do
  require Ash.Query

  alias Vix.Vips.{Image, Operation}

  @image_size 256
  @gap 50
  @border_width 2 * @image_size
  @border_height @image_size
  @default_fps 10
  @default_crf 22

  def video(experiment_id, output_path, opts \\ []) do
    experiment =
      PanicTda.get_experiment!(experiment_id)
      |> Ash.load!(runs: [invocations: Ash.Query.sort(PanicTda.Invocation, sequence_number: :asc)])

    fps = Keyword.get(opts, :fps, @default_fps)
    crf = Keyword.get(opts, :quality, @default_crf)

    resolution =
      case Keyword.get(opts, :resolution, :hd) do
        :hd -> {1920, 1080}
        :"4k" -> {3840, 2160}
      end

    prompts = experiment.prompts
    networks = experiment.networks
    num_runs = experiment.num_runs
    max_length = experiment.max_length

    run_map = build_run_map(experiment.runs)

    {cell_w, cell_h, sub_cols, sub_rows} = cell_layout(num_runs)
    grid_w = length(networks) * cell_w + (length(networks) - 1) * @gap
    grid_h = length(prompts) * cell_h + (length(prompts) - 1) * @gap
    canvas_w = @border_width + grid_w
    canvas_h = @border_height + grid_h

    border = build_border(prompts, networks, canvas_w, canvas_h, cell_w, cell_h)

    tmp_dir = System.tmp_dir!() |> Path.join("panic_tda_export_#{System.unique_integer([:positive])}")
    File.mkdir_p!(tmp_dir)

    try do
      for seq <- 0..(max_length - 1) do
        frame = render_frame(border, run_map, prompts, networks, num_runs, seq, cell_w, sub_cols, sub_rows)
        frame_path = Path.join(tmp_dir, "frame_#{String.pad_leading(Integer.to_string(seq), 6, "0")}.jpg")
        :ok = Image.write_to_file(frame, frame_path, Q: 95)
      end

      {scale_w, scale_h} = resolution
      {_output, 0} =
        System.cmd("ffmpeg", [
          "-y",
          "-framerate", Integer.to_string(fps),
          "-i", Path.join(tmp_dir, "frame_%06d.jpg"),
          "-c:v", "libx265",
          "-preset", "medium",
          "-crf", Integer.to_string(crf),
          "-vf", "scale=w=#{scale_w}:h=#{scale_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=#{scale_w}:#{scale_h}:(ow-iw)/2:(oh-ih)/2",
          "-pix_fmt", "yuv420p",
          "-tag:v", "hvc1",
          "-movflags", "+faststart",
          output_path
        ], stderr_to_stdout: true)

      {:ok, output_path}
    after
      File.rm_rf!(tmp_dir)
    end
  end

  def image(invocation_id, output_path) do
    invocation =
      PanicTda.Invocation
      |> Ash.Query.filter(id == ^invocation_id)
      |> Ash.read_one!()

    case invocation do
      nil ->
        {:error, :not_found}

      %{output_image: nil} ->
        {:error, :no_image}

      %{output_image: data} ->
        ext = Path.extname(output_path) |> String.downcase()

        if ext == ".avif" do
          File.write!(output_path, data)
        else
          {:ok, img} = Image.new_from_buffer(data)
          :ok = Image.write_to_file(img, output_path)
        end

        {:ok, output_path}
    end
  end

  defp build_run_map(runs) do
    Map.new(runs, fn run ->
      {{run.network, run.initial_prompt, run.run_number}, run}
    end)
  end

  defp cell_layout(num_runs) do
    {sub_cols, sub_rows} =
      if num_runs <= 4 do
        {num_runs, 1}
      else
        cols = ceil(:math.sqrt(num_runs * 16 / 9)) |> trunc()
        rows = ceil(num_runs / cols) |> trunc()
        {cols, rows}
      end

    {sub_cols * @image_size, sub_rows * @image_size, sub_cols, sub_rows}
  end

  defp build_border(prompts, networks, canvas_w, canvas_h, cell_w, cell_h) do
    canvas = Operation.black!(canvas_w, canvas_h, bands: 3)

    canvas =
      networks
      |> Enum.with_index()
      |> Enum.reduce(canvas, fn {network, col_idx}, acc ->
        label_text = Enum.join(network, " → ")
        {label, _} = Operation.text!(label_text, font: "Sans 16", rgba: true, width: cell_w)
        label = ensure_3band(label)
        x = @border_width + col_idx * (cell_w + @gap) + div(cell_w - Image.width(label), 2)
        y = div(@border_height - Image.height(label), 2)
        x = max(x, @border_width + col_idx * (cell_w + @gap))
        Operation.insert!(acc, label, x, y)
      end)

    prompts
    |> Enum.with_index()
    |> Enum.reduce(canvas, fn {prompt, row_idx}, acc ->
      truncated = if String.length(prompt) > 40, do: String.slice(prompt, 0, 37) <> "...", else: prompt
      {label, _} = Operation.text!(truncated, font: "Sans 14", rgba: true, width: @border_width - 20)
      label = ensure_3band(label)
      y = @border_height + row_idx * (cell_h + @gap) + div(cell_h - Image.height(label), 2)
      x = div(@border_width - Image.width(label), 2)
      x = max(x, 10)
      Operation.insert!(acc, label, x, y)
    end)
  end

  defp render_frame(border, run_map, prompts, networks, num_runs, seq, cell_w, sub_cols, _sub_rows) do
    prompts
    |> Enum.with_index()
    |> Enum.reduce(border, fn {prompt, row_idx}, frame ->
      networks
      |> Enum.with_index()
      |> Enum.reduce(frame, fn {network, col_idx}, frame ->
        base_x = @border_width + col_idx * (cell_w + @gap)
        base_y = @border_height + row_idx * (cell_w + @gap)

        0..(num_runs - 1)
        |> Enum.reduce(frame, fn run_number, frame ->
          run = Map.get(run_map, {network, prompt, run_number})

          case run && Enum.find(run.invocations, &(&1.sequence_number == seq)) do
            %{output_image: data} when not is_nil(data) ->
              thumb = Operation.thumbnail_buffer!(data, @image_size, height: @image_size, size: :VIPS_SIZE_FORCE)
              thumb = ensure_3band(thumb)
              sub_col = rem(run_number, sub_cols)
              sub_row = div(run_number, sub_cols)
              x = base_x + sub_col * @image_size
              y = base_y + sub_row * @image_size
              Operation.insert!(frame, thumb, x, y)

            _ ->
              frame
          end
        end)
      end)
    end)
  end

  defp ensure_3band(img) do
    case Image.bands(img) do
      4 -> Operation.flatten!(img, background: [0.0, 0.0, 0.0])
      1 -> Operation.bandjoin_const!(img, [0.0, 0.0])
      _ -> img
    end
  end
end
