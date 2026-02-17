defmodule PanicTda.Export do
  require Ash.Query

  alias Vix.Vips.{Image, MutableOperation, Operation}

  @default_fps 10
  @default_crf 22
  @max_thumb_size 512
  @min_thumb_size 16

  def video(experiment_id, output_path, opts \\ []) do
    experiment =
      PanicTda.get_experiment!(experiment_id)
      |> Ash.load!(:runs)

    fps = Keyword.get(opts, :fps, @default_fps)
    crf = Keyword.get(opts, :quality, @default_crf)

    {target_w, target_h} =
      case Keyword.get(opts, :resolution, :hd) do
        :hd -> {1920, 1080}
        :"4k" -> {3840, 2160}
      end

    prompts = experiment.prompts
    networks = experiment.networks
    num_runs = experiment.num_runs
    max_length = experiment.max_length

    layout = compute_layout(length(networks), length(prompts), num_runs, target_w, target_h)

    run_id_map =
      Map.new(experiment.runs, fn run ->
        {{run.network, run.initial_prompt, run.run_number}, run.id}
      end)

    all_run_ids = Enum.map(experiment.runs, & &1.id)
    border = build_border(layout, prompts, networks) |> materialize()

    tmp_dir =
      System.tmp_dir!()
      |> Path.join("panic_tda_export_#{System.unique_integer([:positive])}")

    File.mkdir_p!(tmp_dir)

    try do
      for seq <- 0..(max_length - 1) do
        frame_images = load_frame_images(all_run_ids, seq)
        frame = render_frame(border, layout, run_id_map, frame_images, prompts, networks, num_runs)

        frame_path =
          Path.join(tmp_dir, "frame_#{String.pad_leading(Integer.to_string(seq), 6, "0")}.jpg")

        :ok = Image.write_to_file(frame, frame_path, Q: 95)
      end

      {_output, 0} =
        System.cmd(
          "ffmpeg",
          [
            "-y",
            "-framerate",
            Integer.to_string(fps),
            "-i",
            Path.join(tmp_dir, "frame_%06d.jpg"),
            "-c:v",
            "libx265",
            "-preset",
            "medium",
            "-crf",
            Integer.to_string(crf),
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "hvc1",
            "-movflags",
            "+faststart",
            output_path
          ],
          stderr_to_stdout: true
        )

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

  def compute_layout(n_net, n_prom, n_run, target_w, target_h) do
    sub_cols = :math.sqrt(n_run) |> ceil() |> trunc()
    sub_rows = ceil(n_run / sub_cols) |> trunc()

    {wrap, _score} = find_best_wrap(n_net, n_prom, sub_cols, sub_rows, target_w, target_h)

    eff_cols = ceil(n_net / wrap) |> trunc()
    n_bands = wrap

    font_size = compute_font_size(target_w, target_h, eff_cols, n_prom * n_bands)
    label_w = compute_label_width(font_size, target_w)
    label_h = compute_label_height(font_size)

    avail_w = target_w - label_w
    avail_h = target_h - n_bands * label_h

    gap_frac = 0.15
    band_gap_frac = 0.40

    thumb_w =
      avail_w / (eff_cols * sub_cols + (eff_cols - 1) * gap_frac * sub_cols)

    band_gaps_total = if n_bands > 1, do: (n_bands - 1) * band_gap_frac, else: 0.0
    rows_total = n_prom * n_bands
    inner_gaps = rows_total - n_bands + band_gaps_total

    thumb_h =
      avail_h / (rows_total * sub_rows + inner_gaps * sub_rows)

    thumb_size = min(thumb_w, thumb_h) |> floor() |> trunc()
    thumb_size = thumb_size |> max(@min_thumb_size) |> min(@max_thumb_size)

    gap_px = max(round(thumb_size * gap_frac), 1)
    band_gap_px = max(round(thumb_size * band_gap_frac), 1)

    cell_w = sub_cols * thumb_size
    cell_h = sub_rows * thumb_size

    grid_w = eff_cols * cell_w + max(eff_cols - 1, 0) * gap_px
    band_h = n_prom * cell_h + max(n_prom - 1, 0) * gap_px
    total_grid_h = n_bands * (label_h + band_h) + max(n_bands - 1, 0) * band_gap_px

    content_w = label_w + grid_w
    content_h = total_grid_h

    offset_x = max(div(target_w - content_w, 2), 0)
    offset_y = max(div(target_h - content_h, 2), 0)

    %{
      thumb_size: thumb_size,
      sub_cols: sub_cols,
      sub_rows: sub_rows,
      eff_cols: eff_cols,
      wrap: wrap,
      n_bands: n_bands,
      gap_px: gap_px,
      band_gap_px: band_gap_px,
      cell_w: cell_w,
      cell_h: cell_h,
      label_w: label_w,
      label_h: label_h,
      font_size: font_size,
      canvas_w: target_w,
      canvas_h: target_h,
      offset_x: offset_x,
      offset_y: offset_y,
      n_prom: n_prom,
      n_net: n_net
    }
  end

  defp find_best_wrap(n_net, n_prom, sub_cols, sub_rows, target_w, target_h) do
    target_ratio = target_w / target_h

    1..n_net
    |> Enum.map(fn w ->
      cols = ceil(n_net / w) |> trunc()
      rows = n_prom * w
      aspect = cols * sub_cols / (rows * sub_rows)
      score = abs(aspect - target_ratio)
      {w, score}
    end)
    |> Enum.min_by(fn {_w, score} -> score end)
  end

  defp compute_font_size(target_w, _target_h, eff_cols, total_rows) do
    size_from_width = max(div(target_w, eff_cols * 20), 8)
    size_from_height = max(div(target_w, total_rows * 15), 8)
    min(size_from_width, size_from_height) |> min(24)
  end

  defp compute_label_width(font_size, target_w) do
    max(font_size * 16, div(target_w, 10))
  end

  defp compute_label_height(font_size) do
    font_size * 3
  end

  defp build_border(layout, prompts, networks) do
    canvas = Operation.black!(layout.canvas_w, layout.canvas_h, bands: 3)

    net_chunks =
      networks
      |> Enum.with_index()
      |> Enum.chunk_every(layout.eff_cols)

    net_chunks
    |> Enum.with_index()
    |> Enum.reduce(canvas, fn {band_nets, band_idx}, acc ->
      band_y = band_top_y(layout, band_idx)

      acc = render_network_labels(acc, layout, band_nets, band_y)
      render_prompt_labels(acc, layout, prompts, band_y + layout.label_h)
    end)
  end

  defp render_network_labels(canvas, layout, band_nets, band_y) do
    Enum.reduce(band_nets, canvas, fn {network, global_idx}, acc ->
      col_in_band = rem(global_idx, layout.eff_cols)
      label_text = Enum.join(network, " → ")

      {label, _} =
        Operation.text!(label_text,
          font: "Sans #{layout.font_size}",
          rgba: true,
          width: layout.cell_w
        )

      label = ensure_3band(label)

      x =
        layout.offset_x + layout.label_w + col_in_band * (layout.cell_w + layout.gap_px) +
          div(max(layout.cell_w - Image.width(label), 0), 2)

      y = band_y + div(max(layout.label_h - Image.height(label), 0), 2)
      x = max(x, layout.offset_x + layout.label_w + col_in_band * (layout.cell_w + layout.gap_px))
      Operation.insert!(acc, label, x, y)
    end)
  end

  defp render_prompt_labels(canvas, layout, prompts, grid_y) do
    prompts
    |> Enum.with_index()
    |> Enum.reduce(canvas, fn {prompt, row_idx}, acc ->
      truncated =
        if String.length(prompt) > 40,
          do: String.slice(prompt, 0, 37) <> "...",
          else: prompt

      max_label_w = layout.label_w - 10

      {label, _} =
        Operation.text!(truncated,
          font: "Sans #{max(layout.font_size - 2, 8)}",
          rgba: true,
          width: max(max_label_w, 50)
        )

      label = ensure_3band(label)

      y =
        grid_y + row_idx * (layout.cell_h + layout.gap_px) +
          div(max(layout.cell_h - Image.height(label), 0), 2)

      x = layout.offset_x + div(max(layout.label_w - Image.width(label), 0), 2)
      x = max(x, layout.offset_x + 5)
      Operation.insert!(acc, label, x, y)
    end)
  end

  defp band_top_y(layout, band_idx) do
    band_h = layout.n_prom * layout.cell_h + max(layout.n_prom - 1, 0) * layout.gap_px
    full_band_h = layout.label_h + band_h

    layout.offset_y + band_idx * (full_band_h + layout.band_gap_px)
  end

  defp render_frame(border, layout, run_id_map, frame_images, prompts, networks, num_runs) do
    {:ok, frame} =
      Image.mutate(border, fn mut ->
        for {prompt, prompt_idx} <- Enum.with_index(prompts),
            {network, net_global_idx} <- Enum.with_index(networks),
            run_number <- 0..(num_runs - 1) do
          run_id = Map.get(run_id_map, {network, prompt, run_number})

          case run_id && Map.get(frame_images, run_id) do
            data when not is_nil(data) ->
              thumb =
                Operation.thumbnail_buffer!(data, layout.thumb_size,
                  height: layout.thumb_size,
                  size: :VIPS_SIZE_FORCE
                )

              thumb = ensure_3band(thumb) |> materialize()

              band_idx = div(net_global_idx, layout.eff_cols)
              col_in_band = rem(net_global_idx, layout.eff_cols)
              band_y = band_top_y(layout, band_idx)
              grid_y = band_y + layout.label_h

              base_x =
                layout.offset_x + layout.label_w +
                  col_in_band * (layout.cell_w + layout.gap_px)

              base_y = grid_y + prompt_idx * (layout.cell_h + layout.gap_px)

              sub_col = rem(run_number, layout.sub_cols)
              sub_row = div(run_number, layout.sub_cols)
              x = base_x + sub_col * layout.thumb_size
              y = base_y + sub_row * layout.thumb_size
              :ok = MutableOperation.draw_image(mut, thumb, x, y)

            _ ->
              :ok
          end
        end

        :ok
      end)

    frame
  end

  defp load_frame_images(run_ids, seq) do
    PanicTda.Invocation
    |> Ash.Query.filter(run_id in ^run_ids and sequence_number == ^seq and not is_nil(output_image))
    |> Ash.read!()
    |> Map.new(fn inv -> {inv.run_id, inv.output_image} end)
  end

  defp materialize(img) do
    {:ok, img} = Image.copy_memory(img)
    img
  end

  defp ensure_3band(img) do
    case Image.bands(img) do
      4 -> Operation.flatten!(img, background: [0.0, 0.0, 0.0])
      1 -> Operation.bandjoin_const!(img, [0.0, 0.0])
      _ -> img
    end
  end
end
