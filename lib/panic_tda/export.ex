defmodule PanicTda.Export do
  require Ash.Query

  alias Vix.Vips.{Image, MutableOperation, Operation}

  @default_fps 10
  @default_crf 30

  @grid_shapes %{
    1 => [{1, 1}],
    2 => [{1, 2}, {2, 1}],
    3 => [{1, 3}, {3, 1}],
    4 => [{2, 2}],
    6 => [{2, 3}, {3, 2}],
    8 => [{2, 4}, {4, 2}],
    9 => [{3, 3}],
    12 => [{3, 4}, {4, 3}],
    16 => [{4, 4}]
  }

  @valid_counts Map.keys(@grid_shapes) |> Enum.sort()

  def video(experiment_id, output_path, opts \\ []) do
    experiment =
      PanicTda.get_experiment!(experiment_id)
      |> Ash.load!(:runs)

    fps = Keyword.get(opts, :fps, @default_fps)
    crf = Keyword.get(opts, :quality, @default_crf)
    resolution = Keyword.get(opts, :resolution, :hd)

    {target_w, target_h} =
      case resolution do
        :hd -> {1920, 1080}
        :"4k" -> {3840, 2160}
      end

    prompts = experiment.prompts
    networks = experiment.networks
    num_runs = experiment.num_runs
    max_length = experiment.max_length

    layout =
      compute_layout(length(networks), length(prompts), num_runs, target_w, target_h, resolution)

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
            "libsvtav1",
            "-preset",
            "6",
            "-crf",
            Integer.to_string(crf),
            "-pix_fmt",
            "yuv420p10le",
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

  def compute_layout(n_net, n_prom, n_run, frame_w, frame_h, resolution \\ :hd) do
    outer_count = n_net * n_prom

    sub_shapes = lookup_shapes!(n_run, "num_runs")
    outer_shapes = lookup_shapes!(outer_count, "num_networks × num_prompts")

    {gutter, label_h, label_pad, font_size} =
      case resolution do
        :hd -> {8, 48, 4, 20}
        :"4k" -> {16, 96, 8, 40}
      end

    outer_gutter = gutter * 2

    candidates =
      for {o_r, o_c} <- outer_shapes,
          net_axis <- [:row, :col],
          valid_orientation?(o_r, o_c, n_net, n_prom, net_axis),
          {s_r, s_c} <- sub_shapes do
        img_w = (frame_w - (o_c - 1) * outer_gutter - o_c * (s_c - 1) * gutter) / (o_c * s_c)
        img_h = (frame_h - (o_r - 1) * outer_gutter - o_r * (label_h + label_pad) - o_r * (s_r - 1) * gutter) / (o_r * s_r)
        img_size = floor(min(img_w, img_h)) |> trunc()

        %{
          img_size: img_size,
          outer_rows: o_r,
          outer_cols: o_c,
          sub_rows: s_r,
          sub_cols: s_c,
          net_axis: net_axis
        }
      end

    best = Enum.max_by(candidates, & &1.img_size)

    content_w =
      best.outer_cols * best.sub_cols * best.img_size +
        (best.outer_cols - 1) * outer_gutter +
        best.outer_cols * (best.sub_cols - 1) * gutter

    content_h =
      best.outer_rows * best.sub_rows * best.img_size +
        (best.outer_rows - 1) * outer_gutter +
        best.outer_rows * (label_h + label_pad) +
        best.outer_rows * (best.sub_rows - 1) * gutter

    offset_x = max(div(frame_w - content_w, 2), 0)
    offset_y = max(div(frame_h - content_h, 2), 0)

    %{
      img_size: best.img_size,
      outer_rows: best.outer_rows,
      outer_cols: best.outer_cols,
      sub_rows: best.sub_rows,
      sub_cols: best.sub_cols,
      gutter: gutter,
      outer_gutter: outer_gutter,
      label_h: label_h,
      label_pad: label_pad,
      font_size: font_size,
      canvas_w: frame_w,
      canvas_h: frame_h,
      offset_x: offset_x,
      offset_y: offset_y,
      net_axis: best.net_axis
    }
  end

  defp lookup_shapes!(count, label) do
    case Map.fetch(@grid_shapes, count) do
      {:ok, shapes} ->
        shapes

      :error ->
        raise ArgumentError,
              "unsupported #{label} count #{count}; valid counts are #{inspect(@valid_counts)}"
    end
  end

  defp valid_orientation?(o_r, o_c, n_net, n_prom, :row),
    do: o_r == n_net and o_c == n_prom

  defp valid_orientation?(o_r, o_c, n_net, n_prom, :col),
    do: o_r == n_prom and o_c == n_net

  defp build_border(layout, prompts, networks) do
    canvas = Operation.black!(layout.canvas_w, layout.canvas_h, bands: 3)
    font_size = layout.font_size

    subgrid_w =
      layout.sub_cols * layout.img_size + (layout.sub_cols - 1) * layout.gutter

    net_font_size = round(font_size * 0.8)
    prompt_opts = [font: "Public Sans #{font_size}", rgba: true, width: max(subgrid_w, 50), align: :VIPS_ALIGN_CENTRE]
    net_opts = [font: "Public Sans #{net_font_size}", rgba: true, width: max(subgrid_w, 50), align: :VIPS_ALIGN_CENTRE]
    {combined, _} = Operation.text!("X\nX", prompt_opts)
    {single, _} = Operation.text!("X", prompt_opts)
    line_gap = Image.height(combined) - 2 * Image.height(single)

    for {network, net_idx} <- Enum.with_index(networks),
        {prompt, prom_idx} <- Enum.with_index(prompts),
        reduce: canvas do
      acc ->
        {o_r, o_c} = outer_position(layout, net_idx, prom_idx)

        cell_x =
          layout.offset_x +
            o_c * (subgrid_w + layout.outer_gutter)

        cell_y =
          layout.offset_y +
            o_r * (layout.sub_rows * layout.img_size + (layout.sub_rows - 1) * layout.gutter + layout.label_h + layout.label_pad + layout.outer_gutter)

        net_text = Enum.join(network, " ⇄ ")

        max_chars = max(div(subgrid_w, max(div(font_size, 2), 1)), 10)

        truncated_prompt =
          if String.length(prompt) > max_chars,
            do: String.slice(prompt, 0, max_chars - 3) <> "...",
            else: prompt

        {net_img, _} = Operation.text!(net_text, net_opts)
        net_img = text_to_colour(net_img, 180)

        {prompt_img, _} = Operation.text!(truncated_prompt, prompt_opts)
        prompt_img = text_to_colour(prompt_img, 255)

        total_h = Image.height(net_img) + line_gap + Image.height(prompt_img)
        start_y = cell_y + div(max(layout.label_h - total_h, 0), 2)

        net_lx = cell_x + div(max(subgrid_w - Image.width(net_img), 0), 2)
        prompt_lx = cell_x + div(max(subgrid_w - Image.width(prompt_img), 0), 2)

        acc = Operation.insert!(acc, net_img, max(net_lx, cell_x), max(start_y, cell_y))
        Operation.insert!(acc, prompt_img, max(prompt_lx, cell_x), max(start_y + Image.height(net_img) + line_gap, cell_y))
    end
  end

  defp outer_position(layout, net_idx, prom_idx) do
    case layout.net_axis do
      :row -> {net_idx, prom_idx}
      :col -> {prom_idx, net_idx}
    end
  end

  defp render_frame(border, layout, run_id_map, frame_images, prompts, networks, num_runs) do
    subgrid_w =
      layout.sub_cols * layout.img_size + (layout.sub_cols - 1) * layout.gutter

    {:ok, frame} =
      Image.mutate(border, fn mut ->
        for {prompt, prom_idx} <- Enum.with_index(prompts),
            {network, net_idx} <- Enum.with_index(networks),
            run_number <- 0..(num_runs - 1) do
          run_id = Map.get(run_id_map, {network, prompt, run_number})

          case run_id && Map.get(frame_images, run_id) do
            data when not is_nil(data) ->
              thumb =
                Operation.thumbnail_buffer!(data, layout.img_size,
                  height: layout.img_size,
                  size: :VIPS_SIZE_FORCE
                )

              thumb = ensure_3band(thumb) |> materialize()

              {o_r, o_c} = outer_position(layout, net_idx, prom_idx)

              cell_x =
                layout.offset_x +
                  o_c * (subgrid_w + layout.outer_gutter)

              cell_y =
                layout.offset_y +
                  o_r * (layout.sub_rows * layout.img_size + (layout.sub_rows - 1) * layout.gutter + layout.label_h + layout.label_pad + layout.outer_gutter) +
                  layout.label_h + layout.label_pad

              sub_col = rem(run_number, layout.sub_cols)
              sub_row = div(run_number, layout.sub_cols)
              x = cell_x + sub_col * (layout.img_size + layout.gutter)
              y = cell_y + sub_row * (layout.img_size + layout.gutter)
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

  defp text_to_colour(img, level) do
    case Image.bands(img) do
      4 ->
        alpha = Operation.extract_band!(img, 3)
        scaled = Operation.linear!(alpha, [level / 255.0], [0.0])
        Operation.bandjoin!([scaled, scaled, scaled])

      _ ->
        ensure_3band(img)
    end
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
