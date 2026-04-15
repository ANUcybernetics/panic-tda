defmodule PanicTda.Export do
  require Ash.Query

  alias Vix.Vips.{Image, MutableOperation, Operation}

  @default_fps 10
  @default_crf 30

  @model_abbrev %{
    "SD35Medium" => "SD35",
    "Flux2Klein" => "F2Klein",
    "Flux2Dev" => "F2Dev",
    "ZImageTurbo" => "ZImg",
    "HunyuanImage" => "Hunyuan",
    "GLMImage" => "GLM",
    "Moondream" => "Moon",
    "Qwen25VL" => "Qwen",
    "Gemma3n" => "Gemma",
    "Pixtral" => "Pix",
    "LLaMA32Vision" => "LLaMA",
    "Florence2" => "Flor2"
  }

  def video(experiment_id, output_path, opts \\ []) do
    experiment =
      PanicTda.get_experiment!(experiment_id)
      |> Ash.load!(:runs)

    fps = Keyword.get(opts, :fps, @default_fps)
    crf = Keyword.get(opts, :quality, @default_crf)
    resolution = Keyword.get(opts, :resolution, :"4k")

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
      compute_layout(length(prompts), length(networks), num_runs, target_w, target_h, resolution)

    net_tile_w = layout.run_cols * layout.img_size
    measured_h = measure_net_label_height(networks, net_tile_w, layout.net_font_size)

    layout =
      if measured_h > layout.net_label_h do
        compute_layout(
          length(prompts),
          length(networks),
          num_runs,
          target_w,
          target_h,
          resolution,
          net_label_h: measured_h
        )
      else
        layout
      end

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

        frame =
          render_frame(border, layout, run_id_map, frame_images, prompts, networks, num_runs)

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

  def compute_layout(
        n_prompts,
        n_networks,
        n_runs,
        frame_w,
        frame_h,
        resolution \\ :hd,
        opts \\ []
      ) do
    prompt_shapes = grid_shapes(n_prompts)
    net_shapes = grid_shapes(n_networks)
    run_shapes = grid_shapes(n_runs)

    {prompt_gutter, net_gutter, prompt_label_h, default_net_label_h, prompt_font_size,
     net_font_size} =
      case resolution do
        :hd -> {16, 6, 40, 24, 20, 14}
        :"4k" -> {32, 12, 80, 48, 40, 28}
      end

    net_label_h = Keyword.get(opts, :net_label_h, default_net_label_h)

    candidates =
      for {pr, pc} <- prompt_shapes,
          {nr, nc} <- net_shapes,
          {rr, rc} <- run_shapes do
        img_w =
          (frame_w - (pc - 1) * prompt_gutter - pc * (nc - 1) * net_gutter) / (pc * nc * rc)

        img_h =
          (frame_h - (pr - 1) * prompt_gutter - pr * prompt_label_h - pr * nr * net_label_h -
             pr * (nr - 1) * net_gutter) / (pr * nr * rr)

        img_size = floor(min(img_w, img_h)) |> trunc()

        fill_ratio =
          n_prompts / (pr * pc) * (n_networks / (nr * nc)) * (n_runs / (rr * rc))

        %{
          img_size: img_size,
          score: img_size * fill_ratio,
          prompt_rows: pr,
          prompt_cols: pc,
          net_rows: nr,
          net_cols: nc,
          run_rows: rr,
          run_cols: rc
        }
      end

    best = Enum.max_by(candidates, & &1.score)

    min_img_size = if resolution == :"4k", do: 40, else: 20

    if best.img_size < min_img_size do
      raise ArgumentError,
            "Cannot produce acceptable video layout for " <>
              "#{n_prompts} prompts × #{n_networks} networks × #{n_runs} runs " <>
              "at #{resolution} resolution: best thumbnail size is #{best.img_size}px " <>
              "(minimum #{min_img_size}px). " <>
              "Try a higher resolution, fewer prompts/networks, or a run count with " <>
              "a balanced factorisation (1, 2, 3, 4, 6, 8, 9, 12, 16)."
    end

    net_tile_w = best.run_cols * best.img_size
    prompt_tile_w = best.net_cols * net_tile_w + (best.net_cols - 1) * net_gutter
    content_w = best.prompt_cols * prompt_tile_w + (best.prompt_cols - 1) * prompt_gutter

    net_tile_h = net_label_h + best.run_rows * best.img_size
    prompt_tile_h = prompt_label_h + best.net_rows * net_tile_h + (best.net_rows - 1) * net_gutter
    content_h = best.prompt_rows * prompt_tile_h + (best.prompt_rows - 1) * prompt_gutter

    offset_x = max(div(frame_w - content_w, 2), 0)
    offset_y = max(div(frame_h - content_h, 2), 0)

    %{
      img_size: best.img_size,
      prompt_rows: best.prompt_rows,
      prompt_cols: best.prompt_cols,
      net_rows: best.net_rows,
      net_cols: best.net_cols,
      run_rows: best.run_rows,
      run_cols: best.run_cols,
      prompt_gutter: prompt_gutter,
      net_gutter: net_gutter,
      prompt_label_h: prompt_label_h,
      net_label_h: net_label_h,
      prompt_font_size: prompt_font_size,
      net_font_size: net_font_size,
      canvas_w: frame_w,
      canvas_h: frame_h,
      offset_x: offset_x,
      offset_y: offset_y
    }
  end

  defp grid_shapes(n) do
    1..n
    |> Enum.map(fn c -> {div(n + c - 1, c), c} end)
    |> Enum.uniq()
  end

  defp build_border(layout, prompts, networks) do
    canvas = Operation.black!(layout.canvas_w, layout.canvas_h, bands: 3)

    net_tile_w = layout.run_cols * layout.img_size
    prompt_tile_w = layout.net_cols * net_tile_w + (layout.net_cols - 1) * layout.net_gutter
    net_tile_h = layout.net_label_h + layout.run_rows * layout.img_size

    prompt_tile_h =
      layout.prompt_label_h + layout.net_rows * net_tile_h +
        (layout.net_rows - 1) * layout.net_gutter

    prompt_opts = [
      font: "Public Sans #{layout.prompt_font_size}",
      rgba: true,
      width: max(prompt_tile_w, 50),
      align: :VIPS_ALIGN_CENTRE
    ]

    net_opts = [
      font: "Public Sans #{layout.net_font_size}",
      rgba: true,
      width: max(net_tile_w, 50),
      align: :VIPS_ALIGN_CENTRE
    ]

    for {prompt, prompt_idx} <- Enum.with_index(prompts), reduce: canvas do
      acc ->
        p_row = div(prompt_idx, layout.prompt_cols)
        p_col = rem(prompt_idx, layout.prompt_cols)
        ptx = layout.offset_x + p_col * (prompt_tile_w + layout.prompt_gutter)
        pty = layout.offset_y + p_row * (prompt_tile_h + layout.prompt_gutter)

        max_chars = max(div(prompt_tile_w, max(div(layout.prompt_font_size, 2), 1)), 10)

        truncated =
          if String.length(prompt) > max_chars,
            do: String.slice(prompt, 0, max_chars - 3) <> "...",
            else: prompt

        {prompt_img, _} = Operation.text!(truncated, prompt_opts)
        prompt_img = text_to_colour(prompt_img, 255)

        label_y = pty + div(max(layout.prompt_label_h - Image.height(prompt_img), 0), 2)
        label_x = ptx + div(max(prompt_tile_w - Image.width(prompt_img), 0), 2)
        acc = Operation.insert!(acc, prompt_img, max(label_x, ptx), max(label_y, pty))

        for {network, net_idx} <- Enum.with_index(networks), reduce: acc do
          acc2 ->
            n_row = div(net_idx, layout.net_cols)
            n_col = rem(net_idx, layout.net_cols)
            ntx = ptx + n_col * (net_tile_w + layout.net_gutter)
            nty = pty + layout.prompt_label_h + n_row * (net_tile_h + layout.net_gutter)

            net_text = net_label_text(network)
            {net_img, _} = Operation.text!(net_text, net_opts)
            net_img = text_to_colour(net_img, 180)

            ly = nty + div(max(layout.net_label_h - Image.height(net_img), 0), 2)
            lx = ntx + div(max(net_tile_w - Image.width(net_img), 0), 2)
            Operation.insert!(acc2, net_img, max(lx, ntx), max(ly, nty))
        end
    end
  end

  defp render_frame(border, layout, run_id_map, frame_images, prompts, networks, num_runs) do
    net_tile_w = layout.run_cols * layout.img_size
    prompt_tile_w = layout.net_cols * net_tile_w + (layout.net_cols - 1) * layout.net_gutter
    net_tile_h = layout.net_label_h + layout.run_rows * layout.img_size

    prompt_tile_h =
      layout.prompt_label_h + layout.net_rows * net_tile_h +
        (layout.net_rows - 1) * layout.net_gutter

    {:ok, frame} =
      Image.mutate(border, fn mut ->
        for {prompt, prompt_idx} <- Enum.with_index(prompts),
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

              p_row = div(prompt_idx, layout.prompt_cols)
              p_col = rem(prompt_idx, layout.prompt_cols)
              ptx = layout.offset_x + p_col * (prompt_tile_w + layout.prompt_gutter)
              pty = layout.offset_y + p_row * (prompt_tile_h + layout.prompt_gutter)

              n_row = div(net_idx, layout.net_cols)
              n_col = rem(net_idx, layout.net_cols)
              ntx = ptx + n_col * (net_tile_w + layout.net_gutter)
              nty = pty + layout.prompt_label_h + n_row * (net_tile_h + layout.net_gutter)

              r_row = div(run_number, layout.run_cols)
              r_col = rem(run_number, layout.run_cols)
              x = ntx + r_col * layout.img_size
              y = nty + layout.net_label_h + r_row * layout.img_size

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
    |> Ash.Query.filter(
      run_id in ^run_ids and sequence_number == ^seq and not is_nil(output_image)
    )
    |> Ash.read!()
    |> Map.new(fn inv -> {inv.run_id, inv.output_image} end)
  end

  defp abbreviate(model_name), do: Map.get(@model_abbrev, model_name, model_name)

  defp net_label_text(network), do: network |> Enum.map(&abbreviate/1) |> Enum.join(" ⇄ ")

  defp measure_net_label_height(networks, width, font_size) do
    opts = [font: "Public Sans #{font_size}", rgba: true, width: max(width, 50)]

    networks
    |> Enum.map(fn network ->
      {img, _} = Operation.text!(net_label_text(network), opts)
      Image.height(img)
    end)
    |> Enum.max()
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
