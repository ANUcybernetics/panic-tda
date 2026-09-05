defmodule PanicTda.Models.ImageConverter do
  alias Vix.Vips.{Image, Operation}

  def to_avif!(binary, opts \\ []) when is_binary(binary) do
    quality = Keyword.get(opts, :quality, 50)

    {:ok, image} = Image.new_from_buffer(binary)

    {:ok, avif_binary} =
      Operation.heifsave_buffer(image,
        compression: :VIPS_FOREIGN_HEIF_COMPRESSION_AV1,
        Q: quality
      )

    avif_binary
  end

  @doc """
  Convert a list of images to AVIF concurrently, preserving order.

  AV1 encoding costs ~150 ms per 1024 px image and a batched step returns
  dozens at once, so encoding them serially left the GPU idle.
  """
  def to_avif_many!(binaries, opts \\ []) when is_list(binaries) do
    binaries
    |> Task.async_stream(&to_avif!(&1, opts),
      max_concurrency: System.schedulers_online(),
      timeout: 60_000
    )
    |> Enum.map(fn {:ok, avif} -> avif end)
  end
end
