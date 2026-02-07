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
end
