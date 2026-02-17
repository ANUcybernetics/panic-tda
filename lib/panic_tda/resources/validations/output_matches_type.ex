defmodule PanicTda.Validations.OutputMatchesType do
  use Ash.Resource.Validation

  @impl true
  def init(opts), do: {:ok, opts}

  @impl true
  def validate(changeset, _opts, _context) do
    type = Ash.Changeset.get_attribute(changeset, :type)
    output_text = Ash.Changeset.get_attribute(changeset, :output_text)
    output_image = Ash.Changeset.get_attribute(changeset, :output_image)

    case type do
      :text ->
        cond do
          is_nil(output_text) ->
            {:error, field: :output_text, message: "must be set for text invocations"}

          not is_nil(output_image) ->
            {:error, field: :output_image, message: "must not be set for text invocations"}

          true ->
            :ok
        end

      :image ->
        cond do
          is_nil(output_image) ->
            {:error, field: :output_image, message: "must be set for image invocations"}

          not is_nil(output_text) ->
            {:error, field: :output_text, message: "must not be set for image invocations"}

          true ->
            :ok
        end

      _ ->
        :ok
    end
  end
end
