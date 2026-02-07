defmodule PanicTda.Validations.NonEmptySubArrays do
  use Ash.Resource.Validation

  @impl true
  def init(opts) do
    if is_atom(opts[:attribute]) do
      {:ok, opts}
    else
      {:error, "attribute must be an atom"}
    end
  end

  @impl true
  def validate(changeset, opts, _context) do
    value = Ash.Changeset.get_attribute(changeset, opts[:attribute])

    cond do
      is_nil(value) ->
        :ok

      Enum.all?(value, fn elem -> is_list(elem) and elem != [] end) ->
        :ok

      true ->
        {:error, field: opts[:attribute], message: "must not contain empty sub-arrays"}
    end
  end
end
