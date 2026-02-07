defmodule PanicTda.Validations.NonEmptyNestedLists do
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

    case value do
      nil ->
        :ok

      [] ->
        {:error, field: opts[:attribute], message: "must not be empty"}

      lists when is_list(lists) ->
        if Enum.all?(lists, fn item -> is_list(item) and item != [] end) do
          :ok
        else
          {:error, field: opts[:attribute], message: "each network must not be empty"}
        end

      _ ->
        :ok
    end
  end
end
