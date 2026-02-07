defmodule PanicTda.Validations.NonEmptyList do
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
      nil -> :ok
      [] -> {:error, field: opts[:attribute], message: "must not be empty"}
      _ -> :ok
    end
  end
end
