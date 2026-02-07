defmodule PanicTda.Validations.TimestampOrder do
  use Ash.Resource.Validation

  @impl true
  def init(opts) do
    opts = Keyword.merge([start_field: :started_at, end_field: :completed_at], opts)
    {:ok, opts}
  end

  @impl true
  def validate(changeset, opts, _context) do
    started = Ash.Changeset.get_attribute(changeset, opts[:start_field])
    completed = Ash.Changeset.get_attribute(changeset, opts[:end_field])

    case {started, completed} do
      {nil, _} ->
        :ok

      {_, nil} ->
        :ok

      {s, c} ->
        if DateTime.compare(c, s) in [:gt, :eq] do
          :ok
        else
          {:error,
           field: opts[:end_field],
           message: "must be at or after %{start_field}",
           vars: %{start_field: opts[:start_field]}}
        end
    end
  end
end
