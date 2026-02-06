defmodule PanicTda.Types.PersistenceDiagramData do
  @moduledoc """
  Stores persistence diagram data as compressed binary.
  Uses Erlang term format for serialization.
  """
  use Ash.Type

  @impl true
  def storage_type(_constraints), do: :binary

  @impl true
  def cast_input(nil, _constraints), do: {:ok, nil}

  def cast_input(%{} = data, _constraints) do
    {:ok, :erlang.term_to_binary(data, [:compressed])}
  end

  def cast_input(binary, _constraints) when is_binary(binary), do: {:ok, binary}

  def cast_input(_, _constraints), do: :error

  @impl true
  def cast_stored(nil, _constraints), do: {:ok, nil}

  def cast_stored(binary, _constraints) when is_binary(binary) do
    {:ok, :erlang.binary_to_term(binary, [:safe])}
  end

  def cast_stored(_, _constraints), do: :error

  @impl true
  def dump_to_native(nil, _constraints), do: {:ok, nil}

  def dump_to_native(%{} = data, _constraints) do
    {:ok, :erlang.term_to_binary(data, [:compressed])}
  end

  def dump_to_native(binary, _constraints) when is_binary(binary), do: {:ok, binary}

  def dump_to_native(_, _constraints), do: :error
end
