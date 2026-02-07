defmodule PanicTda.Types.Image do
  @moduledoc """
  Stores images as AVIF binary for SQLite compatibility.
  """
  use Ash.Type

  @impl true
  def storage_type(_constraints), do: :binary

  @impl true
  def cast_input(nil, _constraints), do: {:ok, nil}
  def cast_input(binary, _constraints) when is_binary(binary), do: {:ok, binary}
  def cast_input(_, _constraints), do: :error

  @impl true
  def cast_stored(nil, _constraints), do: {:ok, nil}
  def cast_stored(binary, _constraints) when is_binary(binary), do: {:ok, binary}
  def cast_stored(_, _constraints), do: :error

  @impl true
  def dump_to_native(nil, _constraints), do: {:ok, nil}
  def dump_to_native(binary, _constraints) when is_binary(binary), do: {:ok, binary}
  def dump_to_native(_, _constraints), do: :error
end
