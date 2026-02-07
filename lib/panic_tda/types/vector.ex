defmodule PanicTda.Types.Vector do
  @moduledoc """
  Stores float32 vectors as binary. Works with Nx tensors or lists.
  Compatible with Python's numpy float32 format.
  """
  use Ash.Type

  @impl true
  def storage_type(_constraints), do: :binary

  @impl true
  def cast_input(nil, _constraints), do: {:ok, nil}

  def cast_input(%Nx.Tensor{} = tensor, _constraints) do
    {:ok, Nx.to_binary(tensor)}
  end

  def cast_input(list, _constraints) when is_list(list) do
    binary = list |> Nx.tensor(type: :f32) |> Nx.to_binary()
    {:ok, binary}
  end

  def cast_input(binary, _constraints) when is_binary(binary), do: {:ok, binary}

  def cast_input(_, _constraints), do: :error

  @impl true
  def cast_stored(nil, _constraints), do: {:ok, nil}

  def cast_stored(binary, _constraints) when is_binary(binary) do
    {:ok, Nx.from_binary(binary, :f32)}
  end

  def cast_stored(_, _constraints), do: :error

  @impl true
  def dump_to_native(nil, _constraints), do: {:ok, nil}

  def dump_to_native(%Nx.Tensor{} = tensor, _constraints) do
    {:ok, Nx.to_binary(tensor)}
  end

  def dump_to_native(binary, _constraints) when is_binary(binary), do: {:ok, binary}

  def dump_to_native(_, _constraints), do: :error
end
