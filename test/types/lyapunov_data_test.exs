defmodule PanicTda.Types.LyapunovDataTest do
  use ExUnit.Case, async: true

  alias PanicTda.Types.LyapunovData

  @sample %{
    exponent: 0.003684,
    r_squared: 0.573,
    divergence_curve: [1.0, 1.01, 1.02, 1.04],
    num_pairs: 3,
    num_timesteps: 4
  }

  test "round-trips maps via cast_input/cast_stored" do
    {:ok, bin} = LyapunovData.cast_input(@sample, [])
    assert is_binary(bin)

    {:ok, restored} = LyapunovData.cast_stored(bin, [])
    assert restored == @sample
  end

  test "cast_stored decodes terms whose atoms are newly seen at decode time" do
    # Regression: cast_stored used to pass [:safe] to binary_to_term, which
    # fails on a fresh VM when the atom keys in the map aren't already in
    # the atom table. The writer path (cast_input) is our own code, so the
    # stored bytes are trusted and the safe flag is inappropriate here.
    bin = :erlang.term_to_binary(@sample, [:compressed])
    assert {:ok, decoded} = LyapunovData.cast_stored(bin, [])
    assert decoded == @sample
  end

  test "cast_stored passes nil through" do
    assert {:ok, nil} = LyapunovData.cast_stored(nil, [])
  end

  test "dump_to_native round-trips via cast_stored" do
    {:ok, bin} = LyapunovData.dump_to_native(@sample, [])
    {:ok, restored} = LyapunovData.cast_stored(bin, [])
    assert restored == @sample
  end
end
