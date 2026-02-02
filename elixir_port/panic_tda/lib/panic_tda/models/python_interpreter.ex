defmodule PanicTda.Models.PythonInterpreter do
  @moduledoc """
  Snex interpreter for calling Python models.
  Uses inline Python implementations of dummy models for testing.
  """
  use Snex.Interpreter,
    pyproject_toml: """
    [project]
    name = "panic-tda-elixir-bridge"
    version = "0.0.1"
    requires-python = ">=3.12,<3.15"
    dependencies = [
      "pillow>=10.0",
      "numpy>=1.26"
    ]
    """
end
