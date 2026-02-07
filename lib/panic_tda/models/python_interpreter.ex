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
    requires-python = ">=3.12,<3.13"
    dependencies = [
      "pillow>=10.0",
      "numpy>=1.26",
      "giotto-ph>=0.2.4",
      "persim>=0.3.8",
      "torch>=2.7",
      "diffusers>=0.34,<0.35",
      "transformers>=4.54,<5.0",
      "sentence-transformers>=5.0,<6.0",
      "accelerate>=1.9",
      "pyvips>=2.2",
      "sentencepiece>=0.2",
      "einops>=0.8",
      "timm>=1.0",
      "scikit-learn>=1.6"
    ]
    """
end
