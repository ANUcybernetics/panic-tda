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
      "pillow>=11.0",
      "numpy>=1.26",
      "altair>=5.4",
      "vl-convert-python>=1.6",
      "giotto-ph>=0.2.4",
      "persim>=0.3.8",
      "torch>=2.13,<3.0",
      # 0.40 is the first release carrying the flux2/z_image/glm_image/
      # hunyuan_image pipelines, so this no longer has to track git
      "diffusers>=0.40,<0.41",
      "transformers>=5.16,<6.0",
      "sentence-transformers>=6.0,<7.0",
      "accelerate>=1.9",
      "pyvips>=2.2",
      "sentencepiece>=0.2",
      "einops>=0.8",
      "timm>=1.0",
      "scikit-learn>=1.6",
      "protobuf>=5.0",
      "qwen-vl-utils>=0.0.8",
      "peft>=0.15",
      "bitsandbytes>=0.45",
      "backoff>=2.2",
      "evoc>=0.1"
    ]
    """

  @stop_timeout 30_000

  @doc """
  Stops an interpreter, killing it if it will not go quietly.

  Snex's own `stop/1` waits forever, and after a long clustering run the
  interpreter does not always terminate --- `mix cluster.recompute` printed
  "Done." and then sat there for ninety minutes. A batch job that has finished
  its work must exit.
  """
  def stop_or_kill(interpreter, timeout \\ @stop_timeout) do
    if Process.alive?(interpreter) do
      try do
        GenServer.stop(interpreter, :normal, timeout)
      catch
        :exit, _ -> Process.exit(interpreter, :kill)
      end
    end

    :ok
  end
end
