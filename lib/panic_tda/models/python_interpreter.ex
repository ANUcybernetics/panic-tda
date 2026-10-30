defmodule PanicTda.Models.PythonInterpreter do
  @moduledoc """
  Snex interpreter for calling Python models.

  The venv is `priv/python/pyproject.toml`, resolved by the tracked
  `priv/python/uv.lock` beside it; run `uv lock` there after changing it.
  """
  use Snex.Interpreter, project_path: "priv/python"

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
