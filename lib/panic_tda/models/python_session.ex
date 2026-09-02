defmodule PanicTda.Models.PythonSession do
  @moduledoc """
  A Python interpreter and its Snex env held as one restartable unit.

  Recovering from a CUDA device-side assert needs a fresh OS process, not just
  a fresh call: the assert poisons the process's CUDA context and every later
  CUDA call returns the same sticky error. A session therefore hands out its
  current env rather than being one, so `restart/1` can swap the interpreter
  underneath long-running callers without threading a new env back through
  them.

  `wrap/1` builds a session around an env someone else owns (tests, callers
  that made their own interpreter). Such a session cannot restart, and
  stopping it leaves the caller's interpreter alone.
  """

  use Agent

  alias PanicTda.Models.PythonInterpreter

  # booting an interpreter is well past the Agent default of 5s
  @restart_timeout 600_000

  @doc """
  Start an interpreter and wrap it in a session.

  `on_start` is applied to every env the session produces, including after a
  restart, and is where per-experiment Python state (the image-to-text
  generation ceiling, say) gets re-applied to the fresh process.
  """
  def start(on_start \\ fn _env -> :ok end) do
    {:ok, interpreter, env} = boot(on_start)
    Agent.start_link(fn -> %{interpreter: interpreter, env: env, on_start: on_start} end)
  end

  @doc "Wrap an env owned by the caller. Not restartable."
  def wrap(env, on_start \\ fn _env -> :ok end) do
    with :ok <- on_start.(env) do
      Agent.start_link(fn -> %{interpreter: nil, env: env, on_start: on_start} end)
    end
  end

  @doc "The session's current env."
  def env(session), do: Agent.get(session, & &1.env)

  @doc "Whether this session owns its interpreter and so can replace it."
  def restartable?(session), do: Agent.get(session, &(&1.interpreter != nil))

  @doc """
  Replace the Python process with a fresh one.

  Models are not reloaded here; the next invocation loads what it needs
  through `PythonBridge.swap_model_to_gpu/2`.
  """
  def restart(session) do
    Agent.get_and_update(
      session,
      fn
        %{interpreter: nil} = state ->
          {{:error, :not_restartable}, state}

        %{interpreter: interpreter, on_start: on_start} = state ->
          stop_interpreter(interpreter)

          case boot(on_start) do
            {:ok, new_interpreter, new_env} ->
              {:ok, %{state | interpreter: new_interpreter, env: new_env}}

            {:error, reason} ->
              {{:error, reason}, %{state | interpreter: nil}}
          end
      end,
      @restart_timeout
    )
  end

  @doc "Stop the session, and its interpreter if it owns one."
  def stop(session) do
    interpreter = Agent.get(session, & &1.interpreter)
    if interpreter, do: stop_interpreter(interpreter)
    Agent.stop(session)
  end

  defp boot(on_start) do
    with {:ok, interpreter} <- PythonInterpreter.start_link(),
         {:ok, env} <- Snex.make_env(interpreter),
         :ok <- on_start.(env) do
      {:ok, interpreter, env}
    end
  end

  defp stop_interpreter(interpreter) do
    if Process.alive?(interpreter), do: GenServer.stop(interpreter)
  end
end
