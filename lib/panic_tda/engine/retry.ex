defmodule PanicTda.Engine.Retry do
  @moduledoc """
  Bounded retry around a single model-invocation step.

  A stochastic CUDA fault — GLM-Image's AR prior sampling an out-of-range
  token id, roughly one batched call in a few thousand — used to kill a
  multi-week experiment outright, because the Snex error propagated all the
  way out of the mix task and the GPU then idled until someone noticed.
  Retrying the step is enough to survive it: the failed step wrote nothing, so
  the run picks up exactly where it was, with fresh noise.

  The first retry reuses the Python process, which suffices for a fault that
  leaves the CUDA context intact. Later retries restart the interpreter,
  because a device-side assert poisons the context and every subsequent CUDA
  call in that process returns the same sticky error. The retried invocation
  reloads the model on demand via `PythonBridge.swap_model_to_gpu/2`.

  Retries are bounded so a deterministic failure still surfaces promptly, and
  each one is logged with model, step and attempt so they can be counted after
  the fact.
  """

  require Logger

  alias PanicTda.Models.PythonSession

  @max_attempts 3

  @doc """
  Run `fun` against the session's current env, retrying transient failures.

  `fun` takes an env and returns `{:ok, result}` or `{:error, reason}`; raised
  exceptions and exits (a Snex call timing out, say) are treated as failures
  too. `label` identifies the step in the logs, e.g. `"GLMImage step 24"`.

  Returns `{:ok, result}`, or `{:error, reason}` once the attempts are spent.
  """
  def with_retry(label, session, fun) when is_function(fun, 1) do
    attempt(label, session, fun, 1)
  end

  defp attempt(label, session, fun, n) do
    case invoke(fun, PythonSession.env(session)) do
      {:ok, result} ->
        if n > 1, do: Logger.warning("[retry] #{label}: succeeded on attempt #{n}")
        {:ok, result}

      {:error, reason} when n >= @max_attempts ->
        Logger.error(
          "[retry] #{label}: failed on attempt #{n} of #{@max_attempts}, giving up: #{summarise(reason)}"
        )

        {:error, reason}

      {:error, reason} ->
        Logger.warning(
          "[retry] #{label}: attempt #{n} of #{@max_attempts} failed: #{summarise(reason)}"
        )

        recover(label, session, n)
        attempt(label, session, fun, n + 1)
    end
  end

  # first retry reuses the process; after that assume the CUDA context is gone
  defp recover(_label, _session, 1), do: Process.sleep(backoff_ms())

  defp recover(label, session, _n) do
    Process.sleep(backoff_ms())

    case PythonSession.restart(session) do
      :ok ->
        Logger.warning("[retry] #{label}: restarted the Python interpreter")

      {:error, :not_restartable} ->
        :ok

      {:error, reason} ->
        Logger.error("[retry] #{label}: could not restart the interpreter: #{inspect(reason)}")
    end
  end

  defp invoke(fun, env) do
    case fun.(env) do
      {:ok, result} -> {:ok, result}
      {:error, reason} -> {:error, reason}
      other -> {:error, {:unexpected_result, other}}
    end
  rescue
    exception -> {:error, exception}
  catch
    :exit, reason -> {:error, {:exit, reason}}
  end

  defp backoff_ms, do: Application.get_env(:panic_tda, :retry_backoff_ms, 5_000)

  defp summarise(%{__exception__: true} = exception), do: Exception.message(exception)

  defp summarise(reason) do
    reason |> inspect(limit: 5, printable_limit: 400) |> String.slice(0, 400)
  end
end
