defmodule PanicTda.Models.PythonBridge do
  @moduledoc """
  Python bridge for real ML model execution via Snex.
  Manages one-time setup and lazy model loading into a persistent Python environment.
  """

  @load_timeout 600_000

  def ensure_setup(env) do
    case Snex.pyeval(env, "_panic_setup_done", %{}, returning: "_panic_setup_done") do
      {:ok, true} ->
        :ok

      _ ->
        priv_dir = :code.priv_dir(:panic_tda) |> to_string()
        python_dir = Path.join(priv_dir, "python")

        case Snex.pyeval(
               env,
               """
               import sys
               sys.path.insert(0, _priv_python_dir)
               import panic_models
               panic_models.setup()
               _panic_setup_done = True
               """,
               %{"_priv_python_dir" => python_dir},
               returning: "_panic_setup_done",
               timeout: @load_timeout
             ) do
          {:ok, true} -> :ok
          error -> error
        end
    end
  end

  def ensure_model_loaded(env, model_name) do
    case Snex.pyeval(
           env,
           "result = panic_models.is_model_loaded(model_name)",
           %{"model_name" => model_name},
           returning: "result"
         ) do
      {:ok, true} ->
        :ok

      {:ok, false} ->
        case Snex.pyeval(
               env,
               "panic_models.load_model(model_name)",
               %{"model_name" => model_name},
               returning: "True",
               timeout: @load_timeout
             ) do
          {:ok, _} -> :ok
          error -> error
        end

      error ->
        error
    end
  end

  @swap_timeout 120_000

  def swap_model_to_gpu(env, model_name) do
    with :ok <- ensure_setup(env),
         :ok <- ensure_model_loaded(env, model_name) do
      case Snex.pyeval(
             env,
             "panic_models.swap_to_gpu(model_name)",
             %{"model_name" => model_name},
             returning: "True",
             timeout: @swap_timeout
           ) do
        {:ok, _} -> :ok
        error -> error
      end
    end
  end

  def swap_model_to_cpu(env, model_name) do
    case Snex.pyeval(
           env,
           """
           if "panic_models" in dir():
               panic_models.swap_to_cpu(model_name)
           """,
           %{"model_name" => model_name},
           returning: "True",
           timeout: @swap_timeout
         ) do
      {:ok, _} -> :ok
      error -> error
    end
  end

  @unload_timeout 30_000

  def unload_model(env, model_name) do
    case Snex.pyeval(
           env,
           """
           if "panic_models" in dir():
               panic_models.unload_model(model_name)
           """,
           %{"model_name" => model_name},
           returning: "True",
           timeout: @unload_timeout
         ) do
      {:ok, _} -> :ok
      error -> error
    end
  end

  def unload_all_models(env) do
    case Snex.pyeval(
           env,
           """
           if "panic_models" in dir():
               panic_models.unload_all_models()
           """,
           %{},
           returning: "True",
           timeout: @unload_timeout
         ) do
      {:ok, _} -> :ok
      error -> error
    end
  end
end
