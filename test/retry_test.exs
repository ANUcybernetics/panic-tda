defmodule PanicTda.RetryTest do
  use ExUnit.Case

  import ExUnit.CaptureLog

  alias PanicTda.Engine.Retry
  alias PanicTda.Models.{PythonInterpreter, PythonSession}

  # A fault injector: fails the first `n` calls, then succeeds. Standing in for
  # the stochastic CUDA error that used to kill a whole experiment run.
  defp flaky(n) do
    {:ok, counter} = Agent.start_link(fn -> 0 end)

    fn _env ->
      calls = Agent.get_and_update(counter, &{&1 + 1, &1 + 1})
      if calls <= n, do: {:error, :cuda_device_side_assert}, else: {:ok, "image #{calls}"}
    end
  end

  defp static_session do
    {:ok, session} = PythonSession.wrap(:fake_env)
    on_exit(fn -> if Process.alive?(session), do: PythonSession.stop(session) end)
    session
  end

  describe "with_retry/3" do
    test "passes the session's current env through to the step" do
      session = static_session()
      assert {:ok, :fake_env} = Retry.with_retry("step", session, fn env -> {:ok, env} end)
    end

    test "a single-shot failure is retried and the step succeeds" do
      session = static_session()
      fun = flaky(1)

      log =
        capture_log(fn ->
          assert {:ok, "image 2"} = Retry.with_retry("GLMImage step 24", session, fun)
        end)

      assert log =~ "GLMImage step 24: attempt 1 of 3 failed"
      assert log =~ "cuda_device_side_assert"
      assert log =~ "succeeded on attempt 2"
    end

    test "a persistent failure aborts after bounded retries" do
      session = static_session()
      fun = flaky(1_000)

      log =
        capture_log(fn ->
          assert {:error, :cuda_device_side_assert} = Retry.with_retry("step", session, fun)
        end)

      assert log =~ "attempt 3 of 3, giving up"
    end

    test "a step that raises is retried like any other failure" do
      session = static_session()
      {:ok, counter} = Agent.start_link(fn -> 0 end)

      fun = fn _env ->
        calls = Agent.get_and_update(counter, &{&1 + 1, &1 + 1})
        if calls == 1, do: raise("CUDA out of memory"), else: {:ok, :recovered}
      end

      log =
        capture_log(fn -> assert {:ok, :recovered} = Retry.with_retry("step", session, fun) end)

      assert log =~ "CUDA out of memory"
    end

    test "a step that exits is retried like any other failure" do
      session = static_session()
      {:ok, counter} = Agent.start_link(fn -> 0 end)

      fun = fn _env ->
        calls = Agent.get_and_update(counter, &{&1 + 1, &1 + 1})
        if calls == 1, do: exit(:timeout), else: {:ok, :recovered}
      end

      assert capture_log(fn ->
               assert {:ok, :recovered} = Retry.with_retry("step", session, fun)
             end) =~ "attempt 1 of 3 failed"
    end

    test "a wrapped session cannot restart, so it retries in place" do
      session = static_session()
      refute PythonSession.restartable?(session)

      log = capture_log(fn -> Retry.with_retry("step", session, flaky(1_000)) end)
      refute log =~ "restarted the Python interpreter"
    end
  end

  describe "recovery from a real CUDA fault" do
    # The whole retry design rests on the claim that a device-side assert
    # poisons the process's CUDA context, so that only a fresh interpreter can
    # recover. These tests establish that rather than assuming it. An
    # out-of-range embedding lookup is the same failure class as the GLM-Image
    # prior sampling an out-of-range token id, which is what motivated TASK-79.
    @poison """
    import torch
    _ = torch.nn.functional.embedding(
        torch.tensor([9999], device="cuda"), torch.zeros(8, 4, device="cuda")
    )
    torch.cuda.synchronize()
    return "no assert raised"
    """

    @cuda_probe "import torch\nreturn float(torch.ones(4, device='cuda').sum().item())"

    @tag :gpu
    @tag timeout: 600_000
    test "a device-side assert really does poison the context for the whole process" do
      {:ok, session} = PythonSession.start()
      on_exit(fn -> if Process.alive?(session), do: PythonSession.stop(session) end)
      env = PythonSession.env(session)

      assert {:ok, 4.0} = Snex.pyeval(env, @cuda_probe, %{}, timeout: 120_000)
      assert {:error, _} = Snex.pyeval(env, @poison, %{}, timeout: 120_000)

      # the premise: an unrelated, trivially valid CUDA op now fails too
      assert {:error, _} = Snex.pyeval(env, @cuda_probe, %{}, timeout: 120_000)
    end

    @tag :gpu
    @tag timeout: 600_000
    test "restarting the session recovers a poisoned context" do
      {:ok, session} = PythonSession.start()
      on_exit(fn -> if Process.alive?(session), do: PythonSession.stop(session) end)

      assert {:error, _} =
               Snex.pyeval(PythonSession.env(session), @poison, %{}, timeout: 120_000)

      :ok = PythonSession.restart(session)

      assert {:ok, 4.0} =
               Snex.pyeval(PythonSession.env(session), @cuda_probe, %{}, timeout: 120_000)
    end

    @tag :gpu
    @tag timeout: 600_000
    test "with_retry drives that recovery end to end" do
      {:ok, session} = PythonSession.start()
      on_exit(fn -> if Process.alive?(session), do: PythonSession.stop(session) end)
      {:ok, counter} = Agent.start_link(fn -> 0 end)

      # poison on the first call, then do ordinary CUDA work. Attempt 2 still
      # fails on the sticky context, which is what forces the restart before
      # attempt 3 --- the exact sequence a stochastic fault would produce.
      fun = fn env ->
        n = Agent.get_and_update(counter, &{&1 + 1, &1 + 1})
        script = if n == 1, do: @poison, else: @cuda_probe
        Snex.pyeval(env, script, %{}, timeout: 120_000)
      end

      log =
        capture_log(fn ->
          assert {:ok, 4.0} = Retry.with_retry("GLMImage step 24", session, fun)
        end)

      assert log =~ "restarted the Python interpreter"
      assert log =~ "succeeded on attempt 3"
      assert Agent.get(counter, & &1) == 3
    end
  end

  describe "session restart" do
    test "the second retry replaces the interpreter, and the step runs on the new one" do
      {:ok, session} = PythonSession.start()
      on_exit(fn -> if Process.alive?(session), do: PythonSession.stop(session) end)

      assert PythonSession.restartable?(session)
      first_env = PythonSession.env(session)

      # fails twice, so the first retry reuses the process and the second
      # restarts it; the third attempt must run against the replacement
      fun = flaky(2)

      log =
        capture_log(fn ->
          assert {:ok, "image 3"} = Retry.with_retry("DummyT2I step 0", session, fun)
        end)

      assert log =~ "restarted the Python interpreter"
      assert PythonSession.env(session) != first_env
    end

    test "a restarted interpreter can still run Python" do
      {:ok, session} = PythonSession.start()
      on_exit(fn -> if Process.alive?(session), do: PythonSession.stop(session) end)

      :ok = PythonSession.restart(session)

      assert {:ok, 3} = Snex.pyeval(PythonSession.env(session), "return 1 + 2", %{})
    end

    test "stopping a wrapped session leaves the caller's interpreter alone" do
      {:ok, interpreter} = PythonInterpreter.start_link()
      on_exit(fn -> if Process.alive?(interpreter), do: GenServer.stop(interpreter) end)
      {:ok, env} = Snex.make_env(interpreter)
      {:ok, session} = PythonSession.wrap(env)

      :ok = PythonSession.stop(session)

      assert Process.alive?(interpreter)
      assert {:ok, 2} = Snex.pyeval(env, "return 1 + 1", %{})
    end
  end
end
