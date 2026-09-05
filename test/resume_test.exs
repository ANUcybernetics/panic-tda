defmodule PanicTda.ResumeTest do
  use ExUnit.Case

  require Ash.Query

  alias PanicTda.Engine
  alias PanicTda.Engine.{RunExecutor, EmbeddingsStage}
  alias PanicTda.Models.{PythonInterpreter, PythonSession}

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    :ok
  end

  defp create_started_experiment(overrides \\ %{}) do
    defaults = %{
      networks: [["DummyT2I", "DummyI2T"]],
      prompts: ["test prompt"],
      embedding_models: ["DummyText"],
      max_length: 4
    }

    experiment = PanicTda.create_experiment!(Map.merge(defaults, overrides))
    PanicTda.start_experiment!(experiment)
  end

  defp with_python(_context) do
    {:ok, interpreter} = PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)
    {:ok, session} = PythonSession.wrap(env)

    on_exit(fn ->
      if Process.alive?(interpreter), do: GenServer.stop(interpreter)
    end)

    %{env: env, session: session}
  end

  describe "resume_experiment/1" do
    test "rejects experiment that has not been started" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          prompts: ["test"],
          embedding_models: ["DummyText"],
          max_length: 4
        })

      assert {:error, :not_started} = Engine.resume_experiment(experiment.id)
    end

    test "rejects completed experiment" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          prompts: ["test"],
          embedding_models: ["DummyText"],
          max_length: 4
        })

      {:ok, completed} = Engine.perform_experiment(experiment.id)
      assert {:error, :already_completed} = Engine.resume_experiment(completed.id)
    end

    test "force: true reopens a completed experiment and re-runs resume" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          prompts: ["test"],
          embedding_models: ["DummyText"],
          max_length: 4
        })

      {:ok, completed} = Engine.perform_experiment(experiment.id)
      assert completed.completed_at != nil

      assert {:ok, resumed} = Engine.resume_experiment(completed.id, force: true)
      assert resumed.completed_at != nil
      assert DateTime.compare(resumed.completed_at, completed.completed_at) == :gt
    end
  end

  describe "partial run resume" do
    setup :with_python

    test "completes a partial run", %{session: session} do
      experiment = create_started_experiment(%{max_length: 4})
      [run] = Engine.init_runs(experiment)

      RunExecutor.execute(session, %{run | max_length: 2})

      invocations_before =
        PanicTda.Invocation
        |> Ash.Query.filter(run_id == ^run.id)
        |> Ash.count!()

      assert invocations_before == 2

      :ok = RunExecutor.resume(session, run)

      invocations_after =
        PanicTda.Invocation
        |> Ash.Query.filter(run_id == ^run.id)
        |> Ash.count!()

      assert invocations_after == 4
    end

    test "skips already-complete runs", %{session: session} do
      experiment = create_started_experiment(%{max_length: 4})
      [run] = Engine.init_runs(experiment)

      RunExecutor.execute(session, run)

      invocations_before =
        PanicTda.Invocation
        |> Ash.Query.filter(run_id == ^run.id)
        |> Ash.count!()

      assert invocations_before == 4

      :ok = RunExecutor.resume(session, run)

      invocations_after =
        PanicTda.Invocation
        |> Ash.Query.filter(run_id == ^run.id)
        |> Ash.count!()

      assert invocations_after == 4
    end
  end

  describe "ragged batch resume" do
    setup :with_python

    test "each run continues from its own last step", %{session: session} do
      # Invocations within a batch step are created one at a time, so a crash
      # mid-step leaves the runs at different sequence numbers.
      experiment = create_started_experiment(%{max_length: 6, num_runs: 3})
      [ahead, behind, untouched] = Engine.init_runs(experiment)

      RunExecutor.execute(session, %{ahead | max_length: 4})
      RunExecutor.execute(session, %{behind | max_length: 1})

      last_before =
        Map.new([ahead, behind], fn run -> {run.id, last_invocation(run)} end)

      :ok = RunExecutor.resume_batch(session, [ahead, behind, untouched])

      for run <- [ahead, behind, untouched] do
        invocations =
          PanicTda.Invocation
          |> Ash.Query.filter(run_id == ^run.id)
          |> Ash.Query.sort(sequence_number: :asc)
          |> Ash.read!()

        assert Enum.map(invocations, & &1.sequence_number) == Enum.to_list(0..5)

        # Every step's input is the previous step of the same run, including
        # the first resumed step.
        invocations
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.each(fn [prev, next] ->
          assert next.input_invocation_id == prev.id
        end)
      end

      # The runs that had already progressed were not re-run from the minimum.
      for run <- [ahead, behind] do
        %{id: id, sequence_number: seq} = last_before[run.id]
        assert %{id: ^id} = invocation_at(run, seq)
      end
    end
  end

  defp last_invocation(run) do
    PanicTda.Invocation
    |> Ash.Query.filter(run_id == ^run.id)
    |> Ash.Query.sort(sequence_number: :desc)
    |> Ash.Query.limit(1)
    |> Ash.read_one!()
  end

  defp invocation_at(run, seq) do
    PanicTda.Invocation
    |> Ash.Query.filter(run_id == ^run.id and sequence_number == ^seq)
    |> Ash.read_one!()
  end

  describe "find_or_create_runs/1" do
    test "returns existing runs without creating duplicates" do
      experiment = create_started_experiment()
      runs = Engine.init_runs(experiment)
      assert length(runs) == 1

      found_runs = Engine.find_or_create_runs(experiment)
      assert length(found_runs) == 1
      assert hd(found_runs).id == hd(runs).id
    end

    test "creates missing runs" do
      experiment = create_started_experiment(%{num_runs: 2})
      [_run1] = Engine.init_runs(%{experiment | num_runs: 1})

      found_runs = Engine.find_or_create_runs(experiment)
      assert length(found_runs) == 2
    end
  end

  describe "embeddings resume" do
    setup :with_python

    test "fills in missing embeddings", %{env: env, session: session} do
      experiment = create_started_experiment(%{max_length: 4})
      [run] = Engine.init_runs(experiment)
      RunExecutor.execute(session, run)

      invocations =
        run
        |> Ash.load!(:invocations)
        |> Map.get(:invocations, [])

      text_invocations = Enum.filter(invocations, &(&1.type == :text))
      first_text = hd(text_invocations)

      EmbeddingsStage.compute_for_invocations(env, [first_text], "DummyText")

      embeddings_before =
        PanicTda.Embedding
        |> Ash.Query.filter(invocation.run_id == ^run.id and embedding_model == "DummyText")
        |> Ash.count!()

      assert embeddings_before == 1

      EmbeddingsStage.resume(env, run, ["DummyText"])

      embeddings_after =
        PanicTda.Embedding
        |> Ash.Query.filter(invocation.run_id == ^run.id and embedding_model == "DummyText")
        |> Ash.count!()

      assert embeddings_after == 2
    end
  end

  describe "full resume pipeline" do
    test "resumes a partially completed experiment" do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["DummyT2I", "DummyI2T"]],
          prompts: ["test prompt"],
          embedding_models: ["DummyText"],
          max_length: 4
        })

      {:ok, interpreter} = PythonInterpreter.start_link()
      {:ok, env} = Snex.make_env(interpreter)
      {:ok, session} = PythonSession.wrap(env)

      experiment = PanicTda.start_experiment!(experiment)
      [run] = Engine.init_runs(experiment)

      RunExecutor.execute(session, %{run | max_length: 2})

      GenServer.stop(interpreter)

      {:ok, completed} = Engine.resume_experiment(experiment.id)
      assert completed.completed_at != nil

      invocation_count =
        PanicTda.Invocation
        |> Ash.Query.filter(run_id == ^run.id)
        |> Ash.count!()

      assert invocation_count == 4

      embedding_count =
        PanicTda.Embedding
        |> Ash.Query.filter(invocation.run_id == ^run.id and embedding_model == "DummyText")
        |> Ash.count!()

      assert embedding_count == 2

      pd_count =
        PanicTda.PersistenceDiagram
        |> Ash.Query.filter(run_id == ^run.id)
        |> Ash.count!()

      assert pd_count == 1
    end
  end
end
