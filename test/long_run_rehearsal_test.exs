defmodule PanicTda.LongRunRehearsalTest do
  @moduledoc """
  A rehearsal for a run longer than any before it, kept out of the suite:

      mix test --only rehearsal test/long_run_rehearsal_test.exs
      REHEARSAL_STATES=2000 mix test --only rehearsal test/long_run_rehearsal_test.exs

  One run of that many text states is stored as the pipeline stores it, with
  images of the panel's size and captions of Gemma4's length, and then put
  through the stages that follow a cell by the calls the engine makes. Those
  stages first meet a run's full length when its cell finishes, days into the
  GPU time; this meets it in a minute on CPU.

  The embedding model is the dummy one, so it says nothing about Qwen3Embed's
  throughput. Diagram time climbs steeply with the length (`analysis/pd_cost.py`).
  """
  use ExUnit.Case

  require Ash.Query

  alias PanicTda.Engine.{EmbeddingsStage, PdStage}
  alias PanicTda.Models.PythonInterpreter

  @moduletag :rehearsal
  @moduletag timeout: :infinity

  # the panel's images average 135 kB as stored
  @image_bytes 135_000
  @network ["DummyT2I", "DummyI2T"]

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    {:ok, interpreter} = PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    on_exit(fn ->
      if Process.alive?(interpreter), do: GenServer.stop(interpreter)
    end)

    %{env: env}
  end

  test "the stages that follow a cell, on one run at full length", %{env: env} do
    states = "REHEARSAL_STATES" |> System.get_env("1000") |> String.to_integer()

    experiment =
      PanicTda.create_experiment!(%{
        networks: [@network],
        prompts: ["a red apple on a wooden table"],
        embedding_models: ["DummyText"],
        max_length: 2 * states
      })

    run =
      PanicTda.create_run!(%{
        network: @network,
        run_number: 0,
        max_length: 2 * states,
        initial_prompt: "a red apple on a wooden table",
        experiment_id: experiment.id
      })

    now = DateTime.utc_now()
    # about 214 words, Gemma4's median
    filler = String.duplicate("a long description of the scene ", 36)

    {store_us, _last} =
      :timer.tc(fn ->
        Enum.reduce(0..(2 * states - 1), nil, fn seq, previous ->
          output =
            if rem(seq, 2) == 0 do
              %{
                type: :image,
                model: "DummyT2I",
                seed: seq,
                output_image: :crypto.strong_rand_bytes(@image_bytes)
              }
            else
              %{type: :text, model: "DummyI2T", output_text: "caption #{seq}: #{filler}"}
            end

          invocation =
            PanicTda.create_invocation!(
              Map.merge(output, %{
                sequence_number: seq,
                started_at: now,
                completed_at: now,
                run_id: run.id,
                input_invocation_id: previous
              })
            )

          invocation.id
        end)
      end)

    {embed_us, :ok} = :timer.tc(fn -> EmbeddingsStage.compute(env, run, ["DummyText"]) end)
    {resume_us, :ok} = :timer.tc(fn -> EmbeddingsStage.resume(env, run, ["DummyText"]) end)

    embeddings =
      PanicTda.Embedding
      |> Ash.Query.filter(invocation.run_id == ^run.id)
      |> Ash.read!()

    assert length(embeddings) == states

    # The dummy model's vectors have no structure, and a diagram over a
    # thousand such points does not finish. Swap in a cloud shaped like a
    # run's: a random walk under fresh noise at every state.
    key = Nx.Random.key(0)
    {steps, key} = Nx.Random.normal(key, shape: {states, 256}, type: :f32)
    {noise, _key} = Nx.Random.normal(key, shape: {states, 256}, type: :f32)
    cloud = steps |> Nx.cumulative_sum(axis: 0) |> Nx.add(noise)

    replacements =
      embeddings
      |> Enum.with_index()
      |> Enum.map(fn {embedding, row} ->
        %{
          embedding_model: "DummyText",
          vector: Nx.to_binary(cloud[row]),
          started_at: now,
          completed_at: now,
          invocation_id: embedding.invocation_id
        }
      end)

    Ash.bulk_destroy!(embeddings, :destroy, %{})

    Ash.bulk_create!(replacements, PanicTda.Embedding, :create,
      return_errors?: true,
      stop_on_error?: true
    )

    {diagram_us, :ok} = :timer.tc(fn -> PdStage.compute(env, run, ["DummyText"]) end)
    :ok = PdStage.resume(env, run, ["DummyText"])

    [diagram] =
      PanicTda.PersistenceDiagram
      |> Ash.Query.filter(run_id == ^run.id)
      |> Ash.read!()

    assert [components, _loops, _voids] = diagram.diagram_data.dgms
    assert length(components) == states

    megabytes = states * @image_bytes / 1.0e6
    seconds = fn us -> Float.round(us / 1.0e6, 1) end

    IO.puts("""

    one run of #{states} text states (#{2 * states} invocations, #{round(megabytes)} MB of images)
      stored in         #{seconds.(store_us)} s
      embedding stage   #{seconds.(embed_us)} s (dummy model)
      embeddings resume #{seconds.(resume_us)} s
      diagram stage     #{seconds.(diagram_us)} s
    """)
  end
end
