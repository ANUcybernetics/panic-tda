defmodule PanicTda.RealModelsTest do
  use ExUnit.Case, async: false

  @moduletag :gpu
  @moduletag timeout: 600_000

  require Ash.Query

  alias PanicTda.Engine
  alias PanicTda.Models.{GenAI, Embeddings, PythonInterpreter}

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)

    {:ok, interpreter} = PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    on_exit(fn ->
      if Process.alive?(interpreter), do: GenServer.stop(interpreter)
    end)

    %{env: env}
  end

  describe "real GenAI models" do
    test "SDXLTurbo generates valid AVIF image", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "SDXLTurbo", "A cat sitting on a mat")

      assert is_binary(image)
      assert byte_size(image) > 100
      assert <<_::binary-size(4), "ftyp", _::binary>> = image
    end

    test "Moondream generates text caption from image", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "SDXLTurbo", "A red apple on a table")
      {:ok, caption} = GenAI.invoke(env, "Moondream", image)

      assert is_binary(caption)
      assert String.length(caption) > 0
    end
  end

  describe "real Embedding models" do
    test "STSBMpnet generates 768-dim float32 text embeddings", %{env: env} do
      {:ok, [emb1, emb2]} = Embeddings.embed(env, "STSBMpnet", ["hello world", "test sentence"])

      assert is_binary(emb1)
      assert is_binary(emb2)
      assert byte_size(emb1) == 768 * 4
      assert byte_size(emb2) == 768 * 4
    end

    test "NomicVision generates 768-dim float32 image embeddings", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "SDXLTurbo", "A blue sky")
      {:ok, [emb]} = Embeddings.embed(env, "NomicVision", [image])

      assert is_binary(emb)
      assert byte_size(emb) == 768 * 4
    end
  end

  describe "end-to-end pipeline with real models" do
    test "full experiment with SDXLTurbo + Moondream + STSBMpnet" do
      {:ok, experiment} =
        PanicTda.Experiment
        |> Ash.Changeset.for_create(:create, %{
          networks: [["SDXLTurbo", "Moondream"]],
          prompts: ["A peaceful garden"],
          embedding_models: ["STSBMpnet"],
          max_length: 4
        })
        |> Ash.create()

      {:ok, completed} = Engine.perform_experiment(experiment.id)

      assert completed.started_at != nil
      assert completed.completed_at != nil

      completed = Ash.load!(completed, runs: [:invocations])
      run = hd(completed.runs)
      assert length(run.invocations) == 4

      [inv0, inv1, inv2, inv3] = run.invocations
      assert inv0.type == :image
      assert inv0.model == "SDXLTurbo"
      assert inv0.output_image != nil

      assert inv1.type == :text
      assert inv1.model == "Moondream"
      assert inv1.output_text != nil

      assert inv2.type == :image
      assert inv2.model == "SDXLTurbo"

      assert inv3.type == :text
      assert inv3.model == "Moondream"

      embeddings =
        PanicTda.Embedding
        |> Ash.Query.filter(embedding_model == ^"STSBMpnet")
        |> Ash.read!()

      assert length(embeddings) == 2

      Enum.each(embeddings, fn emb ->
        assert %Nx.Tensor{} = emb.vector
        assert Nx.shape(emb.vector) == {768}
      end)

      pds = Ash.read!(PanicTda.PersistenceDiagram)
      assert length(pds) == 1
      pd = hd(pds)
      assert pd.embedding_model == "STSBMpnet"
      assert pd.diagram_data != nil
    end
  end

  describe "all model combinations smoke test" do
    @real_text_embedding_models ~w(STSBMpnet STSBRoberta STSBDistilRoberta Nomic JinaClip)
    @real_image_embedding_models ~w(NomicVision JinaClipVision)

    for t2i <- ~w(SDXLTurbo FluxDev FluxSchnell),
        i2t <- ~w(Moondream BLIP2) do
      @tag timeout: 600_000
      test "pipeline: #{t2i} + #{i2t} with all text embedding models" do
        t2i = unquote(t2i)
        i2t = unquote(i2t)

        experiment =
          PanicTda.create_experiment!(%{
            networks: [[t2i, i2t]],
            prompts: ["a red apple"],
            embedding_models: @real_text_embedding_models,
            max_length: 4
          })

        {:ok, completed} = Engine.perform_experiment(experiment.id)

        assert completed.completed_at != nil

        completed = Ash.load!(completed, runs: [:invocations])
        run = hd(completed.runs)
        assert length(run.invocations) == 4
        assert Enum.at(run.invocations, 0).model == t2i
        assert Enum.at(run.invocations, 1).model == i2t

        for model <- @real_text_embedding_models do
          embeddings =
            PanicTda.list_embeddings!(query: [filter: [embedding_model: model]])

          assert length(embeddings) == 2,
                 "expected 2 embeddings for #{model}, got #{length(embeddings)}"
        end

        pds = PanicTda.list_persistence_diagrams!()
        assert length(pds) == length(@real_text_embedding_models)
      end
    end

    for t2i <- ~w(SDXLTurbo FluxDev FluxSchnell),
        i2t <- ~w(Moondream BLIP2) do
      @tag timeout: 600_000
      test "pipeline: #{t2i} + #{i2t} with all image embedding models" do
        t2i = unquote(t2i)
        i2t = unquote(i2t)

        experiment =
          PanicTda.create_experiment!(%{
            networks: [[t2i, i2t]],
            prompts: ["a red apple"],
            embedding_models: @real_image_embedding_models,
            max_length: 4
          })

        {:ok, completed} = Engine.perform_experiment(experiment.id)

        assert completed.completed_at != nil

        completed = Ash.load!(completed, runs: [:invocations])
        run = hd(completed.runs)
        assert length(run.invocations) == 4

        for model <- @real_image_embedding_models do
          embeddings =
            PanicTda.list_embeddings!(query: [filter: [embedding_model: model]])

          assert length(embeddings) == 2,
                 "expected 2 embeddings for #{model}, got #{length(embeddings)}"
        end

        pds = PanicTda.list_persistence_diagrams!()
        assert length(pds) == length(@real_image_embedding_models)
      end
    end
  end
end
