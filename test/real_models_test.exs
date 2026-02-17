defmodule PanicTda.RealModelsTest do
  use ExUnit.Case, async: false

  @moduletag :gpu
  @moduletag timeout: 600_000

  require Ash.Query

  alias PanicTda.Engine
  alias PanicTda.Models.{GenAI, Embeddings, PythonBridge, PythonInterpreter}

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)

    {:ok, interpreter} = PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    on_exit(fn ->
      try do
        if Process.alive?(interpreter), do: GenServer.stop(interpreter)
      catch
        :exit, _ -> :ok
      end
    end)

    %{env: env}
  end

  describe "real GenAI models" do
    test "SD35Medium generates valid AVIF image", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "SD35Medium", "A cat sitting on a mat")

      assert is_binary(image)
      assert byte_size(image) > 100
      assert <<_::binary-size(4), "ftyp", _::binary>> = image
    end

    test "Moondream generates text caption from image", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "SD35Medium", "A red apple on a table")
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
      {:ok, image} = GenAI.invoke(env, "SD35Medium", "A blue sky")
      {:ok, [emb]} = Embeddings.embed(env, "NomicVision", [image])

      assert is_binary(emb)
      assert byte_size(emb) == 768 * 4
    end
  end

  describe "end-to-end pipeline with real models" do
    test "full experiment with SD35Medium + Moondream + STSBMpnet" do
      {:ok, experiment} =
        PanicTda.Experiment
        |> Ash.Changeset.for_create(:create, %{
          networks: [["SD35Medium", "Moondream"]],
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
      assert inv0.model == "SD35Medium"
      assert inv0.output_image != nil

      assert inv1.type == :text
      assert inv1.model == "Moondream"
      assert inv1.output_text != nil

      assert inv2.type == :image
      assert inv2.model == "SD35Medium"

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

  describe "per-model T2I tests" do
    for t2i <- ~w(SD35Medium ZImageTurbo Flux2Klein Flux2Dev QwenImage HunyuanImage GLMImage) do
      @tag timeout: 600_000
      test "#{t2i} single invoke", %{env: env} do
        t2i = unquote(t2i)
        PythonBridge.unload_all_models(env)

        t0 = System.monotonic_time(:millisecond)
        {:ok, image} = GenAI.invoke(env, t2i, "a red apple")
        elapsed = System.monotonic_time(:millisecond) - t0
        IO.puts("#{t2i} single: #{elapsed}ms")

        assert is_binary(image)
        assert byte_size(image) > 100
        assert <<_::binary-size(4), "ftyp", _::binary>> = image
      end

      @tag timeout: 600_000
      test "#{t2i} batch invoke (3 prompts)", %{env: env} do
        t2i = unquote(t2i)
        PythonBridge.unload_all_models(env)

        prompts = ["a red apple", "a blue car", "a green tree"]
        t0 = System.monotonic_time(:millisecond)
        {:ok, images} = GenAI.invoke_batch(env, t2i, prompts)
        elapsed = System.monotonic_time(:millisecond) - t0
        IO.puts("#{t2i} batch(3): #{elapsed}ms total, #{div(elapsed, 3)}ms/image")

        assert length(images) == 3

        Enum.each(images, fn image ->
          assert is_binary(image)
          assert byte_size(image) > 100
          assert <<_::binary-size(4), "ftyp", _::binary>> = image
        end)
      end
    end
  end

  describe "per-model I2T tests" do
    for i2t <- ~w(Moondream Qwen25VL Gemma3n Pixtral LLaMA32Vision Phi4Vision) do
      @tag timeout: 600_000
      test "#{i2t} single invoke", %{env: env} do
        i2t = unquote(i2t)
        PythonBridge.unload_all_models(env)

        {:ok, image} = GenAI.invoke(env, "SD35Medium", "a red apple")
        PythonBridge.swap_model_to_cpu(env, "SD35Medium")

        t0 = System.monotonic_time(:millisecond)
        {:ok, caption} = GenAI.invoke(env, i2t, image)
        elapsed = System.monotonic_time(:millisecond) - t0
        IO.puts("#{i2t} single: #{elapsed}ms")

        assert is_binary(caption)
        assert String.length(caption) > 0
      end

      @tag timeout: 600_000
      test "#{i2t} batch invoke (3 images)", %{env: env} do
        i2t = unquote(i2t)
        PythonBridge.unload_all_models(env)

        prompts = ["a red apple", "a blue car", "a green tree"]
        {:ok, images} = GenAI.invoke_batch(env, "SD35Medium", prompts)
        PythonBridge.swap_model_to_cpu(env, "SD35Medium")

        t0 = System.monotonic_time(:millisecond)
        {:ok, captions} = GenAI.invoke_batch(env, i2t, images)
        elapsed = System.monotonic_time(:millisecond) - t0
        IO.puts("#{i2t} batch(3): #{elapsed}ms total, #{div(elapsed, 3)}ms/image")

        assert length(captions) == 3

        Enum.each(captions, fn caption ->
          assert is_binary(caption)
          assert String.length(caption) > 0
        end)
      end
    end
  end

  describe "swap integration" do
    @tag timeout: 600_000
    test "swap models between GPU and CPU", %{env: env} do
      PythonBridge.unload_all_models(env)

      {:ok, img1} = GenAI.invoke(env, "SD35Medium", "a red apple")
      assert is_binary(img1)
      :ok = PythonBridge.swap_model_to_cpu(env, "SD35Medium")

      {:ok, caption} = GenAI.invoke(env, "Moondream", img1)
      assert String.length(caption) > 0
      :ok = PythonBridge.swap_model_to_cpu(env, "Moondream")

      {:ok, img2} = GenAI.invoke(env, "SD35Medium", caption)
      assert is_binary(img2)
      assert byte_size(img2) > 100
      assert <<_::binary-size(4), "ftyp", _::binary>> = img2
    end
  end

  describe "all model combinations smoke test" do
    @real_text_embedding_models ~w(STSBMpnet STSBRoberta STSBDistilRoberta Nomic JinaClip Qwen3Embed)
    @real_image_embedding_models ~w(NomicVision JinaClipVision)

    for t2i <- ~w(SD35Medium ZImageTurbo Flux2Klein Flux2Dev QwenImage HunyuanImage GLMImage),
        i2t <- ~w(Moondream Qwen25VL Gemma3n Pixtral LLaMA32Vision Phi4Vision) do
      @tag timeout: 900_000
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

    for t2i <- ~w(SD35Medium ZImageTurbo Flux2Klein Flux2Dev QwenImage HunyuanImage GLMImage),
        i2t <- ~w(Moondream Qwen25VL Gemma3n Pixtral LLaMA32Vision Phi4Vision) do
      @tag timeout: 900_000
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
