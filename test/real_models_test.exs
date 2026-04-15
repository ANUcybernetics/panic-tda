defmodule PanicTda.RealModelsTest do
  use ExUnit.Case, async: false

  @moduletag :gpu
  @moduletag timeout: 600_000

  require Ash.Query

  alias PanicTda.Engine
  alias PanicTda.Models.{GenAI, Embeddings, PythonBridge, PythonInterpreter}

  setup_all do
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

  setup %{env: env} do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
    PythonBridge.unload_all_models(env)
    :ok
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
    defp assert_unit_norm(binary) do
      f32_count = div(byte_size(binary), 4)
      values =
        for <<v::float-32-little <- binary>>, do: v

      assert length(values) == f32_count
      norm = :math.sqrt(Enum.reduce(values, 0.0, fn v, acc -> acc + v * v end))
      assert_in_delta norm, 1.0, 1.0e-4
    end

    test "STSBMpnet generates 256-dim unit-norm float32 text embeddings", %{env: env} do
      {:ok, [emb1, emb2]} = Embeddings.embed(env, "STSBMpnet", ["hello world", "test sentence"])

      assert is_binary(emb1)
      assert is_binary(emb2)
      assert byte_size(emb1) == 256 * 4
      assert byte_size(emb2) == 256 * 4
      assert_unit_norm(emb1)
      assert_unit_norm(emb2)
    end

    test "NomicVision generates 256-dim unit-norm float32 image embeddings", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "SD35Medium", "A blue sky")
      {:ok, [emb]} = Embeddings.embed(env, "NomicVision", [image])

      assert is_binary(emb)
      assert byte_size(emb) == 256 * 4
      assert_unit_norm(emb)
    end

    test "ColNomic generates float32 text embeddings", %{env: env} do
      {:ok, [emb1, emb2]} = Embeddings.embed(env, "ColNomic", ["hello world", "test sentence"])

      assert is_binary(emb1)
      assert is_binary(emb2)
      assert byte_size(emb1) == byte_size(emb2)
      assert rem(byte_size(emb1), 4) == 0
      assert byte_size(emb1) > 0
    end

    test "ColNomicVision generates float32 image embeddings", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "SD35Medium", "A blue sky")
      {:ok, [emb]} = Embeddings.embed(env, "ColNomicVision", [image])

      assert is_binary(emb)
      assert rem(byte_size(emb), 4) == 0
      assert byte_size(emb) > 0
    end

    test "ColNomic batches 9 texts correctly (spans internal chunk size)", %{env: env} do
      texts =
        for i <- 1..9, do: "sample sentence number #{i} describing an object or scene"

      {:ok, embs} = Embeddings.embed(env, "ColNomic", texts)

      assert length(embs) == 9
      sizes = Enum.map(embs, &byte_size/1)
      assert Enum.uniq(sizes) == [Enum.at(sizes, 0)]
      assert Enum.at(sizes, 0) > 0
      assert rem(Enum.at(sizes, 0), 4) == 0

      # distinct inputs should produce distinct vectors
      assert length(Enum.uniq(embs)) == 9
    end

    test "ColNomicVision batches 6 images correctly (spans internal chunk size)", %{env: env} do
      prompts = [
        "a red apple",
        "a blue ocean",
        "a green forest",
        "a yellow sunflower",
        "a white cloud",
        "a black cat"
      ]

      images =
        Enum.map(prompts, fn p ->
          {:ok, img} = GenAI.invoke(env, "SD35Medium", p)
          img
        end)

      {:ok, embs} = Embeddings.embed(env, "ColNomicVision", images)

      assert length(embs) == 6
      sizes = Enum.map(embs, &byte_size/1)
      assert Enum.uniq(sizes) == [Enum.at(sizes, 0)]
      assert Enum.at(sizes, 0) > 0
      assert rem(Enum.at(sizes, 0), 4) == 0

      # distinct images should produce distinct vectors
      assert length(Enum.uniq(embs)) == 6
    end
  end

  describe "end-to-end pipeline with real models" do
    test "full experiment with SD35Medium + Moondream + STSBMpnet", %{env: env} do
      {:ok, experiment} =
        PanicTda.Experiment
        |> Ash.Changeset.for_create(:create, %{
          networks: [["SD35Medium", "Moondream"]],
          prompts: ["A peaceful garden"],
          embedding_models: ["STSBMpnet"],
          max_length: 4
        })
        |> Ash.create()

      {:ok, completed} = Engine.perform_experiment(experiment.id, env: env)

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
        assert Nx.shape(emb.vector) == {256}
      end)

      pds = Ash.read!(PanicTda.PersistenceDiagram)
      assert length(pds) == 1
      pd = hd(pds)
      assert pd.embedding_model == "STSBMpnet"
      assert pd.diagram_data != nil
    end
  end

  describe "per-model T2I tests" do
    for t2i <- ~w(SD35Medium ZImageTurbo Flux2Klein Flux2Dev HunyuanImage GLMImage) do
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
    for i2t <- ~w(Moondream Qwen25VL Gemma3n Pixtral LLaMA32Vision Florence2) do
      @tag timeout: 600_000
      test "#{i2t} single invoke", %{env: env} do
        i2t = unquote(i2t)
        PythonBridge.unload_all_models(env)

        {:ok, image} = GenAI.invoke(env, "SD35Medium", "a red apple")
        PythonBridge.swap_model_to_cpu(env, "SD35Medium")

        t0 = System.monotonic_time(:millisecond)
        {:ok, caption} = GenAI.invoke(env, i2t, image)
        elapsed = System.monotonic_time(:millisecond) - t0
        IO.puts("#{i2t} single: #{elapsed}ms\n  caption: #{caption}")

        assert is_binary(caption)
        assert String.length(caption) > 0
        refute caption == "[empty]", "#{i2t} returned empty caption"
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

        Enum.each(Enum.zip(prompts, captions), fn {prompt, caption} ->
          IO.puts("  [#{prompt}]: #{caption}")
          assert is_binary(caption)
          assert String.length(caption) > 0
          refute caption == "[empty]", "#{i2t} returned empty caption for '#{prompt}'"
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
    @real_text_embedding_models ~w(STSBMpnet STSBRoberta STSBDistilRoberta Nomic JinaClip Qwen3Embed ColNomic)
    @real_image_embedding_models ~w(NomicVision JinaClipVision ColNomicVision)

    for t2i <- ~w(SD35Medium ZImageTurbo Flux2Klein Flux2Dev HunyuanImage GLMImage),
        i2t <- ~w(Moondream Qwen25VL Gemma3n Pixtral LLaMA32Vision Florence2) do
      @tag timeout: 900_000
      test "pipeline: #{t2i} + #{i2t} with all text embedding models", %{env: env} do
        t2i = unquote(t2i)
        i2t = unquote(i2t)

        experiment =
          PanicTda.create_experiment!(%{
            networks: [[t2i, i2t]],
            prompts: ["a red apple"],
            embedding_models: @real_text_embedding_models,
            max_length: 4
          })

        {:ok, completed} = Engine.perform_experiment(experiment.id, env: env)

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

    for t2i <- ~w(SD35Medium ZImageTurbo Flux2Klein Flux2Dev HunyuanImage GLMImage),
        i2t <- ~w(Moondream Qwen25VL Gemma3n Pixtral LLaMA32Vision Florence2) do
      @tag timeout: 900_000
      test "pipeline: #{t2i} + #{i2t} with all image embedding models", %{env: env} do
        t2i = unquote(t2i)
        i2t = unquote(i2t)

        experiment =
          PanicTda.create_experiment!(%{
            networks: [[t2i, i2t]],
            prompts: ["a red apple"],
            embedding_models: @real_image_embedding_models,
            max_length: 4
          })

        {:ok, completed} = Engine.perform_experiment(experiment.id, env: env)

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
