defmodule PanicTda.RealModelsTest do
  use ExUnit.Case, async: false

  @moduletag :gpu
  @moduletag timeout: 600_000

  require Ash.Query

  alias PanicTda.Engine
  alias PanicTda.Models.{GenAI, PythonBridge, PythonInterpreter}

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

  describe "embedding geometry" do
    # Qwen3-Embedding is a decoder and needs last-token pooling. Under the mean
    # pooling that produced every vector written before 2026-09-03 (TASK-96) the
    # space collapses: unrelated sentences sat about 0.06 apart rather than 0.67.
    # This asserts the separation a library upgrade must not silently undo.
    @unrelated_a "A black bicycle leans against a red brick wall."
    @unrelated_b "The stock market closed lower after a volatile trading session."
    @paraphrase_a "A dark-coloured bicycle is propped up against a wall of red bricks."

    test "unrelated text is far apart and paraphrases are close", %{env: env} do
      {:ok, [a, b, c]} =
        PanicTda.Models.Embeddings.embed(env, "Qwen3Embed", [
          @unrelated_a,
          @unrelated_b,
          @paraphrase_a
        ])

      unrelated = cosine_distance(a, b)
      paraphrase = cosine_distance(a, c)

      assert unrelated > 0.3,
             "unrelated sentences #{Float.round(unrelated, 4)} apart --- mean pooling gives ~0.06"

      assert paraphrase < 0.2,
             "paraphrases should stay close, got #{Float.round(paraphrase, 4)}"

      assert unrelated > paraphrase * 3
    end

    # The geometry test above catches a collapse. This catches any movement at
    # all: sentence-transformers 6 changed the pooling under us once already
    # (TASK-96) and every vector in the database had to be recomputed.
    # Regenerate the fixture with analysis/embedding_reference.py, and only
    # when the embedding path is deliberately changed.
    test "reproduces the stored reference vectors exactly", %{env: env} do
      %{"dimension" => dimension, "vectors" => reference} =
        "test/fixtures/qwen3embed_reference.json" |> File.read!() |> Jason.decode!()

      texts = Enum.map(reference, & &1["text"])
      {:ok, fresh} = PanicTda.Models.Embeddings.embed(env, "Qwen3Embed", texts)

      for {%{"text" => text, "vector_b64" => b64}, actual} <- Enum.zip(reference, fresh) do
        expected = Base.decode64!(b64)

        assert byte_size(actual) == dimension * 4,
               "expected #{dimension} float32s, got #{byte_size(actual) / 4}"

        distance = cosine_distance(expected, actual)

        assert distance < 1.0e-4,
               """
               the embedding path has moved: #{Float.round(distance, 6)} from the reference.

               text: #{String.slice(text, 0, 60)}
               Every stored vector is now on a different scale from fresh ones.
               """
      end
    end

    defp cosine_distance(a, b) do
      va = for <<x::float-32-little <- a>>, do: x
      vb = for <<x::float-32-little <- b>>, do: x

      dot = Enum.zip(va, vb) |> Enum.map(fn {x, y} -> x * y end) |> Enum.sum()
      1.0 - dot
    end
  end

  describe "real GenAI models" do
    test "SD35Medium generates valid AVIF image", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "SD35Medium", "A cat sitting on a mat")

      assert is_binary(image)
      assert byte_size(image) > 100
      assert <<_::binary-size(4), "ftyp", _::binary>> = image
    end

    test "Moondream3 generates text caption from image", %{env: env} do
      {:ok, image} = GenAI.invoke(env, "SD35Medium", "A red apple on a table")
      {:ok, caption} = GenAI.invoke(env, "Moondream3", image)

      assert is_binary(caption)
      assert String.length(caption) > 0
    end
  end

  describe "real Embedding models" do
  end

  describe "end-to-end pipeline with real models" do
    test "full experiment with SD35Medium + Moondream3 + Qwen3Embed", %{env: env} do
      {:ok, experiment} =
        PanicTda.Experiment
        |> Ash.Changeset.for_create(:create, %{
          networks: [["SD35Medium", "Moondream3"]],
          prompts: ["A peaceful garden"],
          embedding_models: ["Qwen3Embed"],
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
      assert inv1.model == "Moondream3"
      assert inv1.output_text != nil

      assert inv2.type == :image
      assert inv2.model == "SD35Medium"

      assert inv3.type == :text
      assert inv3.model == "Moondream3"

      embeddings =
        PanicTda.Embedding
        |> Ash.Query.filter(embedding_model == ^"Qwen3Embed")
        |> Ash.read!()

      assert length(embeddings) == 2

      Enum.each(embeddings, fn emb ->
        assert %Nx.Tensor{} = emb.vector
        assert Nx.shape(emb.vector) == {256}
      end)

      pds = Ash.read!(PanicTda.PersistenceDiagram)
      assert length(pds) == 1
      pd = hd(pds)
      assert pd.embedding_model == "Qwen3Embed"
      assert pd.diagram_data != nil
    end
  end

  describe "seed regeneration" do
    # TASK-93: the recorded seed is what makes any step regenerable. Same
    # seed, same input, same path gives the same bytes; a batch reproduces a
    # single call at the same seed only to within batched-kernel noise, so
    # the batch check is against the distance a different seed produces.
    test "a stored seed regenerates the invocation's image exactly", %{env: env} do
      experiment =
        PanicTda.create_experiment!(%{
          networks: [["SD35Medium", "Moondream3"]],
          prompts: ["a red apple on a wooden table"],
          embedding_models: ["Qwen3Embed"],
          max_length: 1
        })

      experiment = PanicTda.start_experiment!(experiment)
      [run] = Engine.init_runs(experiment)
      {:ok, session} = PanicTda.Models.PythonSession.wrap(env)
      :ok = PanicTda.Engine.RunExecutor.execute(session, run)

      [invocation] =
        PanicTda.Invocation
        |> Ash.Query.filter(run_id == ^run.id)
        |> Ash.read!()

      assert is_integer(invocation.seed)

      {:ok, regenerated} = GenAI.invoke(env, "SD35Medium", run.initial_prompt, invocation.seed)
      assert regenerated == invocation.output_image

      {:ok, other} = GenAI.invoke(env, "SD35Medium", run.initial_prompt, invocation.seed + 1)
      assert mean_pixel_difference(other, invocation.output_image) > 10
    end

    test "a batched item's image depends only on its own seed", %{env: env} do
      prompt = "a red apple on a wooden table"
      seed = GenAI.draw_seed()

      {:ok, [alone]} = GenAI.invoke_batch(env, "SD35Medium", [prompt], [seed])

      {:ok, [_, with_partners]} =
        GenAI.invoke_batch(env, "SD35Medium", ["a blue bicycle", prompt], [
          GenAI.draw_seed(),
          seed
        ])

      {:ok, [_, other_seed]} =
        GenAI.invoke_batch(env, "SD35Medium", ["a blue bicycle", prompt], [
          GenAI.draw_seed(),
          seed + 1
        ])

      assert mean_pixel_difference(alone, with_partners) < 5
      assert mean_pixel_difference(alone, other_seed) > 10
    end

    defp mean_pixel_difference(avif_a, avif_b) do
      {:ok, a} = Vix.Vips.Image.new_from_buffer(avif_a)
      {:ok, b} = Vix.Vips.Image.new_from_buffer(avif_b)
      {:ok, diff} = Vix.Vips.Operation.subtract(a, b)
      {:ok, abs} = Vix.Vips.Operation.abs(diff)
      {:ok, mean} = Vix.Vips.Operation.avg(abs)
      mean
    end
  end

  describe "per-model T2I tests" do
    for t2i <- ~w(SD35Medium ZImageTurbo Flux2Klein Flux2Dev) do
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
    # every registered captioner, one image each
    for i2t <- ~w(Moondream3 Qwen25VL Qwen3VL Gemma4 JoyCaption) do
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

      {:ok, caption} = GenAI.invoke(env, "Moondream3", img1)
      assert String.length(caption) > 0
      :ok = PythonBridge.swap_model_to_cpu(env, "Moondream3")

      {:ok, img2} = GenAI.invoke(env, "SD35Medium", caption)
      assert is_binary(img2)
      assert byte_size(img2) > 100
      assert <<_::binary-size(4), "ftyp", _::binary>> = img2
    end
  end

  describe "all model combinations smoke test" do
    @real_text_embedding_models ~w(Qwen3Embed)

    # the balanced_panel_5x5_v2 lineup: the combinations we will actually run.
    # Captioners outside it are covered by the per-model tests above.
    for t2i <- ~w(SD35Medium ZImageTurbo Flux2Klein Flux2Dev),
        i2t <- ~w(Moondream3 Qwen25VL Qwen3VL Gemma4 JoyCaption) do
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
  end
end
