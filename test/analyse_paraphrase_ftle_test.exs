defmodule PanicTda.AnalyseParaphraseFtleTest do
  use ExUnit.Case

  alias PanicTda.Models.PythonInterpreter

  setup_all do
    {:ok, interpreter} = PythonInterpreter.start_link()
    {:ok, env} = Snex.make_env(interpreter)

    priv_python = :code.priv_dir(:panic_tda) |> to_string() |> Path.join("python")

    {:ok, _} =
      Snex.pyeval(
        env,
        """
        import sys
        if _priv_python not in sys.path:
            sys.path.insert(0, _priv_python)
        import penguin_analysis
        return True
        """,
        %{"_priv_python" => priv_python}
      )

    {:ok, env: env}
  end

  describe "plot_ftle_grid" do
    test "writes a non-empty PNG given a small synthetic CSV", %{env: env} do
      tmp_dir = Path.join(System.tmp_dir!(), "ftle_grid_test_#{System.unique_integer([:positive])}")
      File.mkdir_p!(tmp_dir)
      csv_path = Path.join(tmp_dir, "ftle_values.csv")
      out_path = Path.join(tmp_dir, "ftle_grid.png")

      File.write!(csv_path, """
      experiment_id,network,embedding_model,category,prompt_or_pair,ftle,r_squared,num_pairs,num_timesteps
      exp1,SD35Medium|Moondream,Nomic,identical,p1,0.01,0.9,28,200
      exp1,SD35Medium|Moondream,Nomic,identical,p2,0.012,0.9,28,200
      exp1,SD35Medium|Moondream,Nomic,paraphrase,p1 || p2,0.03,0.9,64,200
      exp1,SD35Medium|Moondream,Qwen3Embed,identical,p1,0.009,0.9,28,200
      exp1,SD35Medium|Moondream,Qwen3Embed,paraphrase,p1 || p2,0.028,0.9,64,200
      """)

      {:ok, true} =
        Snex.pyeval(
          env,
          """
          import penguin_analysis
          penguin_analysis.plot_ftle_grid(csv_path, out_path)
          return True
          """,
          %{"csv_path" => csv_path, "out_path" => out_path}
        )

      stat = File.stat!(out_path)
      assert stat.size > 1000
      File.rm_rf!(tmp_dir)
    end
  end

  describe "plot_divergence_curves" do
    test "writes a non-empty PNG given two synthetic divergence curves", %{env: env} do
      tmp_dir = Path.join(System.tmp_dir!(), "div_curves_test_#{System.unique_integer([:positive])}")
      File.mkdir_p!(tmp_dir)
      out_path = Path.join(tmp_dir, "divergence_curves.png")

      {:ok, true} =
        Snex.pyeval(
          env,
          """
          import penguin_analysis
          import numpy as np

          t = np.arange(200)
          identical_curve = np.exp(0.005 * t + 0.01 * np.random.default_rng(0).normal(size=t.shape)).tolist()
          paraphrase_curve = np.exp(0.03 * t + 0.01 * np.random.default_rng(1).normal(size=t.shape)).tolist()

          penguin_analysis.plot_divergence_curves(
              out_path,
              network="SD35Medium|Moondream",
              embedding_model="Nomic",
              identical_curve=identical_curve,
              paraphrase_curve=paraphrase_curve,
          )
          return True
          """,
          %{"out_path" => out_path}
        )

      stat = File.stat!(out_path)
      assert stat.size > 1000
      File.rm_rf!(tmp_dir)
    end
  end

  describe "Mix.Tasks.Analyse.ParaphraseFtle" do
    setup do
      :ok = Ecto.Adapters.SQL.Sandbox.checkout(PanicTda.Repo)
      :ok
    end

    test "raises on no args" do
      assert_raise Mix.Error, ~r/Usage:/, fn ->
        Mix.Tasks.Analyse.ParaphraseFtle.run([])
      end
    end

    test "raises when no experiment matches the id prefix" do
      assert_raise Mix.Error, ~r/No experiment found/, fn ->
        Mix.Tasks.Analyse.ParaphraseFtle.run(["nonexistent-prefix"])
      end
    end
  end

  describe "cross_prompt_ftle" do
    test "recovers a known exponential divergence rate", %{env: env} do
      {:ok, result} =
        Snex.pyeval(
          env,
          """
          import numpy as np
          import base64

          rng = np.random.default_rng(0)
          num_runs = 8
          num_timesteps = 60
          dimension = 4
          lambda_true = 0.05

          noise_a = rng.normal(0, 0.01, size=(num_runs, num_timesteps, dimension)).astype(np.float32)
          noise_b = rng.normal(0, 0.01, size=(num_runs, num_timesteps, dimension)).astype(np.float32)

          t = np.arange(num_timesteps, dtype=np.float32)
          drift = np.zeros((num_runs, num_timesteps, dimension), dtype=np.float32)
          drift[:, :, 0] = np.exp(lambda_true * t)[None, :]

          traj_a = noise_a
          traj_b = drift + noise_b

          a_b64 = base64.b64encode(traj_a.tobytes()).decode()
          b_b64 = base64.b64encode(traj_b.tobytes()).decode()

          result = penguin_analysis.cross_prompt_ftle(
              a_b64, b_b64, num_runs, num_runs, num_timesteps, dimension
          )
          return {"lambda_true": lambda_true, "result": result}
          """,
          %{}
        )

      lambda_true = result["lambda_true"]
      ftle = result["result"]

      assert_in_delta ftle["exponent"], lambda_true, 0.005
      assert ftle["r_squared"] > 0.999
      assert ftle["num_pairs"] == 64
      assert ftle["num_timesteps"] == 60
      assert length(ftle["divergence_curve"]) == 60
    end
  end
end
