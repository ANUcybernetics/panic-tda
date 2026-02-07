defmodule PanicTda.MixProject do
  use Mix.Project

  def project do
    [
      app: :panic_tda,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      consolidate_protocols: Mix.env() != :dev,
      aliases: aliases(),
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {PanicTda.Application, []}
    ]
  end

  defp deps do
    [
      {:ash, "~> 3.4"},
      {:ash_sqlite, "~> 0.2"},
      {:ecto_sqlite3, "~> 0.18"},
      {:nx, "~> 0.9"},
      {:jason, "~> 1.4"},
      {:snex, "~> 0.3"},
      {:vix, "~> 0.35"},
      {:usage_rules, "~> 0.1", only: [:dev]}
    ]
  end

  defp aliases do
    [
      setup: ["deps.get", "ash.setup"],
      "ash.setup": ["ash.codegen", "ecto.create", "ecto.migrate"],
      "ash.codegen": ["ash.codegen --name initial_setup"],
      test: ["ecto.create --quiet", "ecto.migrate --quiet", "test"]
    ]
  end
end
