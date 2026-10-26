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
      deps: deps(),
      usage_rules: usage_rules()
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
      {:ash, "~> 3.37"},
      {:ash_sqlite, "~> 0.3"},
      {:ecto_sqlite3, "~> 0.26"},
      {:nx, "~> 1.0"},
      {:explorer, "~> 0.13"},
      {:jason, "~> 1.4"},
      {:snex, "~> 0.4"},
      {:vix, "~> 0.42"},
      {:usage_rules, "~> 1.3", only: [:dev]},
      {:tidewave, "~> 0.9", only: :dev},
      {:bandit, "~> 1.14", only: :dev}
    ]
  end

  defp usage_rules do
    [
      file: "CLAUDE.md",
      usage_rules: [{:ash, link: :markdown}, {~r/^ash_/, link: :markdown}]
    ]
  end

  defp aliases do
    [
      setup: ["deps.get", "ash.setup"],
      "ash.setup": ["ash.codegen", "ecto.create", "ecto.migrate"],
      "ash.codegen": ["ash.codegen --name initial_setup"],
      test: ["ecto.create --quiet", "ecto.migrate --quiet", "test"],
      tidewave:
        "run --no-halt -e 'Agent.start(fn -> Bandit.start_link(plug: Tidewave, port: 4000) end)'"
    ]
  end
end
