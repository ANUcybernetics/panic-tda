import Config

config :panic_tda, PanicTda.Repo,
  database: "priv/panic_tda_test.db",
  pool: Ecto.Adapters.SQL.Sandbox,
  ownership_timeout: 600_000

config :ash,
  disable_async?: true

config :logger, level: :warning
