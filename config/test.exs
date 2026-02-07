import Config

config :panic_tda, PanicTda.Repo,
  database: "priv/panic_tda_test.db",
  pool: Ecto.Adapters.SQL.Sandbox

config :ash,
  disable_async?: true

config :logger, level: :warning
