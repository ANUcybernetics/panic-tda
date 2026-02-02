import Config

config :panic_tda, PanicTda.Repo,
  database: System.get_env("DATABASE_PATH") || "priv/panic_tda_#{config_env()}.db"
