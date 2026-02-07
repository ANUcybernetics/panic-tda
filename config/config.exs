import Config

config :panic_tda,
  ecto_repos: [PanicTda.Repo],
  ash_domains: [PanicTda]

config :panic_tda, PanicTda.Repo,
  database: "priv/panic_tda_dev.db",
  pool_size: 5

config :ash,
  disable_async?: false

import_config "#{config_env()}.exs"
