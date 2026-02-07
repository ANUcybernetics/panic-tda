defmodule PanicTda.Repo do
  use AshSqlite.Repo,
    otp_app: :panic_tda

  def installed_extensions do
    []
  end
end
