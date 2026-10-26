defmodule PanicTda.Repo do
  use AshSqlite.Repo,
    otp_app: :panic_tda

  def min_sqlite_version do
    %Version{major: 3, minor: 53, patch: 4}
  end

  def installed_extensions do
    []
  end
end
