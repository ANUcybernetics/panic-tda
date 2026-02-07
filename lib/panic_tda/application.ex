defmodule PanicTda.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      PanicTda.Repo
    ]

    opts = [strategy: :one_for_one, name: PanicTda.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
