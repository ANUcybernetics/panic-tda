defmodule PanicTda do
  use Ash.Domain

  resources do
    resource(PanicTda.Experiment)
    resource(PanicTda.Run)
    resource(PanicTda.Invocation)
    resource(PanicTda.Embedding)
    resource(PanicTda.PersistenceDiagram)
    resource(PanicTda.ClusteringResult)
    resource(PanicTda.EmbeddingCluster)
  end
end
