defmodule PanicTda do
  use Ash.Domain

  resources do
    resource PanicTda.Experiment do
      define :create_experiment, action: :create
      define :get_experiment, action: :read, get_by: [:id]
      define :start_experiment, action: :start
      define :complete_experiment, action: :complete
    end

    resource PanicTda.Run do
      define :create_run, action: :create
      define :list_runs, action: :read
    end

    resource PanicTda.Invocation do
      define :create_invocation, action: :create
    end

    resource PanicTda.Embedding do
      define :create_embedding, action: :create
      define :list_embeddings, action: :read
    end

    resource PanicTda.PersistenceDiagram do
      define :create_persistence_diagram, action: :create
      define :get_persistence_diagram, action: :read, get_by: [:id]
      define :list_persistence_diagrams, action: :read
    end

    resource PanicTda.ClusteringResult do
      define :create_clustering_result, action: :create
      define :list_clustering_results, action: :read
    end

    resource PanicTda.EmbeddingCluster do
      define :create_embedding_cluster, action: :create
      define :list_embedding_clusters, action: :read
    end
  end
end
