defmodule PanicTda.EmbeddingCluster do
  use Ash.Resource,
    domain: PanicTda,
    data_layer: AshSqlite.DataLayer

  sqlite do
    table("embedding_clusters")
    repo(PanicTda.Repo)

    # Expression indexes for the CAST predicates ash_sql emits on uuid columns
    # (backlog/docs/ash-sql-cast-issue.md); drop them with TASK-99.
    custom_indexes do
      index(["CAST(id AS TEXT)"], name: "embedding_clusters_id_text_index")
      index(["CAST(embedding_id AS TEXT)"], name: "embedding_clusters_embedding_id_text_index")

      index(["CAST(clustering_result_id AS TEXT)"],
        name: "embedding_clusters_clustering_result_id_text_index"
      )
    end
  end

  attributes do
    uuid_v7_primary_key(:id)

    create_timestamp(:inserted_at)
    update_timestamp(:updated_at)
  end

  relationships do
    belongs_to :embedding, PanicTda.Embedding do
      allow_nil?(false)
      attribute_type(:uuid_v7)
    end

    belongs_to :clustering_result, PanicTda.ClusteringResult do
      allow_nil?(false)
      attribute_type(:uuid_v7)
    end

    belongs_to :medoid_embedding, PanicTda.Embedding do
      allow_nil?(true)
      attribute_type(:uuid_v7)
      source_attribute(:medoid_embedding_id)
      destination_attribute(:id)
    end
  end

  identities do
    identity(:unique_embedding_clustering, [:embedding_id, :clustering_result_id])
  end

  actions do
    defaults([:read, :destroy])

    create :create do
      accept([:embedding_id, :clustering_result_id, :medoid_embedding_id])
    end

    update :update do
      accept([:medoid_embedding_id])
    end
  end

  calculations do
    calculate(:is_outlier, :boolean, expr(is_nil(medoid_embedding_id)))
  end
end
