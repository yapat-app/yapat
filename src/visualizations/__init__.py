import pandas as pd

from extensions import sqlalchemy_db
from schema_model import EmbeddingResult, ClusteringResult
from sqlalchemy.exc import SQLAlchemyError


class BaseVisualization:


    def __init__(self, clustering_method_name, embedding_result_id) -> None:
        """
        Initialize the BaseClustering class. Sets up the data and labels attributes.
        """
        # The call back args can be changed if needed
        self.clustering_method_name = clustering_method_name
        self.embedding_result_id = embedding_result_id
        self.embeddings = None
        self.cluster_labels = None

    def load_data(self):
        """Loads the clustering and embedding results into the instance.

        Raises:
            ValueError: If the clustering or embedding results cannot be found.

        Returns:
            tuple: A tuple containing the embeddings DataFrame and the cluster labels DataFrame.
        """
        try:
            with sqlalchemy_db.session() as session:
                clustering_result = session.query(ClusteringResult).filter_by(
                    embedding_id=self.embedding_result_id,
                    method_id=self.clustering_method_name
                ).one_or_none()

                embedding_result = session.query(EmbeddingResult).filter_by(
                    id=self.embedding_result_id
                ).one_or_none()

                if not clustering_result:
                    raise ValueError("Clustering result not found.")
                if not embedding_result:
                    raise ValueError("Embedding result not found.")

                cluster_file_path = clustering_result.cluster_file_path
                embedding_file_path = embedding_result.file_path

                # Load clustering labels
                if cluster_file_path.endswith('.csv'):
                    self.cluster_labels = pd.read_csv(cluster_file_path, index_col=0)
                elif cluster_file_path.endswith('.pkl'):
                    self.cluster_labels = pd.read_pickle(cluster_file_path)
                else:
                    raise ValueError("Unsupported file type for clustering results.")

                # Load embeddings
                if embedding_file_path.endswith('.csv'):
                    self.embeddings = pd.read_csv(embedding_file_path, index_col=0)
                elif embedding_file_path.endswith('.pkl'):
                    self.embeddings = pd.read_pickle(embedding_file_path)
                else:
                    raise ValueError("Unsupported file type for embeddings.")

                return self.embeddings, self.cluster_labels

        except SQLAlchemyError as e:
            raise Exception(f"Database error occurred: {e}")


