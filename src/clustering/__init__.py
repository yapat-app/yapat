import dask
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import uuid
import pickle
import logging
from sqlalchemy import and_
from src import server, sqlalchemy_db
from sqlalchemy.exc import SQLAlchemyError
from schema_model import Dataset, EmbeddingResult, ClusteringResult, ClusteringMethod, EmbeddingMethod

logger = logging.getLogger(__name__)

class BaseClustering:
    """
    Base class for clustering models used to group data based on embeddings or other features.

    This class provides core functionality for clustering models, including loading data and storing
    clustering results. It is meant to be subclassed by specific clustering algorithms, which should
    implement their own logic for fitting the model and predicting clusters.

    Attributes:
    -----------
    data : pd.DataFrame or None
        DataFrame containing the data to be clustered.
    labels : pd.Series or None
        Series containing the cluster labels assigned to the data.

    Methods:
    --------
    load_data(file_path: str) -> pd.DataFrame:
        Loads a dataset from a CSV or pickle file into a pandas DataFrame.

    save_labels(file_path: str):
        Saves the cluster labels to a CSV or pickle file.

    """

    def __init__(self) -> None:
        """
        Initialize the BaseClustering class. Sets up the data and labels attributes.
        """

        self.data = None  # Will hold the data to be clustered.
        self.labels = None  # Will store the cluster labels after fitting the model.

    def load_data(self, embedding_method_name):
        session = sqlalchemy_db.session
        try:
            with server.app_context():
                selected_dataset = session.query(Dataset).filter_by(is_selected=True).first()
                if not selected_dataset:
                    logger.error("No dataset is currently selected.")

                embedding_method = session.query(EmbeddingMethod).filter_by(method_name=embedding_method_name).first()
                if not embedding_method:
                    logger.error(f"No embedding method found for '{embedding_method_name}'.")

                embedding_result = session.query(EmbeddingResult).filter_by(dataset_id=selected_dataset.id,
                                                                            embedding_id=embedding_method.id).first()
                if not embedding_result:
                    logger.error(f"No embedding result found for dataset '{selected_dataset.dataset_name}'.")

                # Check if the file path exists
                embedding_file_path = embedding_result.file_path
                if not os.path.exists(embedding_file_path):
                    logger.error(f"Embedding result file does not exist at path: {embedding_file_path}")
                    raise FileNotFoundError(f"Embedding result file does not exist at path: {embedding_file_path}")

                embedding_data_df = pd.read_pickle(embedding_file_path)
                return embedding_data_df

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise Exception(f"Database error: {e}")

        except (ValueError, FileNotFoundError) as e:
            raise e  # Re-raise to handle it outside if needed

        except Exception as e:
            logger.error(f"Error loading embedding data: {e}")
            raise Exception(f"Error loading embedding data: {e}")


    def scale_data(self, data):
        data.dropna(axis=1, inplace=True)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.values)
        return scaled_data

    def save_labels(self, clustering_method_name: str, embedding_method_name: str, labels: pd.DataFrame):
        session = sqlalchemy_db.session
        try:
            with server.app_context():
                selected_dataset = session.query(Dataset).filter_by(is_selected=True).first()
                if not selected_dataset:
                    raise ValueError("No dataset is currently selected.")
                embedding_method = session.query(EmbeddingMethod).filter_by(method_name=embedding_method_name).first()
                if not embedding_method:
                    raise ValueError(f"No embedding method found for '{embedding_method_name}'.")

                embedding_result = session.query(EmbeddingResult).filter_by(
                    dataset_id=selected_dataset.id,
                    embedding_id=embedding_method.id
                ).first()
                if not embedding_result:
                    raise ValueError(
                        f"No embedding result found for dataset '{selected_dataset.dataset_name}' using the '{embedding_method_name}' method.")

                clustering_method = session.query(ClusteringMethod).filter_by(
                    method_name=clustering_method_name).first()
                if not clustering_method:
                    raise ValueError(f"No clustering method found for '{clustering_method_name}'.")

                # Generate a unique file name and save the clustering labels to a pickle file
                unique_filename = f"{uuid.uuid4().hex}_clustering.pkl"
                clustering_file_path = os.path.join('results/', unique_filename)
                labels.to_pickle(clustering_file_path)

                # Check if a ClusteringResult already exists for this embedding and clustering method
                clustering_result = session.query(ClusteringResult).filter_by(
                    embedding_id=embedding_result.id,
                    method_id=clustering_method.method_id
                ).first()

                if not clustering_result:
                    clustering_result = ClusteringResult(
                        embedding_id=embedding_result.id,
                        method_id=clustering_method.method_id,
                        cluster_file_path=clustering_file_path,
                        hyperparameters={},  # Add relevant hyperparameters if any
                        evaluation_results={},
                        task = 'completed'
                    )
                    session.add(clustering_result)
                else:
                    # Update the existing ClusteringResult with the new file path
                    clustering_result.cluster_file_path = clustering_file_path
                    clustering_result.task = 'completed'
                session.commit()
                print(f"Clustering labels saved successfully at {clustering_file_path}")

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise Exception(f"Database error: {e}")
        except Exception as e:
            logger.error(f"Error saving clustering labels: {e}")
            raise Exception(f"Error saving clustering labels: {e}")



def get_clustering_model(method_name: str, *args, **kwargs):
    if method_name == "hdbscan":
        from clustering.hdbscan import HDBSCANClustering
        return HDBSCANClustering(*args, **kwargs)
    elif method_name == "dbscan":
        from clustering.dbscan import DBSCANClustering
        return DBSCANClustering(*args, **kwargs)
    elif method_name == "affinity":
        from clustering.affinity import Affinity
        return Affinity(*args, **kwargs)
    elif method_name == "kmeans":
        from clustering.kmeans import KMeansClustering
        return KMeansClustering(*args, **kwargs)
    elif method_name == "spectral":
        from clustering.spectral import SPECTRALClustering
        return SPECTRALClustering(*args, **kwargs)
    elif method_name == "optics":
        from clustering.optics import OpticsClustering
        return OpticsClustering(*args, **kwargs)
    else:
        print(f"Unknown Clustering method: {method_name}")
