import dask
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy import and_

from extensions import sqlalchemy_db
from schema_model import Dataset, EmbeddingMethod, EmbeddingResult


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

    def __init__(self, dataset_name, embedding_method) -> None:
        """
        Initialize the BaseClustering class. Sets up the data and labels attributes.
        """
        self.dataset_name = dataset_name
        self.embedding_method = embedding_method
        self.data = None  # Will hold the data to be clustered.
        self.labels = None  # Will store the cluster labels after fitting the model.

    def load_data(self) -> pd.DataFrame:
        """
        Load the data to be clustered from a CSV or pickle file.

        :param file_path: Path to the data file (CSV or pickle format).
        :return: DataFrame containing the loaded data.
        """
        dataset_id = sqlalchemy_db.session.execute(
            sqlalchemy_db.select(Dataset.id).where(Dataset.dataset_name == self.dataset_name)
        ).scalar_one_or_none()
        embedding_id = sqlalchemy_db.session.execute(
            sqlalchemy_db.select(EmbeddingMethod.id).where(EmbeddingMethod.method_name == self.embedding_method)
        ).scalar_one_or_none()
        if dataset_id is None or embedding_id is None:
            return None


        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path, index_col=0)
        elif file_path.endswith('.pkl'):
            self.data = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}. Please use CSV or pickle.")
        return self.data

    def scale_data(self, data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.values)
        return scaled_data

    def save_labels(self, file_path: str):
        """
        Save the cluster labels to a CSV or pickle file.

        :param file_path: Path where the cluster labels will be saved.
        """
        if self.labels is None:
            raise ValueError("No cluster labels found. Fit the model or predict labels first.")

        # Save as CSV or pickle depending on the file extension.
        if file_path.endswith('.csv'):
            self.labels.to_csv(file_path, index=False)
        elif file_path.endswith('.pkl'):
            self.labels.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}. Please use CSV or pickle.")

        # SQL Query for saving clustering results




def get_clustering_model(method_name: str, dask_client: dask.distributed.client.Client or None = None):
    if method_name == "hdbscan":
        from clustering.hdbscan import HDBSCANClustering
        return HDBSCANClustering()
    elif method_name == "dbscan":
        from clustering.dbscan import DBSCANClustering
        return DBSCANClustering()
    elif method_name == "affinity":
        from clustering.affinity import Affinity
        return Affinity()
    elif method_name == "kmeans":
        from clustering.kmeans import KMeansClustering
        return KMeansClustering()
    elif method_name == "spectral":
        from clustering.spectral import SPECTRALClustering
        return SPECTRALClustering()
    elif method_name == "optics":
        from clustering.optics import OpticsClustering
        return OpticsClustering()
    else:
        raise ValueError(f"Unknown Clustering method: {method_name}")



