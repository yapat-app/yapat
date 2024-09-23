import pandas as pd

from sklearn.preprocessing import StandardScaler
import dask

from extensions import sqlalchemy_db
from schema_model import Dataset, EmbeddingResult

class BaseDimensionalityReduction:
    """
    Base class for dimensionality reduction models.

    Attributes:
    -----------
    n_components : int
        The number of dimensions to reduce the data to.
    transformed_data : pd.DataFrame or None
        DataFrame containing the reduced dimensionality data.

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

    def __init__(self, n_components: int = 3):
        """
        Initialize the BaseDimensionalityReduction class.

        :param n_components: The number of dimensions to reduce the data to.
        """
        self.n_components = n_components
        self.transformed_data = None  # Placeholder for the transformed data
        self.data = None

    def load_data(self, dataset_id: int, embedding_id: int) -> pd.DataFrame:
        """
        Load the data to be reduced from a CSV or pickle file.

        :param file_path: Path to the data file (CSV or pickle format).
        :return: DataFrame containing the loaded data.
        """
        embedding_result = sqlalchemy_db.session.query(EmbeddingResult).filter_by(
            dataset_id=dataset_id,
            embedding_id=embedding_id
        ).one_or_none()
        file_path =  embedding_result.file_path


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

    def save_transformed_data(self, file_path: str):
        """
        Save the cluster labels to a CSV or pickle file.

        :param file_path: Path where the cluster labels will be saved.
        """
        if self.transformed_data is None:
            raise ValueError("No cluster labels found. Fit the model or predict labels first.")

        # Save as CSV or pickle depending on the file extension.
        if file_path.endswith('.csv'):
            self.transformed_data.to_csv(file_path, index=False)
        elif file_path.endswith('.pkl'):
            self.transformed_data.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}. Please use CSV or pickle.")

def get_dr_model(method_name: str, dask_client: dask.distributed.client.Client or None = None):
    if method_name == "pca":
        from dimensionality_reduction.pca import PCA
        return PCA()
    elif method_name == "tsne":
        from dimensionality_reduction.tsne import TSNE
        return TSNE()
    elif method_name == "umap":
        from dimensionality_reduction.umap_reduction import Umap
        return Umap()
    else:
        raise ValueError(f"Unknown DR method: {method_name}")
