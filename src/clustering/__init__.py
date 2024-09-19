import pandas as pd
from abc import ABC, abstractmethod


class BaseClustering(ABC):
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

    fit(data: pd.DataFrame):
        Abstract method for fitting the clustering algorithm. Must be implemented by subclasses.

    predict(data: pd.DataFrame) -> pd.Series:
        Abstract method for predicting cluster labels. Must be implemented by subclasses.
    """

    def __init__(self) -> None:
        """
        Initialize the BaseClustering class. Sets up the data and labels attributes.
        """
        self.data = None  # Will hold the data to be clustered.
        self.labels = None  # Will store the cluster labels after fitting the model.

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the data to be clustered from a CSV or pickle file.

        :param file_path: Path to the data file (CSV or pickle format).
        :return: DataFrame containing the loaded data.
        """
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith('.pkl'):
            self.data = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}. Please use CSV or pickle.")
        return self.data

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

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """
        Abstract method for fitting the clustering algorithm on the data.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Abstract method for predicting cluster labels for the data.
        This method must be implemented by subclasses.
        """
        pass
