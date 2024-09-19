import hdbscan
import pandas as pd
from sklearn.preprocessing import StandardScaler

from clustering import BaseClustering


class HDBSCANClustering(BaseClustering):
    """
    HDBSCAN clustering algorithm implementation that inherits from BaseClustering.

    This class provides functionality to perform HDBSCAN clustering on a dataset.
    It scales the data using StandardScaler before clustering and allows for the
    prediction of cluster membership probabilities.

    Attributes:
    -----------
    min_cluster_size : int
        The minimum size of a cluster. Smaller clusters will be considered noise.
    clusterer : hdbscan.HDBSCAN or None
        The HDBSCAN model after fitting.

    Methods:
    --------
    fit(data: pd.DataFrame):
        Fit the HDBSCAN clustering algorithm to the data.

    predict(data: pd.DataFrame) -> pd.Series:
        Predicts cluster labels for new data.

    get_membership_probabilities(data: pd.DataFrame) -> pd.DataFrame:
        Returns the membership probabilities for each point belonging to a cluster.
    """

    def __init__(self, min_cluster_size=5):
        """
        Initialize the HDBSCANClustering class with the minimum cluster size.

        :param min_cluster_size: The minimum size of clusters. Clusters smaller than this size will be treated as noise.
        """
        super().__init__()
        self.min_cluster_size = min_cluster_size
        self.clusterer = None  # Placeholder for the HDBSCAN model.

    def fit(self, data: pd.DataFrame):
        """
        Fit the HDBSCAN clustering algorithm to the dataset.

        The data will be scaled using StandardScaler before fitting the model.

        :param data: DataFrame containing the data to be clustered.
        :return: Series containing the cluster labels assigned to the data.
        """
        # Scale the data using StandardScaler before clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Initialize and fit HDBSCAN clusterer
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        self.clusterer.fit(scaled_data)

        # Store cluster labels in self.labels as a pandas Series
        self.labels = pd.Series(self.clusterer.labels_, index=data.index, name='Cluster_Label')
        return self.labels

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict cluster labels for new data using the fitted HDBSCAN model.

        Note: HDBSCAN does not support true prediction on new, unseen data (like K-Means).
        Instead, it provides membership probabilities, which can be used to assign clusters.

        :param data: DataFrame containing the new data to assign to clusters.
        :return: Series of predicted cluster labels.
        """
        if self.clusterer is None:
            raise ValueError("Model is not fitted. Call `fit` first.")

        # Scale the data using the same scaler used during training
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Use the cluster membership strength (probabilities) to determine cluster membership
        cluster_probabilities = self.clusterer.approximate_predict(scaled_data)
        cluster_labels = pd.Series(cluster_probabilities, index=data.index, name='Cluster_Label')
        return cluster_labels

    def get_membership_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the membership probabilities for each point belonging to a cluster.

        :param data: DataFrame containing the data to compute membership probabilities.
        :return: DataFrame with membership probabilities for each point.
        """
        if self.clusterer is None:
            raise ValueError("Model is not fitted. Call `fit` first.")

        # Scale the data before computing probabilities
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Return the soft clustering probabilities (membership probabilities)
        probabilities = self.clusterer.membership_vector(scaled_data)
        return pd.DataFrame(probabilities, index=data.index)
