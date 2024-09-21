import pandas as pd
from sklearn.cluster import KMeans

from clustering import BaseClustering

class KMeansClustering(BaseClustering):
    """
    K-Means clustering algorithm implementation that inherits from BaseClustering.
    """

    def __init__(self, n_clusters=8):
        """
        Initialize the KMeansClustering class with the specified number of clusters.
        :param n_clusters: The number of clusters to find.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.kmeans = None  # Placeholder for the KMeans model.

    def fit(self, data: pd.DataFrame):
        """
        Fit the K-Means clustering algorithm on the data.
        :param data: DataFrame containing the data to cluster.
        """
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(data)
        self.labels = pd.Series(self.kmeans.labels_, name='Cluster_Label')
        return self.labels

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict cluster labels for new data using the fitted K-Means model.
        :param data: DataFrame containing new data to assign to clusters.
        :return: A Series of predicted cluster labels.
        """
        if self.kmeans is None:
            raise ValueError("Model is not fitted. Call `fit` first.")

        cluster_labels = self.kmeans.predict(data)
        return pd.Series(cluster_labels, name='Cluster_Label')
