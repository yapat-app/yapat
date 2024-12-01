from sklearn.cluster import KMeans
import pandas as pd

from src.clustering import BaseClustering


class KMeansClustering(BaseClustering):
    """

    Methods:
    --------
    fit(data: pd.DataFrame):
        Fit the KMeans clustering algorithm to the data.

    """

    def __init__(self, n_clusters: int = 8):

        super().__init__()
        self.clusterer = KMeans(random_state=42, n_clusters=n_clusters)

    def fit(self, embedding_method_name):
        """
        :param data: DataFrame containing the data to be clustered.
        :return: DataFrame containing the cluster labels assigned to the data.
        """
        data = self.load_data(embedding_method_name)
        self.scaled_data = self.scale_data(data)
        self.clusterer.fit(self.scaled_data)
        self.labels = pd.DataFrame(self.clusterer.labels_, columns=['Cluster Label'], index=data.index)
        self.save_labels('kmeans', embedding_method_name, self.labels)
        return












