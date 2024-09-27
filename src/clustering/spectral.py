import pandas as pd
from sklearn.cluster import SpectralClustering

from clustering import BaseClustering

class SPECTRALClustering(BaseClustering):
    """

    Methods:
    --------
    fit(data: pd.DataFrame):
        Fit the Spectral clustering algorithm to the data.

    """

    def __init__(self, n_clusters: int = 8):
        """
        Initialize the SpectralClustering class with the minimum cluster size.

        :param n_clusters: The minimum size of clusters. Clusters smaller than this size will be treated as noise.
        """
        super().__init__()
        self.clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42)

    def fit(self, embedding_method_name):
        """
        :param data: DataFrame containing the data to be clustered.
        :return: DataFrame containing the cluster labels assigned to the data.
        """
        data = self.load_data(embedding_method_name)
        self.scaled_data = self.scale_data(data)
        self.clusterer.fit(self.scaled_data)
        self.labels = pd.DataFrame(self.clusterer.labels_, columns=['Cluster Label'], index=data.index)
        self.save_labels('spectral', embedding_method_name, self.labels)
        return
