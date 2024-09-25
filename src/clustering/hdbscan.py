from sklearn.cluster import HDBSCAN
import pandas as pd

from clustering import BaseClustering

class HDBSCANClustering(BaseClustering):
    """

    Methods:
    --------
    fit(data: pd.DataFrame):
        Fit the HDBSCAN clustering algorithm to the data.
    """

    def __init__(self, dataset_name, embedding_method):
        """
        Initialize the HDBSCANClustering class with the minimum cluster size.

        :param min_cluster_size: The minimum size of clusters. Clusters smaller than this size will be treated as noise.
        """
        super().__init__(dataset_name, embedding_method)
        self.clusterer = HDBSCAN()

    def fit(self):
        """
        :param data: DataFrame containing the data to be clustered.
        :return: DataFrame containing the cluster labels assigned to the data.
        """
        data = self.load_data()
        self.scaled_data = self.scale_data(data)
        self.clusterer.fit(self.scaled_data)
        self.labels = pd.DataFrame(self.clusterer.labels_, columns=['Cluster Label'], index=data.index)
        self.save_labels(self.labels)
        return self.labels




