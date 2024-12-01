from sklearn.cluster import AffinityPropagation
import pandas as pd

from src.clustering import BaseClustering


class Affinity(BaseClustering):
    """
    Methods:
    --------
    fit(data: pd.DataFrame):
        Fit the Affinity Propagation clustering algorithm to the data.

    """

    def __init__(self):
        """
        Initialize the AffinityClustering class with the minimum cluster size.

        :param min_cluster_size: The minimum size of clusters. Clusters smaller than this size will be treated as noise.
        """
        super().__init__()
        self.clusterer = AffinityPropagation(random_state=43)

    def fit(self, embedding_method_name):
        """
        :param data: DataFrame containing the data to be clustered.
        :return: DataFrame containing the cluster labels assigned to the data.
        """
        data = self.load_data(embedding_method_name)
        self.scaled_data = self.scale_data(data)
        self.clusterer.fit(self.scaled_data)
        self.labels = pd.DataFrame(self.clusterer.labels_, columns=['Cluster Label'], index=data.index)
        self.save_labels('affinity', embedding_method_name, self.labels)
        return
