from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.preprocessing import StandardScaler

from clustering import BaseClustering

class DBSCANClustering(BaseClustering):
    """

    Methods:
    --------
    fit(data: pd.DataFrame):
        Fit the DBSCAN clustering algorithm to the data.

    """

    def __init__(self):
        """
        Initialize the HDBSCANClustering class with the minimum cluster size.

        :param min_cluster_size: The minimum size of clusters. Clusters smaller than this size will be treated as noise.
        """
        super().__init__()
        self.clusterer = DBSCAN(eps=0.001)

    def fit(self, dataset_id: int, embedding_id: int):
        """
        :param data: DataFrame containing the data to be clustered.
        :return: DataFrame containing the cluster labels assigned to the data.
        """
        data = self.load_data(dataset_id, embedding_id)

        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(data.values)

        self.clusterer.fit(self.scaled_data)
        self.labels = pd.Series(self.clusterer.labels_, index=data.index, name='Cluster_Label')
        #self.save_labels()
        return self.labels




