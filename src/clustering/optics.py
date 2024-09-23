import pandas as pd
from sklearn.cluster import OPTICS

from clustering import BaseClustering

class OpticsClustering(BaseClustering):
    """

    Methods:
    --------
    fit(data: pd.DataFrame):
        Fit the Optics clustering algorithm to the data.

    """

    def __init__(self):
        """
        Initialize the OpticsClustering class with the minimum cluster size.

        """
        super().__init__()
        self.clusterer = OPTICS() # min_samples default = 5

    def fit(self, dataset_id: int, embedding_id: int):
        """
        :param data: DataFrame containing the data to be clustered.
        :return: DataFrame containing the cluster labels assigned to the data.
        """
        data = self.load_data(dataset_id, embedding_id)
        self.scaled_data = self.scale_data(data)
        self.clusterer.fit(self.scaled_data)
        self.labels = pd.DataFrame(self.clusterer.labels_, columns=['Cluster Label'], index=data.index)
        #self.save_labels()
        return self.labels







