import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from evaluations import BaseEvaluation
class ExplainedVariance(BaseEvaluation):
    def __init__(self, variance_threshold: float = 0.95):
        """
        Initialize the ExplainedVariance evaluator with a variance threshold.

        Parameters:
        - variance_threshold: float, the cumulative variance threshold to determine
                              the number of principal components to retain.
        """
        super().__init__()
        self.variance_threshold = variance_threshold


    def calculate_result(self, dataset_id: int, embedding_id: int):

        data = self.load_data(dataset_id, embedding_id)
        self.scaled_data = self.scale_data(data)
        pca = PCA(n_components=None)  # Use all components
        pca.fit(self.scaled_data)
        pca_data = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(pca_data)
        num_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        percentage_of_components = (num_components / len(pca_data)) * 100  # Calculate percentage since all embeddings have different dimensionality
        self.evaluation_result = percentage_of_components
        #self.save_result(self.evaluation_result)
        return self.evaluation_result


