import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from npeet.entropy_estimators import entropy as calculate_entropy

from evaluations import BaseEvaluation


class EmbeddingsEvaluation(BaseEvaluation):
    def __init__(self, variance_threshold: float = 0.95):
        """
        Initialize the EmbeddingsEvaluation with a variance threshold for PCA.
        Parameters:
        - variance_threshold: float, the cumulative variance threshold to determine
                              the percentage of principal components to retain.
        """
        super().__init__()
        self.variance_threshold = variance_threshold

    def calculate_entropy(self, data):
        return calculate_entropy(data)

    def calculate_explained_variance(self, data):
        pca = PCA(n_components=None)  # Use all components
        pca.fit(data)
        pca_data = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(pca_data)
        num_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        percentage_of_components = (num_components / len(pca_data)) * 100
        return percentage_of_components

    def evaluate(self, dataset_id: int, embedding_id: int):

        data = self.load_data(dataset_id, embedding_id)
        self.scaled_data = self.scale_data(data)
        entropy_result = self.calculate_entropy(self.scaled_data)
        explained_variance_result = self.calculate_explained_variance(self.scaled_data)
        results = {
            "Entropy": entropy_result,
            "Explained Variance": explained_variance_result
        }
        # Will work on it later
        # self.save_results()
        return results
