import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from npeet.entropy_estimators import entropy as calculate_entropy

from evaluations import BaseEvaluation


class EmbeddingsEvaluation(BaseEvaluation):
    def __init__(self, variance_threshold: float = 0.95, to_predict:str = 'Hour', max_iter:int = 50000, C: float = 0.1,
                 gamma:float = 0.01, kernel:str = 'linear' ):
        """
                Initialize the EmbeddingsEvaluation with parameters for PCA and SVC.
                Parameters:
                - variance_threshold: float, the cumulative variance threshold for PCA.
                - max_iter: int, maximum iterations for the SVC.
                - C: float, regularization parameter for the SVC.
                - gamma: float, kernel coefficient for the SVC.
                - kernel: str, type of kernel to be used in SVC.
                """
        super().__init__()
        self.variance_threshold = variance_threshold # For explained variance
        self.to_predict = to_predict
        self.model = SVC(random_state=42, max_iter=max_iter, C=C, gamma=gamma, kernel=kernel) # For classifier

    def evaluate_classifier(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted')
        }
        return results

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
        #classifier_results = self.evaluate_classifier(self.scaled_data, y)
        results = {
            "Entropy": entropy_result,
            "Explained Variance": explained_variance_result,
           # **classifier_results
        }
        # self.save_results()
        return results
