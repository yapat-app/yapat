from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN

from evaluations import BaseEvaluation

class ClusteringEvaluation(BaseEvaluation):
    def __init__(self):
        """
                Initialize the ClusteringEvaluation.
        """
        super().__init__()

    def sil_score(self, embeddings, cluster_labels):
        return silhouette_score(embeddings, cluster_labels) # Silhouette Score

    def db_score(self, embeddings, cluster_labels): # Davies Bouldin Score
        return davies_bouldin_score(embeddings, cluster_labels)

    def evaluate(self, dataset_id: int, embedding_id: int):
        # using dataset_id and embedding_id assuming that evaluation would be called for every clustering result
        data = self.load_data('embeddings', dataset_id, embedding_id)
        embedding_columns = [col for col in data.columns if col != 'Cluster Label']
        sil_score = self.sil_score(data[embedding_columns], data['Cluster Label'])
        db_score = self.db_score(data[embedding_columns], data['Cluster Label'])
        evaluation_results = {
            "Silhouette Score": sil_score,
            "Davies Bouldin Score": db_score
        }
        #self.save_results('clustering', evaluation_results, dataset_id, embedding_id)
        return evaluation_results







