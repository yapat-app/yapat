import pandas as pd
from datetime import datetime

from extensions import sqlalchemy_db
from schema_model import EmbeddingResult, ClusteringResult
from sqlalchemy.exc import SQLAlchemyError


class BaseVisualization:


    def __init__(self, result_id) -> None:
        """
        Initialize the BaseClustering class. Sets up the data and labels attributes.
        """
        # The call back args can be changed if needed
        self.result_id = result_id
        self.embeddings = None
        self.cluster_labels = None

    def load_data(self):
        # SQL query to be implemented for embedding_file_path and cluster_file_path
        embedding_file_path = '/Users/ridasaghir/Desktop/exp/anura/anura_encodings.csv'
        cluster_file_path = '/Users/ridasaghir/Desktop/exp/anura/anura_labels.csv'
        self.embeddings = pd.read_csv(embedding_file_path, index_col=0)
        self.cluster_labels = pd.read_csv(cluster_file_path, index_col=0)
        # self.embeddings = pd.read_pickle(embedding_file_path)
        # self.cluster_labels = pd.read_pickle(cluster_file_path)
        return self.embeddings, self.cluster_labels

    def parse_datetime_from_filename(self, filename):
        parts = filename.split('_')
        date_str = parts[1]  # Date part
        time_str = parts[2]  # Time part
        return datetime.strptime(date_str, '%Y%m%d'), datetime.strptime(time_str, '%H%M%S').time()







