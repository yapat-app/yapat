import pandas as pd
import json

import dask
from sklearn.preprocessing import StandardScaler

from sqlalchemy.exc import SQLAlchemyError

from extensions import sqlalchemy_db
from schema_model import Dataset, EmbeddingResult

class BaseEvaluation:
    def __init__(self):
        self.data = None  # Will hold the data to be clustered.
        self.evaluation_result = None

    def load_data(self, result_indicator: str, dataset_id: int, embedding_id: int) -> pd.DataFrame:
            """
            Load the data to be clustered from a CSV or pickle file.

            :result_indicator: tells if the data to be loaded are embeddings or clustering
            :param dataset_id: dataset_id in embedding results
            :param embedding_id: embedding_id in embedding results.


            """
            if result_indicator == 'embeddings':
                # embedding_result = sqlalchemy_db.session.query(EmbeddingResult).filter_by(
                #     dataset_id=dataset_id,
                #     embedding_id=embedding_id
                # ).one_or_none()
                # file_path =  embedding_result.file_path
                file_path = '/Users/ridasaghir/Desktop/exp/anura/anura_encodings_INCT17.pkl'
                if file_path.endswith('.csv'):
                    self.data = pd.read_csv(file_path, index_col=0)
                elif file_path.endswith('.pkl'):
                    self.data = pd.read_pickle(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_path}. Please use CSV or pickle.")
                return self.data


            elif result_indicator == 'clustering':
                # This should give embeddings and cluster labels
                pass






    def scale_data(self, data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.values)
        return scaled_data

    def save_results(self, results_indicator: str, evaluation_results, dataset_id: int, embedding_id: int):
        """
        Save the evaluation results to a JSON file.
        Parameters:
        - results_indicator: 'clustering' or 'embeddings' to determine right table
        - evaluation_results: dict, the results to be saved.
        - dataset_id : dataset_id value in the EmbeddingResult table
        - embedding_id : embedding_id value in the EmbeddingResult table
        """
        if results_indicator == 'embeddings':
            result = sqlalchemy_db.query(EmbeddingResult).filter_by(dataset_id=dataset_id, embedding_id=embedding_id).first()
            result.evaluation_results = json.dumps(evaluation_results)
            sqlalchemy_db.session.commit()
            sqlalchemy_db.session.close()
        elif results_indicator == 'clustering':
            # to be discussed
            pass









