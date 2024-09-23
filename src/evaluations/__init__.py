import pandas as pd

import dask
from sklearn.preprocessing import StandardScaler

from sqlalchemy.exc import SQLAlchemyError

from extensions import sqlalchemy_db
from schema_model import Dataset, EmbeddingResult

class BaseEvaluation:
    def __init__(self):
        self.data = None  # Will hold the data to be clustered.
        self.evaluation_result = None

    def load_data(self, dataset_id: int, embedding_id: int) -> pd.DataFrame:
            """
            Load the data to be clustered from a CSV or pickle file.

            :param file_path: Path to the data file (CSV or pickle format).
            :return: DataFrame containing the loaded data.
            """
            embedding_result = sqlalchemy_db.session.query(EmbeddingResult).filter_by(
                dataset_id=dataset_id,
                embedding_id=embedding_id
            ).one_or_none()
            file_path =  embedding_result.file_path

            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path, index_col=0)
            elif file_path.endswith('.pkl'):
                self.data = pd.read_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}. Please use CSV or pickle.")
            return self.data

    def scale_data(self, data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.values)
        return scaled_data

    def save_result(self, evaluation_result):
        pass


def get_evaluation_method(method_name: str, dask_client: dask.distributed.client.Client or None = None):
    if method_name == "entropy":
        from evaluations.entropy_evaluation import EntropyEvaluation
        return EntropyEvaluation()
    elif method_name == "explained_variance":
        from evaluations.explained_variance_evaluation import ExplainedVariance
        return ExplainedVariance()
    else:
        raise ValueError(f"Unknown Evaluation Method: {method_name}")



