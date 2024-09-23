import pandas as pd
from umap_reduction import UMAP
from dimensionality_reduction import BaseDimensionalityReduction

class Umap(BaseDimensionalityReduction):
    """
    Class for reducing dimensionality using UMAP.

    Attributes:
    -----------
    n_components : int
        The number of dimensions to reduce the data to.

    Methods:
    --------
    fit_transform(data: pd.DataFrame) -> pd.DataFrame:
        Fit the UMAP model to the data and transform it.
    """

    def __init__(self, n_components: int = 3):
        """
        Initialize the Umap class with the specified number of components for UMAP.

        :param n_components: The number of components for UMAP.
        """
        super().__init__()
        self.n_components = n_components
        self.dim_reducer = UMAP(n_components=self.n_components)

    def fit_transform(self, dataset_id: int, embedding_id: int):
        """
        Fit the UMAP model to the dataset and transform the dataset.

        :param dataset_id: ID for the dataset.
        :param embedding_id: ID for the embedding.
        :return: DataFrame with the reduced dimensions.
        """
        data = self.load_data(dataset_id, embedding_id)
        self.scaled_data = self.scale_data(data)
        reduced_data = self.dim_reducer.fit_transform(self.scaled_data)
        # Creating a DataFrame for the reduced data
        columns = [f'UMAP {i + 1}' for i in range(self.n_components)]
        self.transformed_data = pd.DataFrame(reduced_data, columns=columns, index=data.index)
        #self.save_transformed_data(file_path)
        return self.transformed_data

