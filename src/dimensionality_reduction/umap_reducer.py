import pandas as pd
from umap import UMAP
from dimensionality_reduction import BaseDimensionalityReduction

class UmapReducer(BaseDimensionalityReduction):
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

    def fit_transform(self, embedding_method_name):

        data = self.load_data(embedding_method_name)
        print('dataloaded')
        self.scaled_data = self.scale_data(data)
        print('data scaled')
        reduced_data = self.dim_reducer.fit_transform(self.scaled_data)
        # Creating a DataFrame for the reduced data
        columns = [f'UMAP {i + 1}' for i in range(self.n_components)]
        self.transformed_data = pd.DataFrame(reduced_data, columns=columns, index=data.index)
        print('data transformed')
        self.save_transformed_data('umap_reduction', embedding_method_name, self.transformed_data)
        print('data saved to db')
        return

