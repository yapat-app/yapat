import pandas as pd

import umap
from dimensionality_reduction import BaseDimensionalityReduction


class UMAPReduction(BaseDimensionalityReduction):
    """
    UMAP (Uniform Manifold Approximation and Projection) class for dimensionality reduction, using the umap-learn library.

    This class extends the BaseDimensionalityReduction and implements UMAP.

    Attributes:
    -----------
    umap_model : UMAP
        The UMAP model from the umap-learn library.

    Methods:
    --------
    fit(data: pd.DataFrame):
        Fits the UMAP model to the data.

    transform(data: pd.DataFrame) -> pd.DataFrame:
        Transforms the data using the fitted UMAP model.
    """

    def __init__(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1):
        """
        Initialize the UMAPReduction class.

        :param n_components: The number of components for UMAP.
        :param n_neighbors: The number of neighbors to consider for UMAP.
        :param min_dist: The minimum distance between points for UMAP.
        """
        super().__init__(n_components)
        self.umap_model = umap.UMAP(n_components=self.n_components, n_neighbors=n_neighbors, min_dist=min_dist)

    def fit(self, data: pd.DataFrame):
        """
        Fit the UMAP model to the dataset.

        :param data: DataFrame containing the data to reduce.
        """
        self.umap_model.fit(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data into the reduced UMAP space.

        :param data: DataFrame containing the data to reduce.
        :return: DataFrame with reduced dimensionality.
        """
        umap_transformed = self.umap_model.transform(data)
        return pd.DataFrame(umap_transformed, index=data.index,
                            columns=[f'UMAP{i + 1}' for i in range(self.n_components)])
