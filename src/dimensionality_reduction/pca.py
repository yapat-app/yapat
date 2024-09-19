import pandas as pd
from sklearn.decomposition import PCA

from dimensionality_reduction import BaseDimensionalityReduction


class PCAReduction(BaseDimensionalityReduction):
    """
    PCA (Principal Component Analysis) class for dimensionality reduction, using the sklearn library.

    This class extends the BaseDimensionalityReduction and implements PCA.

    Attributes:
    -----------
    pca : PCA
        The PCA model from sklearn.

    Methods:
    --------
    fit(data: pd.DataFrame):
        Fits the PCA model to the data.

    transform(data: pd.DataFrame) -> pd.DataFrame:
        Transforms the data using the fitted PCA model.
    """

    def __init__(self, n_components: int = 2):
        """
        Initialize the PCAReduction class.

        :param n_components: The number of components for PCA.
        """
        super().__init__(n_components)
        self.pca = PCA(n_components=self.n_components)  # Initialize the PCA model

    def fit(self, data: pd.DataFrame):
        """
        Fit the PCA model to the dataset.

        :param data: DataFrame containing the data to reduce.
        """
        self.pca.fit(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data into the reduced PCA space.

        :param data: DataFrame containing the data to reduce.
        :return: DataFrame with reduced dimensionality.
        """
        pca_transformed = self.pca.transform(data)
        return pd.DataFrame(pca_transformed, index=data.index, columns=[f'PC{i + 1}' for i in range(self.n_components)])
