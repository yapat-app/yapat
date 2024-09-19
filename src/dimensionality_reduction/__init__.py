import pandas as pd
from abc import ABC, abstractmethod


class BaseDimensionalityReduction(ABC):
    """
    Base class for dimensionality reduction models.

    This class provides the structure for dimensionality reduction techniques like PCA, UMAP, etc.
    It defines methods for fitting and transforming data.

    Attributes:
    -----------
    n_components : int
        The number of dimensions to reduce the data to.
    transformed_data : pd.DataFrame or None
        DataFrame containing the reduced dimensionality data.

    Methods:
    --------
    fit(data: pd.DataFrame):
        Abstract method for fitting the dimensionality reduction model. Must be implemented by subclasses.

    transform(data: pd.DataFrame) -> pd.DataFrame:
        Abstract method for transforming data into the reduced dimensionality space. Must be implemented by subclasses.

    fit_transform(data: pd.DataFrame) -> pd.DataFrame:
        Combines fitting and transforming the data into a single method.
    """

    def __init__(self, n_components: int = 2):
        """
        Initialize the BaseDimensionalityReduction class.

        :param n_components: The number of dimensions to reduce the data to.
        """
        self.n_components = n_components
        self.transformed_data = None  # Placeholder for the transformed data

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """
        Abstract method for fitting the dimensionality reduction model on the data.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for transforming data into the reduced dimensionality space.
        Must be implemented by subclasses.
        """
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the dimensionality reduction model on the data and returns the reduced data.

        :param data: DataFrame containing the data to reduce.
        :return: DataFrame with reduced dimensionality data.
        """
        self.fit(data)
        self.transformed_data = self.transform(data)
        return self.transformed_data
