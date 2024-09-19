import numpy as np
import pandas as pd
from maad import features

from embeddings import BaseEmbedding


def compute_spectral_features(audio_file: np.array, sampling_rate: int or None = None, **kwargs) -> pd.DataFrame:
    """
    Compute all spectral features for a single audio file using the `maad.features.all_spectral_features` method.

    :param audio_file: Path to the audio file to process.
    :param sampling_rate: Sampling rate of the audio file.
    :param kwargs: Additional parameters for the feature computation, such as 'nperseg', 'roi', and 'method'.
    :return: A pandas DataFrame containing all spectral features for the audio file.
    """

    # Compute all spectral features using the maad library's all_spectral_features function.
    spectral_features = features.all_spectral_features(audio_file, sampling_rate, **kwargs)

    # Convert the features into a pandas DataFrame.
    features_df = pd.DataFrame(spectral_features).T

    return features_df


class AcousticIndices(BaseEmbedding):
    """
    AcousticIndices class for computing spectral features from audio data.

    This class extends BaseEmbedding and provides functionality for computing 
    various spectral features using non-machine-learning methods. It computes features
    such as peak frequency, bandwidth, skewness, etc., using the method described.

    Attributes:
    -----------
    model_path : str
        Not used in this class, but inherited from BaseEmbedding for consistency.
    dask_client : dask.distributed.client.Client or None
        Optional Dask client for handling distributed task execution.
    data : pd.DataFrame or None
        DataFrame holding the processed audio data (e.g., file paths, audio features).
    embeddings : pd.DataFrame or None
        DataFrame containing the computed acoustic features for the audio dataset.

    Methods:
    --------
    load_model():
        This method is not applicable for this class, as no model is required.

    process(dataset_name: str, extension: str = '.wav', sampling_rate: int = 48000, **kwargs):
        Processes the audio dataset to compute spectral features.

    compute_spectral_features(audio_file: str, sampling_rate: int = 48000, **kwargs) -> pd.DataFrame:
        Computes all spectral features for a single audio file using the `maad.features.all_spectral_features` method.
    """

    def load_model(self):
        """
        AcousticIndices does not require a machine learning model, so this method is not used.
        """
        pass  # No model loading required for this class.

    def process(self, dataset_name: str, extension: str = '.wav', sampling_rate: int = 48000, **kwargs):
        """
        Processes the dataset by reading the audio files and computing the spectral features.

        :param dataset_name: Name of the dataset to process.
        :param extension: File extension for audio files (default is '.wav').
        :param sampling_rate: Sampling rate for the audio files (default is 48,000).
        :param kwargs: Additional keyword arguments for feature computation (e.g., 'nperseg', 'roi', 'method').
        :return: A pandas DataFrame containing computed acoustic features for each audio file.
        """
        # Load the dataset of audio files into a DataFrame.
        self.data = self.read_audio_dataset(dataset_name, extension, sampling_rate)

        # Initialize a list to store computed features for all audio files.
        all_features = []

        # Loop through each audio file in the dataset and compute spectral features.
        for audio_file in self.data['sound_clip_url']:
            features_df = compute_spectral_features(audio_file, sampling_rate, **kwargs)
            all_features.append(features_df)

        # Concatenate the features for all audio files into a single DataFrame.
        self.embeddings = pd.concat(all_features, axis=0)
        return self.embeddings