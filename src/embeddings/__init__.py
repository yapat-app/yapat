import glob
import itertools
import logging
import os
from importlib import import_module

import dask
import librosa
import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from extensions import db
from schema_model import Dataset, EmbeddingMethod
# from src import server

logger = logging.getLogger(__name__)


def register_dataset(dataset_name, path_audio, flask_server):
    with flask_server.app_context():
        try:
            db.session.add(Dataset(dataset_name=dataset_name, path_audio=path_audio))
            db.session.commit()
        except SQLAlchemyError as e:
            db.session.rollback()
            logger.exception(e)


def _split_audio_into_chunks(filename: str, chunk_duration: float, sampling_rate: int = None) -> pd.DataFrame:
    """
    Splits an audio file into non-overlapping chunks of specified duration.

    Args:
        filename (str): Path to the audio file.
        chunk_duration (float): Duration of each chunk in seconds.
        sampling_rate (int, optional): Sampling rate to load the audio file. If None, the default sampling rate
            is used by librosa.

    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - 'filename': The chunked filename with the format 'original_filename_starttime_endtime.ext'.
            - 'audio_data': The corresponding chunked audio data.
    """
    # Load the audio file with librosa
    audio, sampling_rate = librosa.load(filename, sr=sampling_rate)

    # Ensure the length is a multiple of the chunk size by truncating the extra samples
    chunk_size = sampling_rate * chunk_duration
    audio = audio[:len(audio) // chunk_size * chunk_size]

    # Split the audio into non-overlapping chunks
    chunked_audio = np.split(audio, len(audio) // chunk_size)

    # Generate filenames for each chunk
    prefix, suffix = filename.rsplit('.', 1)
    indices = [f"{prefix}_{i * chunk_duration}_{(i + 1) * chunk_duration:.0f}.{suffix}" for i in
               range(len(chunked_audio))]

    # Create a DataFrame with chunked filenames and corresponding audio data
    df = pd.DataFrame({"filename": indices, "audio_data": chunked_audio})
    df.set_index("filename", inplace=True)

    return df


class BaseEmbedding:
    """
        Base class for embedding models used to generate embeddings from audio data.

        This class provides core functionality for embedding models, including loading the model,
        reading and processing audio datasets, and optionally using Dask for distributed processing.
        It is meant to be subclassed by specific embedding models, which should implement their own
        model loading and processing logic.

        Attributes:
        -----------
        dataset_name : pd.DataFrame or None
            DataFrame holding the processed audio data (e.g., file paths, audio features).
        dask_client : dask.distributed.client.Client or None
            Optional Dask client for handling distributed task execution.
        embeddings : pd.DataFrame or None
            DataFrame containing the generated embeddings for the audio dataset.

        Methods:
        --------
        load_model():
            Abstract method for loading the model. Must be implemented by subclasses.

        process(audio_files):
            Abstract method for processing audio files to generate embeddings. Must be implemented by subclasses.

        read_audio_dataset(dataset_name: str, extension: str = '.wav', sampling_rate: int = 48000) -> pd.DataFrame:
            Reads and processes an audio dataset, optionally using Dask for parallel processing.
            Returns a pandas DataFrame containing the audio file paths and other metadata.
        """

    def __init__(self, dataset_name: str, dask_client: dask.distributed.client.Client or None = None, flask_server = None) -> None:
        """
        Initialize the BaseEmbedding class with the model path and an optional Dask client.

        :param dataset_name: Name of dataset as stored in Dataset.dataset_name from the database.
        :param dask_client: Optional Dask client for handling distributed task execution.
        :param data: Initialized to None, will hold the processed audio data once read.
        :param embeddings: Initialized to None, will store the generated embeddings.
        """
        self.dataset_name = dataset_name
        self.dask_client = dask_client  # Dask client is used for distributed processing of tasks.
        self.data = None
        self.embeddings = None

    def load_model(self):
        """
        Placeholder method for loading the model. This should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def process(self, audio_files):
        """
        Placeholder method for processing audio files. This should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def read_audio_dataset(self, dataset_name: str, extension: str = '.wav',
                           sampling_rate: int = 48000, flask_server = None) -> pd.DataFrame:
        """
                Read the dataset of audio files, and optionally process it using Dask for parallelization.

                :param dataset_name: Name of the dataset to load from the database.
                :param extension: File extension for audio files to be included (default is '.wav').
                :param sampling_rate: The sampling rate for the audio files (default is 48,000).
                :return: A pandas DataFrame containing audio file paths and any other relevant metadata.
                """
        # Fetch the path to the dataset from the database using a Flask app context.
        with flask_server.app_context():
            path_dataset = db.session.execute(
                db.select(Dataset.path_audio).where(Dataset.dataset_name == dataset_name)).scalar_one_or_none()

        # Recursively find all audio files in the dataset with the specified extension.
        list_of_audio_files = glob.glob(os.path.join(path_dataset, '**', '*.' + extension.lstrip('.')), recursive=True)

        # If a Dask client is provided, parallelize the audio chunking process using Dask.
        if self.dask_client is not None:
            # Use Dask to map audio files to chunking tasks and gather the results.
            dfs_audio = self.dask_client.map(_split_audio_into_chunks, list_of_audio_files,
                                             itertools.repeat(sampling_rate))
            df_audio = pd.concat(self.dask_client.gather(dfs_audio))  # Concatenate the results.
        else:
            # If no Dask client is provided, process the audio files locally.
            dfs_audio = [_split_audio_into_chunks(df_single_file) for df_single_file in list_of_audio_files]
            df_audio = pd.concat(dfs_audio)

        # Return the concatenated DataFrame of processed audio files.
        return df_audio


def compute_embeddings(dataset_name: str, embedding_method: str, flask_server):
    with flask_server.app_context():
        path_audio = db.session.execute(
            db.select(Dataset.path_audio).filter_by(dataset_name=dataset_name)).scalar_one_or_none()
        embedding = db.session.execute(db.select(EmbeddingMethod).filter_by(method_name=embedding_method))
    embedding = import_module(f"embeddings.{embedding}")
    return embedding.fit_transform(path_audio)


def get_embedding_model(method_name : str, dask_client : dask.distributed.client.Client or None = None):
    if method_name == "birdnet":
        from embeddings.birdnet import BirdnetEmbedding
        return BirdnetEmbedding(method_name, dask_client)
    elif method_name == "acoustic_indices":
        from embeddings.acoustic_indices import AcousticIndices
        return AcousticIndices(method_name, dask_client)
    elif method_name == "vae":
        from embeddings.vae import VAEEmbedding
        return VAEEmbedding(method_name, dask_client)
    else:
        raise ValueError(f"Unknown embedding method: {method_name}")