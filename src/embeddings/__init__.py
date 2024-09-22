import glob
import logging
import os
import pathlib
from importlib import import_module
from multiprocessing import Pool

import dask.distributed
import librosa
import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from extensions import sqlalchemy_db
from schema_model import Dataset, EmbeddingMethod

logger = logging.getLogger(__name__)


def register_dataset(dataset_name, path_audio, flask_server):
    with flask_server.app_context():
        try:
            sqlalchemy_db.session.add(Dataset(dataset_name=dataset_name, path_audio=path_audio))
            sqlalchemy_db.session.commit()
        except SQLAlchemyError as e:
            sqlalchemy_db.session.rollback()
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
    total_duration = len(audio) / sampling_rate
    # Checks for pattern like anura data set
    pattern_with_seconds = r'.*_\d{6,8}_\d{6}_\d{1,3}_\d{1,3}\.\w{3}'

    if abs(total_duration - chunk_duration) < 0.1:
        if re.match(pattern_with_seconds, filename):
            a = 'not identified'
            return pd.DataFrame({"filename": [filename], "audio_data":[audio]}).set_index("filename")
    else:
        chunk_size = sampling_rate * chunk_duration
        remainder = len(audio) % chunk_size
        if remainder != 0:
            audio = audio[:-remainder]
        chunked_audio = np.split(audio, len(audio) // chunk_size)
        prefix, suffix = filename.rsplit('.', 1)
        indices = [f"{prefix}_{i * chunk_duration}_{(i + 1) * chunk_duration:.0f}.{suffix}" for i in
                   range(len(chunked_audio))]
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
            :raises NotImplementedError: This method must be implemented by subclasses for model loading.


        process(audio_files):
            Abstract method for processing audio files to generate embeddings. Must be implemented by subclasses.
            :raises NotImplementedError: This method must be implemented by subclasses for model loading.


        read_audio_dataset() -> pd.DataFrame:
            Reads and processes an audio dataset, optionally using Dask for parallel processing.
            Returns a pandas DataFrame containing the audio file paths and other metadata.
            :return: A pandas DataFrame indexed by 'filename' and a data column 'audio_data' containing processed audio chunks.
        """

    def __init__(self, model_path: str, dataset_name: str, dask_client: dask.distributed.client.Client or None = None,
                 flask_server=None) -> None:
        """
        Initialize the BaseEmbedding class with the model path and an optional Dask client.

        :param dataset_name: Name of dataset as stored in Dataset.dataset_name from the database.
        :param clip_duration: Audio files are chunked in clips of duration specified in seconds.
        :param model_path: Path of pre-saved model (if any)
        :param sampling_rate: Sample rate used by `librosa.load`. If None, native sampling rate will be used.
        :param dask_client: Optional Dask client for handling distributed task execution.
        """
        self.dataset_name = dataset_name
        self.dask_client = dask_client  # Dask client is used for distributed processing of tasks.
        self.data = None
        self.embeddings = None
        self.model_path = model_path

    def load_model(self):
        """
        Placeholder method for loading the model. This should be implemented by subclasses if needed.
        """
        if self.model_path:
            raise NotImplementedError("Subclasses should implement this method if a model path is provided.")
        else:
            pass
        #raise NotImplementedError("This method should be implemented by subclasses")

    def process(self, audio_files):
        """
        Placeholder method for processing audio files. This should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def read_audio_dataset(self, dataset_name: str,
                           sampling_rate: int = 48000, chunk_duration : float=3, flask_server=None) -> pd.DataFrame:
        """
                Read the dataset of audio files, and optionally process it using Dask for parallelization.

                :param flask_server:
                :param chunk_duration:
                :param dataset_name: Name of the dataset to load from the database.
                :param extension: File extension for audio files to be included (default is '.wav').
                :param sampling_rate: The sampling rate for the audio files (default is 48,000).
                :return: A pandas DataFrame containing audio file paths and any other relevant metadata.
                """
        # For standalone testing without flask. Will remove after embeddings test
        if flask_server:
            with flask_server.app_context():
                path_dataset = sqlalchemy_db.session.execute(sqlalchemy_db.select(Dataset.path_audio).where(
                    Dataset.dataset_name == dataset_name)).scalar_one_or_none()
        else:
            path_dataset = '/Users/ridasaghir/Desktop/data/anura_subset'
        # Fetch the path to the dataset from the database using a Flask app context.
        # with flask_server.app_context():
        #     path_dataset = sqlalchemy_db.session.execute(sqlalchemy_db.select(Dataset.path_audio).where(
        #         Dataset.dataset_name == dataset_name)).scalar_one_or_none()
        extensions = ['wav', 'aac', 'm4a', 'flac', 'mp3']
        list_of_audio_files = []
        for extension in extensions:
            audio_files = glob.glob(os.path.join(path_dataset, '**', '*.' + extension.lstrip('.')),
                                            recursive=True)
            list_of_audio_files.extend(audio_files)

        # If a Dask client is provided, parallelize the audio chunking process using Dask.
        if self.dask_client is not None:
            # Use Dask to map audio files to chunking tasks and gather the results.
            dfs_audio = self.dask_client.map(
                _split_audio_into_chunks,
                self.list_of_audio_files,
                [self.clip_duration] * len(self.list_of_audio_files),  # Repeat clip_duration for each file
                [self.sampling_rate] * len(self.list_of_audio_files)  # Repeat sampling_rate for each file
            )
            self.data = pd.concat(self.dask_client.gather(dfs_audio))  # Concatenate the results.
        else:
            # If no Dask client is provided, process the audio files locally.
            with Pool() as pool:
                # Use multiprocessing to process audio files in parallel
                dfs_audio = pool.starmap(_split_audio_into_chunks,
                                         [(path_audio_file, self.clip_duration, self.sampling_rate)
                                          for path_audio_file in self.list_of_audio_files])
            self.data = pd.concat(dfs_audio)

        # Return the concatenated DataFrame of processed audio files.
        return df_audio


def compute_embeddings(dataset_name: str, embedding_method: str, flask_server):
    with flask_server.app_context():
        path_audio = sqlalchemy_db.session.execute(
            sqlalchemy_db.select(Dataset.path_audio).filter_by(dataset_name=dataset_name)).scalar_one_or_none()
        embedding = sqlalchemy_db.session.execute(
            sqlalchemy_db.select(EmbeddingMethod).filter_by(method_name=embedding_method))
    embedding = import_module(f"embeddings.{embedding}")
    return embedding.fit_transform(path_audio)


def get_embedding_model(method_name: str, dask_client: dask.distributed.client.Client or None = None):
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
