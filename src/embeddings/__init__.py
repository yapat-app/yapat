import glob
import logging
import os
import pathlib
from importlib import import_module
from multiprocessing import Pool
import re

import dask.distributed
import librosa
import numpy as np
import pandas as pd
# from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
# from sqlalchemy.orm import sessionmaker
from src import sqlalchemy_db, server

#from src.extensions import sqlalchemy_db
from src.schema_model import Dataset, EmbeddingMethod, EmbeddingResult

logger = logging.getLogger(__name__)


def _split_audio_into_chunks(filename: str, chunk_duration: float, sampling_rate: int or None = None) -> pd.DataFrame:
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
            return pd.DataFrame({"filename": [filename], "audio_data": [audio]}).set_index("filename")
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

    def __init__(
            self,
            clip_duration: float = 3.0,
            model_path: str or pathlib.Path or None = None,
            sampling_rate: int or None = None,
            dask_client: dask.distributed.client.Client or None = None
    ) -> None:
        """
        Initialize the BaseEmbedding class with the model path and an optional Dask client.

        :param dataset_name: Name of dataset as stored in Dataset.dataset_name from the database.
        :param clip_duration: Audio files are chunked in clips of duration specified in seconds.
        :param model_path: Path of pre-saved model (if any)
        :param sampling_rate: Sample rate used by `librosa.load`. If None, native sampling rate will be used.
        :param dask_client: Optional Dask client for handling distributed task execution.
        """

        # Input args
        self.model_path = model_path
        self.sampling_rate = sampling_rate
        self.clip_duration = clip_duration
        self.dask_client = dask_client  # Dask client is used for distributed processing of tasks.

        # Placeholders
        self.data = None
        self.embeddings = None
        self.list_of_audio_files = None
        self.path_dataset = None

    def load_model(self):
        """
        Placeholder method for loading the model. This should be implemented by subclasses if needed.
        """
        if self.model_path:
            raise NotImplementedError("Subclasses should implement this method if a model path is provided.")
        else:
            pass

    def process(self):
        """
        Placeholder method for processing audio files. This should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_path_dataset(self):
        session = sqlalchemy_db.session
        try:
            with server.app_context():
                selected_dataset = session.query(Dataset).filter_by(is_selected=True).first()
                if not selected_dataset:
                    # implement handling
                    logger.warning("No dataset is currently selected")
                    return None
                return selected_dataset.path_audio
        except SQLAlchemyError as e:
            # Handle and log database errors
            logger.error(f"Error fetching selected dataset from the database: {e}")
            return None

    def read_audio_dataset(self) -> pd.DataFrame:
        """
                Read the dataset of audio files, and optionally process it using Dask for parallelization.

                :return: A pandas DataFrame containing audio file paths and any other relevant metadata.
                """
        self.path_dataset = self.get_path_dataset()
        extensions = ['wav', 'aac', 'm4a', 'flac', 'mp3']
        extensions += [ext.upper() for ext in extensions]
        self.list_of_audio_files = []
        for extension in extensions:
            audio_files = glob.glob(os.path.join(self.path_dataset, '**', '*.' + extension), recursive=True)
            self.list_of_audio_files.extend(audio_files)

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
        return self.data

    def save_embeddings(self, embedding_method_name:str, embeddings):
        if embeddings is None:
            logger.warning("No embeddings available to save")
            return
        try:
            with server.app_context():
                # It should be handled if the dataset is changed while the emebedding are calculated
                selected_dataset = sqlalchemy_db.session.query(Dataset).filter_by(is_selected=True).first()
                if not selected_dataset:
                    logger.warning("No dataset is currently selected. Cannot save embeddings.")
                    return
                embedding_method = sqlalchemy_db.session.query(EmbeddingMethod).filter_by(
                    method_name=embedding_method_name).first()
                if not embedding_method:
                    logger.error(f"Embedding method '{embedding_method_name}' not found in the database.")
                    return

                existing_entry = sqlalchemy_db.session.query(EmbeddingResult).filter_by(
                    dataset_id=selected_dataset.id,
                    embedding_id=embedding_method.id
                ).first()

                if existing_entry:
                    logger.warning(
                        f"Embeddings for dataset ID {selected_dataset.id} and embedding method ID {embedding_method.id} already exist. Skipping save.")
                    return

                os.makedirs('results', exist_ok=True)
                embedding_file_path = os.path.join('results',
                                           f"{selected_dataset.dataset_name}_{embedding_method_name}_embeddings.pkl")
                embeddings.to_pickle(embedding_file_path)
                self._save_embedding_metadata_to_db(
                    dataset_id=selected_dataset.id,
                    embedding_id=embedding_method.id,
                    file_path=embedding_file_path
                )
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return

    def _save_embedding_metadata_to_db(self, dataset_id: int, embedding_id: int, file_path: str) -> None:

        try:
            # Add metadata to the EmbeddingResult table
            embedding_result = EmbeddingResult(
                dataset_id=dataset_id,
                embedding_id=embedding_id,
                file_path=file_path,
                hyperparameters={},  # Optionally add hyperparameters here
                evaluation_results={},  # Optionally add evaluation results here
                created_at=pd.Timestamp.now(),
                task='completed'
            )
            sqlalchemy_db.session.add(embedding_result)
            sqlalchemy_db.session.commit()
            logger.info(f"Embedding metadata saved to the database for dataset {dataset_id}")

        except SQLAlchemyError as e:
            sqlalchemy_db.session.rollback()
            logger.error(f"Error saving embedding metadata to the database: {e}")


def compute_embeddings(dataset_name: str, embedding_method: str, flask_server):
    with flask_server.app_context():
        path_audio = sqlalchemy_db.session.execute(
            sqlalchemy_db.select(Dataset.path_audio).filter_by(dataset_name=dataset_name)).scalar_one_or_none()
        embedding = sqlalchemy_db.session.execute(
            sqlalchemy_db.select(EmbeddingMethod).filter_by(method_name=embedding_method))
    embedding = import_module(f"embeddings.{embedding}")
    return embedding.fit_transform(path_audio)


def get_embedding_model(method_name, *args, **kwargs):
    """
    Retrieve the appropriate embedding model based on the given method name.

    Parameters:
    - method_name (str): The name of the embedding method to use.
    - *args: Additional positional arguments to pass to the embedding model.
    - **kwargs: Additional keyword arguments to pass to the embedding model.

    Returns:
    - An instance of the specified embedding model.

    Raises:
    - ValueError: If the provided method_name is not recognized.
    """
    if method_name == "birdnet":
        from src.embeddings.birdnet import BirdnetEmbedding
        return BirdnetEmbedding(*args, **kwargs)

    elif method_name == "acoustic_indices":
        from src.embeddings.acoustic_indices import AcousticIndices
        return AcousticIndices(*args, **kwargs)

    elif method_name == "vae":
        from src.embeddings.vae import VAEEmbedding
        return VAEEmbedding(*args, **kwargs)

    else:
        raise ValueError(f"Unknown embedding method: {method_name}")

