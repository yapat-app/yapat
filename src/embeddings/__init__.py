import logging
import os
import pathlib
from typing import Optional, Union

import librosa
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import sessionmaker

from schema_model import Dataset
from utils import glob_audio_dataset

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

    # Ensure the length is a multiple of the chunk size by truncating the extra samples
    chunk_size = sampling_rate * chunk_duration

    # Clip audio at a multiple of chunk_size
    audio = audio[:int(len(audio) // chunk_size * chunk_size)]

    # Split the audio into non-overlapping chunks
    chunked_audio = np.split(audio, len(audio) // chunk_size)

    # Generate filenames for each chunk
    prefix, _ = filename.rsplit('.', 1)
    prefix = os.path.basename(prefix)
    t_start = [i * chunk_duration for i in range(len(chunked_audio))]
    t_end = [(i + 1) * chunk_duration for i in range(len(chunked_audio))]

    # Create a DataFrame with chunked filenames and corresponding audio data
    df = pd.DataFrame(
        data={
            "audio_data": chunked_audio,
            "filename": filename,
            "t_start": t_start,
            "t_end": t_end
        },
        index=[f"{prefix}_{i}_{j}" for i, j in zip(t_start, t_end)]
    )
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
            dataset_name: str,
            clip_duration: float = 3.0,
            model_path: Optional[Union[str, pathlib.Path]] = None,
            sampling_rate: Optional[int] = None,
    ) -> None:
        """
        Initialize the BaseEmbedding class with the model path and an optional Dask client.

        :param dataset_name: Name of dataset as stored in Dataset.dataset_name from the database.
        :param clip_duration: Audio files are chunked in clips of duration specified in seconds.
        :param model_path: Path of pre-saved model (if any).
        :param sampling_rate: Sample rate used by `librosa.load`. If None, native sampling rate will be used.
        """

        # Input args
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.sampling_rate = sampling_rate
        self.clip_duration = clip_duration

        # Placeholders
        self.data = pd.DataFrame()  # Initialize as empty DataFrame
        self.embeddings = pd.DataFrame()  # Initialize as empty DataFrame
        self.list_of_audio_files = []
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

    def get_path_dataset(self, url_db: Optional[str] = None):

        # Create a new DB session for the task
        url_db = url_db or 'sqlite:///instance/pipeline_data.db'
        engine = create_engine(url_db)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Fetch the path to the dataset from the database within this session
        try:
            path_dataset = session.execute(
                select(Dataset.path_audio).where(Dataset.dataset_name == self.dataset_name)
            ).scalar_one()
        except NoResultFound:
            raise ValueError(f"No dataset found for name: {self.dataset_name}")
        finally:
            # Close the session after the task is done
            session.close()

        return path_dataset

    def read_audio_dataset(self) -> pd.DataFrame:
        """
        Read the dataset of audio files and process it using Dask for parallelization if available.

                :return: A pandas DataFrame containing audio file paths and any other relevant metadata.
                """

        self.path_dataset = self.get_path_dataset()
        self.list_of_audio_files = glob_audio_dataset(path_dataset=self.path_dataset)

        if not self.list_of_audio_files:
            raise ValueError("No audio files found in the dataset.")

            # if self.dask_client is not None:
            #     dfs_audio = self.dask_client.map(
            #         _split_audio_into_chunks,
            #         self.list_of_audio_files,
            #         [self.clip_duration] * len(self.list_of_audio_files),
            #         [self.sampling_rate] * len(self.list_of_audio_files)
            #     )
            #     self.data = pd.concat(self.dask_client.gather(dfs_audio))
            # else:
            #     # Process audio files locally if no Dask client is provided
            #     with Pool() as pool:
            #         dfs_audio = pool.starmap(
            #             _split_audio_into_chunks,
            #             [(path_audio_file, self.clip_duration, self.sampling_rate) for path_audio_file in
            #              self.list_of_audio_files]
            #         )
        dfs_audio = [_split_audio_into_chunks(path_audio_file, self.clip_duration, self.sampling_rate) for
                     path_audio_file in self.list_of_audio_files]
        self.data = pd.concat(dfs_audio)

        # Return the concatenated DataFrame of processed audio files.
        return self.data


def compute_embeddings(dataset_name: str, embedding_method: str):
    with server.app_context():
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
        from embeddings.birdnet import BirdnetEmbedding
        return BirdnetEmbedding(*args, **kwargs)

    elif method_name == "acoustic_indices":
        from embeddings.acoustic_indices import AcousticIndices
        return AcousticIndices(*args, **kwargs)

    elif method_name == "vae":
        from embeddings.vae import VAEEmbedding
        return VAEEmbedding(*args, **kwargs)

    else:
        raise ValueError(f"Unknown embedding method: {method_name}")
