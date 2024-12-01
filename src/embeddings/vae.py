import os
import pathlib
import datetime
from multiprocessing import Pool

import dask
import librosa.feature
import numpy as np
import pandas as pd
import tensorflow as tf

from src.assets.models.vae_xprize import VAE
from src.embeddings import BaseEmbedding


class VAEEmbedding(BaseEmbedding):
    """
    VAEEmbedding class for computing spectrograms from audio data and fitting a Variational Autoencoder (VAE).

    This class extends BaseEmbedding and provides functionality for computing spectrograms
    and training a VAE on those spectrograms.

    Attributes:
    -----------
    model_path : str
        Path where the VAE model will be saved or loaded.
    dask_client : dask.distributed.client.Client or None
        Optional Dask client for handling distributed task execution.
    data : pd.DataFrame or None
        DataFrame holding the computed spectrograms.
    vae : tensorflow.keras.Model
        The VAE model used to fit the spectrogram data.

    Methods:
    --------
    load_model():
        Loads a pre-trained VAE model if available.

    process(dataset_name: str, extension: str = '.wav', sampling_rate: int = 48000, **kwargs):
        Processes the audio dataset by computing spectrograms and fitting a VAE.

    """

    def __init__(
            self,
            dataset_name: str,
            clip_duration: float = 3.0,
            model_path: str or pathlib.Path or None = None,
            sampling_rate: int or None = None,
            dask_client: dask.distributed.client.Client or None = None,
            learning_rate=0.05,
            batch_size=16,
            epochs=10,
            latent_dim=128,
            beta_kl=1,
            kw_spectrograms: dict or None = None
    ):
        super().__init__(clip_duration, model_path, sampling_rate, dask_client)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.kw_spectrograms = kw_spectrograms or {
            "resolution": 0.05,
            "overlap": 0.5,
            "freq_min": 10,
            "freq_max": 22050,
            "n_freqs": 400
        }

        self.model = None
        self.spectrograms = None

        if not np.isclose(self.clip_duration, 3.0):
            raise ValueError(f"VAE model expects 3.0-s long clips. Got {self.clip_duration} s")

    def compute_spectrograms(self):
        if self.data is None:
            self.read_audio_dataset()
        # If a Dask client is provided, parallelize the audio chunking process using Dask.
        if self.dask_client is not None:
            # Use Dask to map audio files to chunking tasks and gather the results.

            n_fft = int(self.kw_spectrograms.get('resolution') * self.sampling_rate)
            hop_length = int(n_fft * (1 - self.kw_spectrograms.get('overlap')))

            task_spectrograms = self.dask_client.map(
                _compute_spectrogram,
                self.data['audio_data'],
                [self.sampling_rate] * len(self.list_of_audio_files),
                [self.kw_spectrograms.get('resolution')] * len(self.list_of_audio_files),
                [self.kw_spectrograms.get('overlap')] * len(self.list_of_audio_files),
                [self.kw_spectrograms.get('freq_min')] * len(self.list_of_audio_files),
                [self.kw_spectrograms.get('freq_max')] * len(self.list_of_audio_files),
                [self.kw_spectrograms.get('n_freqs')] * len(self.list_of_audio_files),
            )
            self.spectrograms = self.dask_client.gather(task_spectrograms)  # Concatenate the results.
        else:
            # If no Dask client is provided, process the audio files locally.
            # TODO Use tensorflow, or at least ensure it scales beyond memory capacity
            with Pool() as pool:
                # Use multiprocessing to process audio files in parallel
                spectrograms = pool.starmap(
                    _compute_spectrogram,
                    [(audio_data, self.sampling_rate, self.kw_spectrograms.get('resolution'),
                      self.kw_spectrograms.get('overlap'), self.kw_spectrograms.get('freq_min'),
                      self.kw_spectrograms.get('freq_max'), self.kw_spectrograms.get('n_freqs')) for audio_data in
                     self.data['audio_data']]
                )
            self.spectrograms = spectrograms

    def load_model(self):
        input_shape = (self.kw_spectrograms.get('n_freqs'), int(1 + self.clip_duration / (
                self.kw_spectrograms.get('resolution') * (1 - self.kw_spectrograms.get('overlap')))), 1)
        self.model = VAE(input_shape=input_shape, latent_dim=self.latent_dim, beta_kl=self.beta_kl)
        self.model.compile(tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

    def train_model(self):
        if self.model is None:
            self.load_model()
        os.makedirs(os.path.join('instance', 'tensorboard'), exist_ok=True)
        model_id_string = f"VAE_dim{self.latent_dim:03d}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ds_train = tf.data.Dataset.from_tensor_slices(self.spectrograms)
        ds_train = tf.data.Dataset.zip((ds_train, ds_train))
        ds_train = ds_train.batch(self.batch_size).prefetch(buffer_size=self.batch_size)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='reconstruction_loss', patience=3,
                                                          restore_best_weights=True, mode="min")

        history = self.model.fit(ds_train, epochs=self.epochs, verbose=2, callbacks=[early_stopping])
        return history

    def process(self):
        if self.data is None:
            self.data = self.read_audio_dataset()
        if self.spectrograms is None:
            self.compute_spectrograms()
        if self.model is None:
            self.load_model()
        history = self.train_model()
        ds_test = tf.data.Dataset.from_tensor_slices(self.spectrograms)
        embeddings, _, _ = self.model.encoder.predict(ds_test)
        self.embeddings = pd.DataFrame(embeddings, index=self.data.index,
                                       columns=[f'embedding_dim_{i}' for i in range(self.latent_dim)])
        return self.embeddings


def _compute_spectrogram(
        audio: np.array,
        sample_rate: int,
        resolution: float,
        overlap: float,
        freq_min: float,
        freq_max: float or None,
        n_freqs: int,
        **kwargs
):
    n_fft = int(resolution * sample_rate)
    hop_length = int(n_fft * (1 - overlap))

    # Compute the magnitude spectrogram
    S = np.abs(librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length))

    # Convert the spectrogram to the mel scale
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_freqs, fmin=freq_min, fmax=freq_max)

    # Apply the mel filter bank
    S_mel = np.dot(mel_basis, S)

    # Convert the mel spectrogram to dB
    S_mel_db = librosa.amplitude_to_db(S_mel)

    # Apply robust scaling
    S_mel_db = (S_mel_db - np.percentile(S_mel_db, 50)) / (
            np.percentile(S_mel_db, 75) - np.percentile(S_mel_db, 25))  # Robust scaling

    S_mel_db = np.expand_dims(S_mel_db, axis=2)

    return S_mel_db
