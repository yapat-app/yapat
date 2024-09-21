import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from scipy.io import wavfile
import dask

from embeddings import BaseEmbedding
from models.vae.autoencoder import VAE


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

    def __init__(self, model_path='../assets/models/vae/autoencoder.py',
                 dask_client: dask.distributed.client.Client or None = None,
                 learning_rate = 0.05, batch_size=16, epochs=1, latent_dim = 60):
        super().__init__(model_path, dask_client)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim

    def _preprocess(self, signal, sample_rate, duration, frame_size, hop_length, min_val, max_val):

        num_expected_samples = int(sample_rate * duration)
        # Pad the signal if necessary
        if len(signal) < num_expected_samples:
            num_missing_samples = num_expected_samples - len(signal)
            signal = np.pad(signal, (0, num_missing_samples), mode='constant')
        stft = librosa.stft(signal, n_fft=frame_size, hop_length=hop_length)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        min_val = log_spectrogram.min()
        max_val = log_spectrogram.max()
        norm_log_spectrogram = (log_spectrogram - min_val) / (max_val - min_val)
        norm_log_spectrogram = norm_log_spectrogram * (max_val - min_val) + min_val
        return norm_log_spectrogram, min_val, max_val

    def load_model(self):
        # Structure for 3 seconds audio files
        autoencoder = VAE(
            input_shape=(256, 282, 1),
            conv_filters=(512, 256, 128, 64, 32),
            conv_kernels=(3, 3, 3, 3, 3),
            conv_strides=(1, 2, 2, 2, 1),
            latent_space_dim=60
        )
        autoencoder.summary()
        autoencoder.compile(self.learning_rate)
        return autoencoder


    def process(self, dataset_name: str, sampling_rate: int = 48000):

        self.data = self.read_audio_dataset(dataset_name, sampling_rate, chunk_duration=3)
        x_train = []
        for row in self.data.iterrows():
            norm_log_spectrogram, min_val, max_val = self._preprocess(row[1].audio_data, sample_rate=sampling_rate, duration=3,
                                                                      frame_size=512,
                                                                      hop_length=512, min_val=0, max_val=1)
            norm_log_spectrogram = norm_log_spectrogram[..., np.newaxis]
            x_train.append(norm_log_spectrogram)

        autoencoder = self.load_model()
        x_train = np.array(x_train)
        autoencoder.train(x_train, self.batch_size, self.epochs)
        autoencoder.save("vae_model")
        encodings, _, _ = autoencoder.encoder.predict(x_train)
        self.embeddings = pd.DataFrame(encodings, index=self.data.index, columns=[f'embedding_{i}' for i in range(self.latent_dim)])
        return self.embeddings

