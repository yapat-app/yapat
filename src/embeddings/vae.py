import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from scipy.io import wavfile

from embeddings import BaseEmbedding


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

    compute_spectrogram(audio_file: str, sampling_rate: int = 48000, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        Computes a spectrogram from an audio file using the specified parameters.

    build_vae(input_shape: tuple, latent_dim: int = 64) -> tensorflow.keras.Model:
        Builds a VAE model using TensorFlow/Keras.
    """

    def load_model(self):
        """
        Loads a pre-trained VAE model from the model path if it exists.
        """
        try:
            self.vae = tf.keras.models.load_model(self.model_path)
            print(f"Loaded VAE model from {self.model_path}")
        except Exception as e:
            print(f"No pre-trained model found at {self.model_path}. Will build a new one.")
            self.vae = None

    def process(self, dataset_name: str, extension: str = '.wav', sampling_rate: int = 48000, **kwargs):
        """
        Processes the dataset by reading audio files, computing their spectrograms,
        and fitting a VAE to the spectrograms.

        :param dataset_name: Name of the dataset to process.
        :param extension: File extension for audio files (default is '.wav').
        :param sampling_rate: Sampling rate for the audio files (default is 48,000).
        :param kwargs: Additional keyword arguments for VAE training (e.g., 'epochs', 'batch_size').
        :return: None
        """
        # Load the dataset of audio files into a DataFrame
        self.data = self.read_audio_dataset(dataset_name, extension, sampling_rate)

        # Initialize a list to store computed spectrograms
        spectrograms = []

        # Compute spectrogram for each audio file
        for audio_file in self.data['sound_clip_url']:
            spec = self.compute_spectrogram(audio_file, sampling_rate)
            spectrograms.append(spec)

        # Stack all spectrograms into a single numpy array
        spectrograms = np.stack(spectrograms, axis=0)

        # Build or load the VAE model
        input_shape = spectrograms.shape[1:]  # Shape of individual spectrograms
        if self.vae is None:
            self.vae = self.build_vae(input_shape)

        # Train the VAE on the spectrograms
        self.vae.compile(optimizer=Adam(learning_rate=0.001), loss=self.vae_loss)
        self.vae.fit(spectrograms, spectrograms, epochs=kwargs.get('epochs', 50),
                     batch_size=kwargs.get('batch_size', 32))

        # Save the trained VAE model
        self.vae.save(self.model_path)

    def compute_spectrogram(self, audio_file: str, sampling_rate: int = 48000, n_fft: int = 2048,
                            hop_length: int = 512) -> np.ndarray:
        """
        Compute the spectrogram for a single audio file.

        :param audio_file: Path to the audio file.
        :param sampling_rate: Sampling rate for the audio file (default is 48,000 Hz).
        :param n_fft: Number of FFT components (default is 2048).
        :param hop_length: Number of samples between successive frames (default is 512).
        :return: A 2D numpy array representing the computed spectrogram.
        """
        # Load the audio file
        s, fs = librosa.load(audio_file, sr=sampling_rate)

        # Compute the Short-Time Fourier Transform (STFT) spectrogram
        S = librosa.stft(s, n_fft=n_fft, hop_length=hop_length)

        # Convert the complex-valued spectrogram to magnitude
        S_magnitude = np.abs(S)

        # Convert to decibels (logarithmic scale)
        S_db = librosa.amplitude_to_db(S_magnitude, ref=np.max)

        return S_db

    def build_vae(self, input_shape: tuple, latent_dim: int = 64) -> Model:
        """
        Build a Variational Autoencoder (VAE) using TensorFlow/Keras.

        :param input_shape: Shape of the input spectrograms.
        :param latent_dim: Dimensionality of the latent space (default is 64).
        :return: A Keras VAE model.
        """
        # Encoder
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)

        # Latent space
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim))
            return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling)([z_mean, z_log_var])

        # Decoder
        decoder_input = layers.Input(shape=(latent_dim,))
        x = layers.Dense(np.prod(input_shape), activation='relu')(decoder_input)
        x = layers.Reshape(input_shape)(x)
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        outputs = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

        # Full VAE model
        encoder = Model(inputs, z)
        decoder = Model(decoder_input, outputs)
        vae_outputs = decoder(encoder(inputs))
        vae = Model(inputs, vae_outputs)

        return vae

    def vae_loss(self, inputs, outputs):
        """
        VAE loss function, combining reconstruction loss and KL divergence.
        """
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(inputs - outputs))
        kl_loss = -0.5 * tf.keras.backend.sum(
            1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
        return reconstruction_loss + kl_loss
