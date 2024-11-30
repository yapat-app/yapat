import logging
import os
from functools import wraps

import librosa
import tensorflow as tf
from dash import dcc
from flask_login import current_user

logger = logging.getLogger(__name__)


def login_required_layout(layout_func):
    @wraps(layout_func)
    def decorated_layout(*args, **kwargs):
        if not current_user.is_authenticated:
            return dcc.Location(pathname='/login', id='redirect-login')
        return layout_func(*args, **kwargs)

    return decorated_layout


def get_list_files(input_dir):
    list_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg')):
                input_file_path = os.path.join(root, file)
                list_files.append(input_file_path)

    return list_files


def split_single_audio(file_path, output_dir, clip_duration, sr=None):
    try:
        audio, sr = librosa.load(file_path, sr=sr)
    except EOFError as e:
        logger.error(e)
        return None
    q, mod = divmod(len(audio), sr * clip_duration)
    n_clips = q + int(mod > 0)
    clips = np.split(
        np.pad(audio, pad_width=(0, sr * clip_duration - mod), mode='wrap'),
        n_clips
    )
    clips = np.array(clips).T
    clips = ((clips - clips.min(axis=0)) / (clips.max(axis=0) - clips.min(axis=0))).T * 2 - 1
    fnames = [f"{os.path.basename(file_path).split('.')[0]}_{t * clip_duration}_{(t + 1) * clip_duration}.wav" for t
              in range(n_clips)]

    for clip, fname in zip(clips, fnames):
        sf.write(os.path.join(output_dir, fname), data=clip, samplerate=sr, format='wav')


def embed_single_clip(file_path, embedding_model, sr=None):
    audio = librosa.load(file_path, sr=sr)
    embedding = embedding_model(audio)
    return embedding


def load_audio_files_with_tf_dataset(file_paths: list, sample_rate=None):
    def _load_audio(file_path):
        audio, sr = librosa.load(file_path.numpy(), sr=sample_rate)
        return audio

    def _load_audio_wrapper(file_path):
        audio = tf.py_function(_load_audio, [file_path], tf.float32)
        return audio

    # Create a TensorFlow dataset from the file paths
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    # Map the audio loading function to each element in the dataset
    dataset = dataset.map(_load_audio_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


# Function to get embeddings for each frame
def get_embeddings_for_audio(audio, model):
    audio = tf.cast(audio, tf.float32)
    embeddings = model(audio)
    return embeddings
