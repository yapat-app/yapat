import dask
import numpy as np
import pandas as pd
import tensorflow as tf

from src.embeddings import BaseEmbedding


# BirdNet embedding class that inherits from BaseEmbedding and provides specific implementation for BirdNet.
class BirdnetEmbedding(BaseEmbedding):

    def __init__(self, clip_duration: float = 3.0, sampling_rate: int = 48000,
                 model_path='assets/models/birdnet/V2.4/BirdNET_GLOBAL_6K_V2.4_Model',
                 dask_client: dask.distributed.client.Client = None):
        super().__init__(clip_duration, model_path, sampling_rate, dask_client)
        self.model = None

    def load_model(self):
        """
        Load the BirdNet-specific model, using a TensorFlow SMLayer for generating embeddings.
        """

        input_layer = tf.keras.layers.Input(shape=(144000,), dtype='float32',
                                            name='input_layer')
        tfsm_layer = tf.keras.layers.TFSMLayer(self.model_path, call_endpoint='embeddings')(input_layer)
        self.model = tf.keras.Model(inputs=input_layer, outputs=tfsm_layer)

    def process(self):

        self.data = self.read_audio_dataset()
        self.load_model()
        results = []
        for row in self.data.iterrows():
            audio_data = np.expand_dims(row[1].audio_data, axis=0)
            embedding = self.model(audio_data)
            embedding_array = embedding['embeddings'].numpy().flatten()
            results.append(embedding_array.tolist())

        self.embeddings = pd.DataFrame(results, index=self.data.index, columns=[f'embedding_{i}' for i in range(1024)])
        self.save_embeddings('birdnet', self.embeddings)
        return

if __name__ == '__main__':
    embedding = BirdnetEmbedding()
    embedding.process()
