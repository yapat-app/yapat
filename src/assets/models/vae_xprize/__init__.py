import tensorflow as tf

# For debugging
tf.config.run_functions_eagerly(True)


class SamplingLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def _get_encoder(input_shape, latent_dim):
    encoder_inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(encoder_inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = SamplingLayer()([z_mean, z_log_var])

    encoder = tf.keras.Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var, z], name="encoder")
    return encoder


def _get_decoder(output_shape, latent_dim):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))

    x = tf.keras.layers.Dense(4 * 12 * 64)(latent_inputs)  # Adjust according to flattened shape
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Reshape((4, 12, 64))(x)

    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)

    decoder_outputs = tf.keras.layers.Resizing(
        height=output_shape[0], width=output_shape[1], interpolation="bilinear", crop_to_aspect_ratio=False
    )(x)

    decoder = tf.keras.Model(inputs=latent_inputs, outputs=decoder_outputs, name="decoder")
    return decoder


class VAE(tf.keras.Model):
    def __init__(
            self,
            input_shape,
            latent_dim,
            beta_kl=1,
            beta_reconst=1,
            encoder=None,
            decoder=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = _get_encoder(input_shape, latent_dim) if encoder is None else encoder
        self.decoder = _get_decoder(input_shape, latent_dim) if decoder is None else decoder
        self.beta_kl = beta_kl
        self.beta_reconst = beta_reconst
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=None, mask=None):
        _, _, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return [z, reconstruction]

    def train_step(self, data):
        x, y = data
        try:
            tf.debugging.check_numerics(x, message="Check Input Tensor")
        except Exception as e:
            print(e)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.MeanSquaredError(reduction=None)(x, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = self.beta_reconst * reconstruction_loss + self.beta_kl * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        try:
            tf.debugging.check_numerics(grads, message="Check Gradients")
        except Exception as e:
            print(e)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        # after https://stackoverflow.com/a/67951345/3174232
        x, y = data

        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.mean_squared_error(x, reconstruction),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        total_loss = self.beta_reconst * reconstruction_loss + self.beta_kl * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
