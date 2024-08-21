import numpy as np
import tensorflow as tf


def get_model(num_outputs, input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.Dense(num_outputs, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_model_mil(shape, units, learning_rate=.01):
    inputs = tf.keras.Input(shape=shape)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units))(inputs)
    x = tf.keras.layers.Lambda(lambda y: tf.reduce_mean(y, axis=1))(x)
    outputs = tf.keras.layers.Activation('sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"])
    return model


def train_model(model, x_train, y_train):
    epochs = 100
    batch_size = min(np.shape(x_train)[0], 8)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
