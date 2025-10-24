import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import pandas as pd
import sys
import math

from IPython import display


def main():
    df = pd.read_csv('data/verarbeitete_zeitreihen.csv', dtype=float, usecols=lambda x: x.startswith('UN_'))
    training_data = prepare_load_data(df)

    BUFFER_SIZE = 1300
    BATCH_SIZE = 32

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    noise = tf.random.normal([1, 672])
    generated_data = generator(noise, training=False)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    EPOCHS = 50
    noise_dim = 672
    num_examples_to_generate = 16

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)


    @tf.function
    def train_step(timeseries):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = generator(noise, training=True)

            real_output = discriminator(timeseries, training=True)
            fake_output = discriminator(generated_data, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return generated_data

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)


    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for series_batch in dataset:
                generated_data = train_step(series_batch)

            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                values = generated_data.numpy().flatten()  # or .reshape(-1)

                print(f'created data for epoch: {epoch}')
                plot_vector(generated_data, epoch)

                np.savetxt(f'generated_data/load_series_epoch_{epoch}.csv', values, delimiter=',')

        plot_vector(generated_data, epoch)

    train(train_dataset, EPOCHS)


def make_generator_model():
    model = tf.keras.Sequential()

    # Project inpout to higher dimensional space
    model.add(layers.Dense(256, use_bias=False, input_shape=(672, )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((1, 256)))

    model.add(layers.Conv1DTranspose(128, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(64, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(672, kernel_size=3, strides=1, padding='same', use_bias=False, activation='tanh'))
    model.add(layers.Flatten())
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, kernel_size=3, strides=1, padding='same', input_shape=(672, 1)))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(128, kernel_size=3, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def prepare_load_data(df):
    # Nur Spalten mit dem Präfix 'UN_' auswählen
    un_spalten = [col for col in df.columns if col.startswith('UN_')]
    df = df[un_spalten].fillna(0)  # Fehlende Werte durch 0 ersetzen

    # Daten in ein numpy-array umwandeln
    daten = df.values

    # Annahmen:
    # - 96 Werte pro Tag (Viertelstundenwerte)
    # - 7 Tage pro Woche
    # - 1300 Wochen insgesamt

    tage_pro_woche = 7
    werte_pro_tag = 96
    werte_pro_woche = tage_pro_woche * werte_pro_tag

    # Anzahl der Wochen berechnena
    anzahl_wochen = daten.shape[0] // werte_pro_woche

    # Alle Wochen aller Spalten nacheinander aneinanderhängen
    result = np.zeros((anzahl_wochen * len(un_spalten), werte_pro_woche))

    # Für jede Spalte
    for i, spalte in enumerate(un_spalten):
        spalten_daten = daten[:, i]
        # Für jede Woche
        for woche in range(anzahl_wochen):
            start = woche * werte_pro_woche
            ende = start + werte_pro_woche
            wochen_daten = spalten_daten[start:ende]
            # Nur die ersten 96 Werte pro Woche nehmen (z.B. nur Montag)
            # oder: wochen_daten = wochen_daten[:96]  # Falls du die ersten 96 Werte pro Woche möchtest
            # oder: wochen_daten = np.mean(wochen_daten.reshape(tage_pro_woche, werte_pro_tag), axis=0)  # Falls du den Tagesmittelwert pro Viertelstunde möchtest
            result[woche + i * anzahl_wochen, :] = wochen_daten  # Hier nur die ersten 96 Werte pro Woche

    # Ergebnis: [1300, 96]
    return result


def plot_vector(generated_data, epoch):
    # output_vector = np.squeeze(generated_data.numpy())  # Shape: (672,)

    last_row = generated_data[-1, :]  # Shape: (672,)
    output_vector = np.squeeze(last_row)  # Shape: (672,)

    # Create a time axis (e.g., quarter-hourly timesteps)
    timesteps = np.arange(0, 672)

    # Plot the generated output
    plt.figure(figsize=(12, 4))
    plt.plot(timesteps, output_vector, label=f'Generated Load Profile Epoch {epoch}', color='blue')
    plt.xlabel('Timestep (Quarter-Hourly)')
    plt.ylabel('Value (e.g., kW)')
    plt.title('GAN-Generated Load Profile')
    plt.grid(True)
    plt.legend()
    plt.show()

    print('stop')


if __name__ == '__main__':
    main()