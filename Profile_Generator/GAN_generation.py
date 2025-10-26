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

'''
Dynamische Größe und Breite von Daten hinzugefügt. 

Nach einigen Experimenten scheint Relu bisher für Lastprofile das beste Ergebnis zu liefern
'''


def main():
    plot_single_vector = False
    dir_name = 'V3-2_20_times_2024'
    dirs = ['generated_data', 'generated_plots', 'training_checkpoints']
    for dir in dirs:
        if not os.path.exists(f'{dir}/{dir_name}'):
            os.makedirs(f'{dir}/{dir_name}')
            print(f"Verzeichnis {dir}/{dir_name} wurde erstellt.")

    energy_charts_data = pd.read_csv('data/energy-charts_Öffentliche_Nettostromerzeugung_in_Deutschland_2024(1).csv')
    training_data, energy_cols = prepare_energy_charts_data(energy_charts_data)

    # Repeat the data 20 times along the first axis
    training_data = np.repeat(training_data, 20, axis=0)

    BUFFER_SIZE = training_data.shape[0]
    BATCH_SIZE = calculate_batch_size(training_data.shape[0])
    EPOCHS = 500
    noise_dim = training_data.shape[-1]
    num_examples_to_generate = 16

    output_shape = training_data[0].shape

    # Batch and shuffle the data
    #ragged_tensor = tf.ragged.constant(training_data)
    train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = make_generator_model(noise_dim, output_shape)
    discriminator = make_discriminator_model(output_shape)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    checkpoint_dir = f'./training_checkpoints/{dir_name}'
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

            if (epoch + 1) % 100 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                values = generated_data.numpy().flatten()  # or .reshape(-1)

                print(f'created data for epoch: {epoch}')
                if plot_single_vector:
                    plot_vector(generated_data, epoch, dir_name)
                else:
                    plot_generation(generated_data, epoch, dir_name)

                np.savetxt(f'generated_data/{dir_name}/load_series_epoch_{epoch}.csv', values, delimiter=',')

        if plot_single_vector:
            plot_vector(generated_data, epoch, dir_name)
        else:
            plot_generation(generated_data, epoch, dir_name)

    train(train_dataset, EPOCHS)


def make_generator_model(noise_dim, output_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(512, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((1, 512)))
    model.add(layers.Conv1DTranspose(256, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv1DTranspose(128, kernel_size=3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(
        layers.Conv1DTranspose(output_shape[-1], kernel_size=3, strides=output_shape[0], padding='same', use_bias=False,
                               activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Reshape(output_shape))
    return model


def make_discriminator_model(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(128, kernel_size=3, strides=1, padding='same', input_shape=input_shape))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(256, kernel_size=3, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def prepare_energy_charts_data(df, datetime_col='Datum (MEZ)'):
    # Set the datetime column as the index
    df = df.set_index(datetime_col)
    df.index = pd.to_datetime(df.index)

    # Select only the energy columns (exclude non-energy columns if needed)
    energy_cols = [
        'Laufwasser', 'Biomasse', 'Braunkohle', 'Steinkohle', 'Öl',
        'Kohlegas', 'Erdgas', 'Geothermie', 'Speicherwasser',
        'Pumpspeicher', 'Andere', 'Müll', 'Wind Offshore',
        'Wind Onshore', 'Solar', 'Last', 'Day Ahead Auktion (DE-LU)'
    ]

    sum_energy_cols = len(energy_cols)

    df = df[energy_cols].fillna(0)

    # Constants
    werte_pro_tag = 96
    tage_pro_woche = 7
    werte_pro_woche = tage_pro_woche * werte_pro_tag

    # Resample to 15-minute intervals if not already
    df = df.asfreq('15T')

    # Calculate the number of full weeks
    total_values = len(df)
    anzahl_wochen = total_values // werte_pro_woche

    # Reshape the data into weeks
    # Result shape: (number_of_weeks, number_of_columns, werte_pro_woche)
    result = np.zeros((anzahl_wochen, len(energy_cols), werte_pro_woche))

    for i, col in enumerate(energy_cols):
        col_data = df[col].values[:anzahl_wochen * werte_pro_woche]
        result[:, i, :] = col_data.reshape(anzahl_wochen, werte_pro_woche)

    return result, sum_energy_cols



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
    return result, 1


def calculate_batch_size(x):
    """
    Berechnet den y-Wert für einen gegebenen x-Wert zwischen 2 und 1000.
    - Für x < 2 wird y = 2 zurückgegeben.
    - Für x > 1000 wird y = 64 zurückgegeben.
    - Für 2 ≤ x ≤ 1000 wird eine potenzbasierte Funktion verwendet.
    """
    if x <= 2:
        return 2
    elif x >= 1000:
        return 64
    else:
        a = 62 / (1000**0.5 - 2**0.5)
        b = 0.5
        c = 2 - a * (2**b)
        y = a * (x**b) + c
        return int(round(y))


def plot_vector(generated_data, epoch, dir_name):
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
    plt.savefig(f'generated_plots/{dir_name}/{epoch}.png')

    print('stop')


def plot_generation(data, epoch, dir_name):
    data = data[-1]
    array = data.numpy()
    df = pd.DataFrame(array)
    df = df.T  # Now shape is (17, 672)
    headers = [
        "Laufwasser", "Biomasse", "Braunkohle", "Steinkohle", "Öl",
        "Kohlegas", "Erdgas", "Geothermie", "Speicherwasser", "Pumpspeicher",
        "Andere", "Müll", "Wind Offshore", "Wind Onshore", "Solar", "Last",
        "Day Ahead Auktion (DE-LU)"
    ]
    timesteps = np.arange(672)
    plt.figure(figsize=(16, 8))
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2"
    ]

    # Die ersten 15 Spalten auswählen
    df_subset = df.iloc[:, :15]

    # Stackplot erstellen
    fig, ax = plt.subplots(figsize=(12, 6))

    # X-Achse (z.B. Index oder eine Zeitspalte)
    x = df_subset.index  # oder df['deine_zeit_spalte'] falls du eine spezifische Spalte nutzen möchtest

    # Stackplot mit den ersten 15 Spalten
    ax.stackplot(x,
                 *[df_subset.iloc[:, i] for i in range(len(df_subset.columns))],
                 labels=headers[:15],
                 colors=colors[:15],
                 alpha=0.8)

    ax.plot(x, df.iloc[:, 15], color="black", linewidth=2, label="Last")
    ax.plot(x, df.iloc[:, 16], color="gray", linestyle="--", linewidth=2, label="Day Ahead Auktion (DE-LU)")

    # Beschriftungen und Titel
    ax.set_xlabel("Datum (MESZ)")
    ax.set_ylabel("Leistung (MW)")
    ax.set_title('Stromerzeugung und -verbrauch in Deutschland')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'generated_plots/{dir_name}/{epoch}.png')
    plt.close()  # Close the figure to free memory


if __name__ == '__main__':
    main()