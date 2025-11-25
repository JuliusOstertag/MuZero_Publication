'''
environment to train agent to manage and control battery charge and discharge in order to minimize costs

there is an emphasis on peak load curtailment in order to smooth expensive load peaks

Episodes:
    - Episode length?
    - How do different episodes look like?
        - Ein Tool, das für 4 Jahreszeiten zufällige (realistische) Profile
          erstellt
    - Use of real data or dynamically created dummy data?
    - How to ensure diverse data for training?
    - Open programming for easy change of episode length later

Observation Space:
    - Time and day
    - Battery state of charge
    - Current power consumption
    - Consumption forecast
    - Grid consumption
    - Renewable ressources
    - Renewable forecast (24 - 48 hours)
    - Grid Price forecast

Action Space:
    - Charge/Discharge battery
    - Load shedding? (später)
'''

import gymnasium as gym
import numpy as np

from config import Config
from generate_load_profile import generate_profile_one_day
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class BatteryEnv(gym.Env):

    def __init__(self, wind_scale=1, pv_scale=1):
        super(BatteryEnv, self).__init__()

        # General simulation parameters
        self.wind_scale = wind_scale
        self.pv_scale = pv_scale
        self.episode_length = Config.episode_length
        self.current_step = 0
        self.pv_raw = pd.read_csv("../data/pv.csv")
        self.wind_raw = pd.read_csv("../data/wind.csv")
        # Battery parameters
        self.battery_capacity = Config.battery_capacity
        self.reset()

    def reset(self):
        self.current_step = 0
        date = self._get_random_date()
        self._build_observation_space(date)

    def _get_random_date(self):
        dates = self.pv_raw["Date"].unique()
        return np.random.choice(dates)

    def _build_observation_space(self, date):
        renewable_resources = self._get_renewable_resources(date)
        self.observation_space = renewable_resources #gym.spaces.Box(low=-1, high=1, shape=(1,))

    def _get_renewable_resources(self, date) -> pd.DataFrame:
        pv = self._get_pv(date) * self.pv_scale
        wind = self._get_wind(date) * self.wind_scale
        return pd.DataFrame([pv, wind]).T

    def _get_pv(self, date) -> pd.Series:
        return self.pv_raw["PVOUT"][self.pv_raw["Date"] == date]

    def _get_wind(self, date) -> pd.Series:
        return self.wind_raw["WINDOUT"][self.wind_raw["Date"] == date]

def get_power_consumption(timestamp='2025-01-01'):
    # Da die GAN Generierung noch keine zuverlässigen Profile erstellt, wird hier eine 
    # Schleife zur Generierung verwendet, bis die Bedingung erfüllt ist.
    n = 1
    while True:
        # Generiere das Profil mit seed=seed kann ein seed für reproduzierbare Ergebnisse hinzugefügt werden
        profile = generate_profile_one_day(r'generated_models/V2-1_relu/generator')

        # Überprüfe, ob mehr als 20 % der Werte über 0,5 liegen
        if (profile > 0.5).mean() > 0.20:
            break  # Bedingung erfüllt, Schleife verlassen
        n += 1
    print(f"Profile generated after {n} attempt(s).")
    # Erstelle einen Datetime-Index für den gegebenen timestamp, von 00:00 bis 23:45 in 15-Minuten-Schritten
    date = pd.to_datetime(timestamp)
    time_index = pd.date_range(start=date, periods=96, freq='15min')

    # Erstelle eine Pandas-Series mit dem Datetime-Index
    power_series = pd.Series(profile.flatten(), index=time_index)

    power_series = adjust_amplitude(power_series)

    return power_series


def adjust_amplitude(power_series, min_peak=5, max_peak=10):
    """
    Skaliert die Zeitreihe so, dass die maximale Amplitude auf einen zufälligen Wert zwischen 
    min_peak und max_peak gesetzt wird.
    Das Minimum bleibt bei 0, der Rest wird proportional skaliert.

    Parameter:
    - power_series: Pandas-Series mit den Lastdaten
    - min_peak: Minimaler Wert für die maximale Amplitude (Standard: 5)
    - max_peak: Maximaler Wert für die maximale Amplitude (Standard: 10)

    Rückgabe:
    - Skalierte Pandas-Series
    """
    current_max = power_series.max()
    current_min = power_series.min()

    # Zufälliger Wert für die neue maximale Amplitude
    new_max = np.random.uniform(min_peak, max_peak)

    # Skalierungsfaktor berechnen
    scale_factor = new_max / current_max

    # Serie skalieren (Minimum bleibt bei 0)
    scaled_series = power_series * scale_factor

    return scaled_series


def simulate_forecast(power_series, noise_std=0.25):
    """
    Simuliert eine Vorhersage der Zeitreihe mit gaußschem Rauschen.

    Parameter:
    - power_series: Pandas-Series mit den Originaldaten
    - noise_std: Standardabweichung des Rauschens (Standard: 0.1)

    Rückgabe:
    - Vorhersage als Pandas-Series mit gleichem Index
    """
    noise = np.random.normal(0, noise_std, size=len(power_series))
    forecast = power_series + noise
    # Negative Werte auf 0 setzen (falls nötig)
    forecast = forecast.clip(lower=0)
    return forecast


def plot_power_consumption(power_series, forecast_series=None, title="Stromverbrauch (15-Minuten-Auflösung)"):
    """
    Plottet die Originaldaten und optional die Vorhersage mit Unsicherheit.

    Parameter:
    - power_series: Pandas-Series mit den Originaldaten
    - forecast_series: Pandas-Series mit der Vorhersage (optional)
    - title: Titel des Plots
    """
    plt.figure(figsize=(12, 6))

    # Originaldaten plotten
    plt.plot(power_series.index, power_series.values, marker='o', linestyle='-', markersize=3, color='blue', label='Original')

    # Vorhersage plotten (falls vorhanden)
    if forecast_series is not None:
        plt.plot(forecast_series.index, forecast_series.values, marker='o', linestyle='--', markersize=3, color='red', alpha=0.7, label='Vorhersage')

    # Achsenbeschriftungen und Titel
    plt.title(title)
    plt.xlabel("Uhrzeit")
    plt.ylabel("Stromverbrauch")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gcf().autofmt_xdate()

    # Legende hinzufügen
    if forecast_series is not None:
        plt.legend()

    # Plot anzeigen
    plt.tight_layout()
    plt.show()






if __name__ == '__main__':
    self = BatteryEnv()
    
    consumption = get_power_consumption(0)
    consumption_forecast = simulate_forecast(consumption)
    plot_power_consumption(consumption, consumption_forecast)

























