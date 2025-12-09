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
from config import Config
from generate_load_profile import generate_profile_one_day
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


cfg = Config.from_yaml("config/exp1.yaml")


class BatteryEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, wind_scale=1.0, pv_scale=1.0):
        super(BatteryEnv, self).__init__()

        # General simulation parameters
        self.wind_scale = wind_scale
        self.pv_scale = pv_scale
        self.episode_length = cfg.episode_length  # should be 96 for 1 day at 15 min
        self.current_step = 0
        self.pv_raw = pd.read_csv("../data/pv.csv")
        self.wind_raw = pd.read_csv("../data/wind.csv")
        self.grid_price_raw = pd.read_csv("../data/grid_price.csv")
        self.episode_consumption = None

        self.capacity = cfg.battery_capacity
        self.dt = 0.25  # 15 minutes in hours
        self.max_power = self.capacity / self.dt

        self.soc = 0.5  # initial state of charge, relative [0, 1]

        # Action space: 11 discrete values mapping to -100 percent ... 0 ... +100 percent
        # Convention:
        #   negative -> charging (battery takes power)
        #   positive -> discharging (battery delivers power to load)
        self.action_values = np.linspace(-1.0, 1.0, 11, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(11)

        self.n_scalar_features = 6  # time_of_day_norm, soc, current load, pv, wind, price (or forecast_t etc.)
        self.n_forecast_features = 4  # load, pv, wind, price
        self.forecast_horizon = self.episode_length  # 96

        obs_dim = self.n_scalar_features + self.n_forecast_features * self.forecast_horizon

        # Define bounds (here a simple 0..20 kW for all power-like values, 0..1 for normalized terms)
        low = np.full(obs_dim, 0.0, dtype=np.float32)
        high = np.full(obs_dim, 20.0, dtype=np.float32)

        low[0:2] = 0.0
        high[0:2] = 1.0

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.date = None
        self.time_index = None
        self.time_df = None
        self.renewable_resources = None
        self.consumption = None
        self.consumption_forecast = None
        self.last_grid_consumption = 0.0
        self.forecast_horizon = self.episode_length  # 96 for 24h
        self.forecast_df = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.soc = np.random.random()
        self.last_grid_consumption = 0.0

        self.date = self._get_random_date()
        self._build_time_index(self.date)
        self.renewable_resources = self._get_renewable_resources(self.date)
        self.consumption = self._get_power_consumption(timestamp=self.date)
        self.forecast_df = self._get_forecast(self.date)
        self.consumption_forecast = self.forecast_df["consumption_forecast"]

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: int):
        rate = float(self.action_values[action])  # -1 ... 1

        t = self.current_step

        load_t = float(self.consumption.iloc[t])
        pv_t = float(self.renewable_resources["pv"].iloc[t])
        wind_t = float(self.renewable_resources["wind"].iloc[t])

        desired_power = rate * self.max_power

        allowed_power = self._limit_battery_power(desired_power)

        energy_change = -allowed_power * self.dt
        self.soc = np.clip(self.soc + energy_change / self.capacity, 0.0, 1.0)

        grid_consumption = load_t - pv_t - wind_t - allowed_power
        self.last_grid_consumption = grid_consumption

        grid_import = max(grid_consumption, 0.0)
        reward = -grid_import

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        obs = self._get_observation()
        info = {
            "grid_consumption": grid_consumption,
            "battery_power": allowed_power,
            "soc": self.soc,
            "load": load_t,
            "pv": pv_t,
            "wind": wind_t,
        }

        return obs, reward, terminated, truncated, info

    def _get_random_date(self):
        dates = self.pv_raw["Date"].unique()
        return np.random.choice(dates)

    def _build_time_index(self, date):
        start = pd.to_datetime(date)
        self.time_index = pd.date_range(start=start, periods=self.episode_length, freq="15min")
        self.time_df = pd.DataFrame({"timestamp": self.time_index})

    def _get_renewable_resources(self, date) -> pd.DataFrame:
        pv = self._get_pv(date).reset_index(drop=True) * self.pv_scale
        wind = self._get_wind(date).reset_index(drop=True) * self.wind_scale

        df = pd.DataFrame({
            "pv": pv.values,
            "wind": wind.values,
        })
        df = df.iloc[: self.episode_length].copy()
        df.index = self.time_index
        return df

    def _get_pv(self, date) -> pd.Series:
        return self.pv_raw.loc[self.pv_raw["Date"] == date, "PVOUT"]

    def _get_wind(self, date) -> pd.Series:
        return self.wind_raw.loc[self.wind_raw["Date"] == date, "WINDOUT"]

    def _get_grid_price(self, date) -> pd.Series:
        return self.grid_price_raw.loc[self.grid_price_raw["Date"] == date, "DAYAHEADPRICE"]

    def _get_forecast(self, date) -> pd.DataFrame:
        pv_forecast = self._simulate_forecast(self._get_pv(date) * self.pv_scale)
        wind_forecast = self._simulate_forecast(self._get_wind(date) * self.wind_scale)

        # Use the episode consumption as "true" to forecast from
        consumption_forecast = self._simulate_forecast(self.consumption)

        grid_price_forecast = self._simulate_forecast(self._get_grid_price(date))

        df = pd.DataFrame({
            "pv_forecast": pv_forecast.values,
            "wind_forecast": wind_forecast.values,
            "consumption_forecast": consumption_forecast.values,
            "grid_price_forecast": grid_price_forecast.values,
        })

        df = df.iloc[: self.episode_length].copy()
        df.index = self.time_index
        return df

    def _get_observation(self) -> np.ndarray:
        t = min(self.current_step, self.episode_length - 1)

        time_of_day_norm = t / self.episode_length  # 0 .. <1
        load_t = float(self.consumption.iloc[t])
        pv_t = float(self.renewable_resources["pv"].iloc[t])
        wind_t = float(self.renewable_resources["wind"].iloc[t])

        price_series = self._get_grid_price(self.date)
        price_t = float(price_series.iloc[t])

        scalar_part = np.array([
            time_of_day_norm,
            self.soc,
            load_t,
            pv_t,
            wind_t,
            price_t,
        ], dtype=np.float32)

        H = self.forecast_horizon  # 96
        nF = self.n_forecast_features
        future = np.zeros((H, nF), dtype=np.float32)

        remaining = self.episode_length - t
        steps_to_fill = min(H, remaining)

        future[:steps_to_fill, 0] = self.forecast_df["consumption_forecast"].iloc[t:t + steps_to_fill].values
        future[:steps_to_fill, 1] = self.forecast_df["pv_forecast"].iloc[t:t + steps_to_fill].values
        future[:steps_to_fill, 2] = self.forecast_df["wind_forecast"].iloc[t:t + steps_to_fill].values
        future[:steps_to_fill, 3] = self.forecast_df["grid_price_forecast"].iloc[t:t + steps_to_fill].values
        future_flat = future.flatten().astype(np.float32)
        obs = np.concatenate([scalar_part, future_flat], axis=0)

        return obs

    def _limit_battery_power(self, desired_power: float) -> float:
        """
        Enforce state of charge constraints for the given desired power.

        desired_power > 0: discharge (battery supplies power)
        desired_power < 0: charge (battery absorbs power)
        """
        if desired_power > 0:
            max_discharge_energy = self.soc * self.capacity
            max_discharge_power = max_discharge_energy / self.dt
            allowed_power = min(desired_power, max_discharge_power)
        elif desired_power < 0:
            remaining_capacity = (1.0 - self.soc) * self.capacity
            max_charge_power = remaining_capacity / self.dt
            allowed_power = max(desired_power, -max_charge_power)
        else:
            allowed_power = 0.0

        return allowed_power


    def _get_power_consumption(self, timestamp='2025-01-01'):
        while True:
            profile = generate_profile_one_day(r'generated_models/V2-1_relu/generator') # seed kann festgelegt werden

            if (profile > 0.5).mean() > 0.20:
                break

        date = pd.to_datetime(timestamp)
        time_index = pd.date_range(start=date, periods=96, freq='15min')

        power_series = pd.Series(profile.flatten(), index=time_index)
        power_series = self._adjust_amplitude(power_series)

        return power_series

    def _adjust_amplitude(self, power_series, min_peak=5, max_peak=10):
        current_max = power_series.max()

        new_max = np.random.uniform(min_peak, max_peak)

        scale_factor = new_max / current_max
        scaled_series = power_series * scale_factor

        return scaled_series

    def _simulate_forecast(self, power_series, noise_std=0.25):
        noise = np.random.normal(0, noise_std, size=len(power_series))
        forecast = power_series + noise

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
    
    print(self.observation_space)
