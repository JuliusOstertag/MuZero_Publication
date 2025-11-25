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
import pandas as pd

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


if __name__ == '__main__':
    self = BatteryEnv()

























