from windpowerlib import WindTurbine
from windpowerlib import ModelChain

import pandas as pd

df = pd.read_csv(
    "../data/000_Aeroport_min10_D1_f.csv",
    sep=";",
    comment="#"
)

df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d.%m.%Y %H:%M:%S")
df = df.set_index("datetime")

weather_df = pd.DataFrame({
    ("wind_speed", 10): df["WS"],
    ("temperature", 2): df["TEMP"],
    ("pressure", 0): df["AP"] * 100,
    ("roughness_length", 0): [0.15] * len(df)
})

enercon_e126 = {
    "turbine_type": "E-126/4200",
    "hub_height": 135,
}
turbine = WindTurbine(**enercon_e126)


mc = ModelChain(
    turbine,
    wind_speed_model="logarithmic",
    density_model="ideal_gas",
    temperature_model="linear_gradient"
)

mc.run_model(weather_df)

power_series = mc.power_output / mc.power_output.max()
export = pd.DataFrame([df["Date"], df["Time"], power_series]).T
export.columns = ["Date", "Time", "WINDOUT"]
export.to_csv("../data/wind.csv", index=False)
print(export)