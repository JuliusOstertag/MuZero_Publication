
24h per day
15 min resolution

Observation Space:

    - Time and day (Patrick)
    
    - Battery state of charge (Patrick)
    
    - Current power consumption (Julius)
    
    - Consumption forecast (ok)
    
    - Grid consumption (Patrick)
    
    - Renewable ressources (ok)
    
    - Renewable forecast (24 - 48 hours) (ok)
    
    - Grid Price forecast (Julius)

    - step function (Patrick)

def get_power_consumption(timestamp):
  return pd.Series(24h, 15min res [00:00 --> 23:45)

def get_renewables(timestamp):
  return pd.Series(24h, 15min res [00:00 --> 23:45)

def make_forecast(pd.Series):
  gauss function --> greater deviation the longer the forecast
  return pd.Series(24h, 15min res [00:00 --> 23:45)

grid_price_forecast()
  download data 
  train regressor --> simple approach just use prices from Fraunhofer
