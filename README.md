
24h per day
15 min resolution

Observation Space:

    - Time and day (ok)
    
    - Battery state of charge (ok)
    
    - Current power consumption (ok)
    
    - Consumption forecast (ok)
    
    - Grid consumption (ok)
    
    - Renewable ressources (ok)
    
    - Renewable forecast (24 - 48 hours) (ok)
    
    - Grid Price forecast (ok)

    - step function (ok)

## TODO
- class config as yaml file opener (Patrick)
- check battery behaviour (both)
- concept for input data (Julius)
- plan for paper
- Journals/Conference? (Patrick)

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
