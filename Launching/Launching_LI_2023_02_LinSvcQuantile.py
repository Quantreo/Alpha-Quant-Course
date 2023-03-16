from Strategies.LI_2023_02_LinSvcQuantile import *
from Quantreo.Backtest import *
from Quantreo.WalkForwardOptimization import *

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)


params_range = {
    "sma_slow": [80., 90., 100.],
    "atr": [5., 10., 15.]
}

params_fixed = {
    "look_ahead_period": 20,
    "sma_fast": 30,
    "rsi": 21,
    "tp": 0.007,
    "sl": -0.012,
    "cost": 0.0001,
    "leverage": 5,
    "list_X": ["SMA_diff", "RSI", "ATR"],
    "train_mode": True,
}

RO = WalkForwardOptimization(df, LinSvcQuantile, params_fixed, params_range,length_train_set=5_000)
RO.run_optimization()
RO.display()
params = RO.best_params_smoothed
print(params)