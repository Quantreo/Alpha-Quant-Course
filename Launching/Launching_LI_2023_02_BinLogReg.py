from Strategies.LI_2023_02_BinLogReg import *
from Quantreo.Backtest import *
from Quantreo.WalkForwardOptimization import *

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("../Data/FixTimeBars/EURUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)


params_range = {
    "tp": [0.005 + i*0.001 for i in range(5)],
    "sl": [-0.005 - i*0.001 for i in range(5)],
}

params_fixed = {
    "look_ahead_period": 20,
    "sma_fast": 30,
    "sma_slow":80,
    "rsi":14,
    "atr":5,
    "cost": 0.0001,
    "leverage": 5,
    "list_X": ["SMA_diff", "RSI", "ATR"],
    "train_mode": True,
}

WFO = WalkForwardOptimization(df, BinLogReg, params_fixed, params_range,length_train_set=2_000)
WFO.run_optimization()
WFO.display()
params = WFO.best_params_smoothed
print(params)

