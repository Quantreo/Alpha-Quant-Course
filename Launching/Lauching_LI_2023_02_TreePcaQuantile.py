from Strategies.LI_2023_02_TreePcaQuantile import *
from Quantreo.Backtest import *
from Quantreo.WalkForwardOptimization import *

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)


params_range = {
    "tp": [0.005 + i*0.001 for i in range(5)],
    "sl": [-0.005 - i*0.001 for i in range(5)],
}

params_fixed = {
    "look_ahead_period": 20,
    "sma_slow": 120,
    "sma_fast": 30,
    "rsi": 21,
    "atr": 15,
    "cost": 0.0001,
    "leverage": 5,
    "list_X": ["SMA_diff", "RSI", "ATR", "candle_way", "filling", "amplitude", "SPAN_A", "SPAN_B", "BASE", "STO_RSI",
               "STO_RSI_D", "STO_RSI_K", "previous_ret"],
    "train_mode": True,
}

RO = WalkForwardOptimization(df, TreePcaQuantile, params_fixed, params_range,length_train_set=5_000)
RO.run_optimization()
RO.display()
params = RO.best_params_smoothed
print(params)