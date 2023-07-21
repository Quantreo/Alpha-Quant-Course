from Strategies.LI_2023_02_BinLogReg import *
from Quantreo.Backtest import *
from Quantreo.WalkForwardOptimization import *

import warnings
warnings.filterwarnings("ignore")

# SAVE WEIGHTS
save = False
name = "LI_2023_02_BinLogReg_AUDUSD"

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

# You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)
WFO = WalkForwardOptimization(df, BinLogReg, params_fixed, params_range,length_train_set=5_000)
WFO.run_optimization()

# Extract best parameters
params = WFO.best_params_smoothed[-1]
print("BEST PARAMETERS")
print(params)

model = params["model"]
if save:
    dump(model, f"../models/saved/{name}_model.jolib")

# Show the results
WFO.display()

