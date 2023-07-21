from Strategies.LI_2023_02_LinSvcQuantile import *
from Quantreo.Backtest import *
from Quantreo.WalkForwardOptimization import *

import warnings
warnings.filterwarnings("ignore")
from joblib import dump

# SAVE WEIGHTS
save = True
name = "LI_2023_02_LinSvcQuantile_AUDUSD"

# Import the data
df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params_range = {
    "sma_slow": [80.],
    "atr": [5., 10.]
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

# You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)
WFO = WalkForwardOptimization(df, LinSvcQuantile, params_fixed, params_range,length_train_set=5_000, randomness=1.00)
WFO.run_optimization()

# Extract best parameters
params = WFO.best_params_smoothed[-1]
print("BEST PARAMETERS")
print(params)

model = params["model"]
sc = params["sc"]

# Save the weights
if save:
    dump(model, f"../models/saved/{name}_model.jolib")
    dump(sc, f"../models/saved/{name}_sc.jolib")

# Show the results
WFO.display()