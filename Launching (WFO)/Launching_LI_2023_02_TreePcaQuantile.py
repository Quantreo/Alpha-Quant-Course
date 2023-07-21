from Strategies.LI_2023_02_TreePcaQuantile import *
from Quantreo.Backtest import *
from Quantreo.WalkForwardOptimization import *

import warnings
warnings.filterwarnings("ignore")

# SAVE WEIGHTS
save = False
name = "LI_2023_02_TreePcaQuantile_AUDUSD"

df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)


params_range = {
    "tp": [0.005 + i*0.002 for i in range(3)],
    "sl": [-0.005 - i*0.002 for i in range(3)],
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

# You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)

WFO = WalkForwardOptimization(df, TreePcaQuantile, params_fixed, params_range,length_train_set=5_000, randomness=1.00)
WFO.run_optimization()

# Extract best parameters
params = WFO.best_params_smoothed[-1]
print("BEST PARAMETERS")
print(params)

# Extract the
model = params["model"]
sc = params["sc"]
pca = params["pca"]

if save:
    dump(model, f"../models/saved/{name}_model.jolib")
    dump(sc, f"../models/saved/{name}_sc.jolib")
    dump(pca, f"../models/saved/{name}_pca.jolib")

# Show the results
WFO.display()
