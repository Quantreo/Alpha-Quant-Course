from Strategies.LI_2023_02_Ichimoku_1 import *
from Quantreo.WalkForwardOptimization import *
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)


params_range = {
    "ichi_window_1": [7.,9.,12.],
    "ichi_window_2": [26.,39.,52.]
}

params_fixed = {
    "tp": 0.005,
    "sl": -0.005,
    "cost": 0.0001,
    "leverage": 5
}

# You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)

WFO = WalkForwardOptimization(df, Ichimoku_1, params_fixed, params_range,length_train_set=3_500,
                             title="Walk-Forward optimization 2017-2023 EURUSD LI-01-2023-Ichimoku_1", randomness=1.00)
WFO.run_optimization()

# Extract best parameters
params = WFO.best_params_smoothed[-1]
print("BEST PARAMETERS")
print(params)

# Show the results
WFO.display()