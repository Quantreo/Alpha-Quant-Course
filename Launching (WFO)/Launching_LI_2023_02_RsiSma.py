from Strategies.LI_2023_02_RsiSma import *
from Quantreo.WalkForwardOptimization import *
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)


params_range = {
    "tp": [0.0035,],
    "sl": [-0.0035, -0.005],
}

params_fixed = {
    "fast_sma": 72,
    "slow_sma": 120,
    "rsi": 25,
    "cost": 0.0001,
    "leverage": 5
}

# You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)
WFO = WalkForwardOptimization(df, RsiSma, params_fixed, params_range,length_train_set=6_000, randomness=1.00)
WFO.run_optimization()

# Extract best parameters
params = WFO.best_params_smoothed[-1]
print("BEST PARAMETERS")
print(params)

# Show the results
WFO.display()