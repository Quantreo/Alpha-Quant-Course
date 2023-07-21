from Strategies.LI_2023_02_RsiSmaAtr import *
from Quantreo.WalkForwardOptimization import *
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)


params_range = {
    "fast_sma": [60.,80.],
    "slow_sma": [120.,150.]
}

params_fixed = {
    "rsi": 20,
    "cost": 0.0001,
    "leverage": 5
}

# You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)
WFO = WalkForwardOptimization(df, RsiSmaAtr, params_fixed, params_range,length_train_set=7_000)
WFO.run_optimization()

# Extract best parameters
params = WFO.best_params_smoothed[-1]
print("BEST PARAMETERS")
print(params)

# Show the results
WFO.display()


