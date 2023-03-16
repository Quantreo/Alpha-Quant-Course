from Strategies.LI_2023_02_RsiSmaAtr import *
from Quantreo.WalkForwardOptimization import *
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)



params_range = {
    "fast_sma": [60.,80.],
    "slow_sma": [120.,150.],
    "rsi": [20.,30.]
}

params_fixed = {
    "cost": 0.0001,
    "leverage": 5
}

RO = WalkForwardOptimization(df, RsiSmaAtr, params_fixed, params_range,length_train_set=5_000)
RO.run_optimization()
RO.display()
params = RO.best_params_smoothed
print(params)

