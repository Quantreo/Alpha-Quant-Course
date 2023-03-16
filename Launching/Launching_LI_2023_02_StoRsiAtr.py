from Strategies.LI_2023_02_StoRsiAtr import *
from Quantreo.WalkForwardOptimization import *
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)


params_range = {
    "sto_period": [7.,14.,21.],
    "atr_period": [7.,14.,21.]
}

params_fixed = {
    "cost": 0.0001,
    "leverage": 5
}

RO = WalkForwardOptimization(df, StoRsiAtr, params_fixed, params_range,length_train_set=6_000)
RO.run_optimization()
RO.display()
params = RO.best_params_smoothed
print(params)