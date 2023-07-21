from Strategies.LI_2023_02_RsiSma import *
from Quantreo.CombinatorialPurgedCV import *


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params_range = {
    "tp": [0.0035, 0.005, 0.010],
    "sl": [-0.0035, -0.005, -0.010],
}
params_fixed = {
    "fast_sma": 72,
    "slow_sma": 120,
    "rsi": 25,
    "cost": 0.0001,
    "leverage": 5
}

CPCV = CombinatorialPurgedCV(data=df, TradingStrategy=RsiSma, fixed_parameters=params_fixed,
                             parameters_range=params_range, N=10, k=2, purge_pct=0.10)
CPCV.get_index_samples()
CPCV.get_sub_samples()
CPCV.run_optimization()
CPCV.get_pbo()
CPCV.display_all_graph()