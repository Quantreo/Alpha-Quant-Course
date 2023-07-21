from Strategies.LI_2023_02_RsiSmaAtr import *
from Quantreo.CombinatorialPurgedCV import *


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

CPCV = CombinatorialPurgedCV(data=df, TradingStrategy=RsiSmaAtr, fixed_parameters=params_fixed,
                             parameters_range=params_range, N=10, k=2, purge_pct=0.10)
CPCV.get_index_samples()
CPCV.get_sub_samples()
CPCV.run_optimization()
CPCV.get_pbo()
CPCV.display_all_graph()