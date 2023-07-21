from Strategies.LI_2023_02_Ichimoku_1 import *
from Quantreo.CombinatorialPurgedCV import *


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

CPCV = CombinatorialPurgedCV(data=df, TradingStrategy=Ichimoku_1, fixed_parameters=params_fixed,
                             parameters_range=params_range, N=10, k=2, purge_pct=0.10)
CPCV.get_index_samples()
CPCV.get_sub_samples()
CPCV.run_optimization()
CPCV.get_pbo()
CPCV.display_all_graph()