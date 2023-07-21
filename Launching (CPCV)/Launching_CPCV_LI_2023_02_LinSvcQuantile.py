from Strategies.LI_2023_02_LinSvcQuantile import *
from Quantreo.CombinatorialPurgedCV import *


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params_range = {
    "sma_slow": [80., 90.],
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

CPCV = CombinatorialPurgedCV(data=df, TradingStrategy=LinSvcQuantile, fixed_parameters=params_fixed,
                             parameters_range=params_range, N=10, k=2, purge_pct=0.10)
CPCV.get_index_samples()
CPCV.get_sub_samples()
CPCV.run_optimization()
CPCV.get_pbo()
CPCV.display_all_graph()