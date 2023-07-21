from Strategies.LI_2023_02_BinLogReg import *
from Quantreo.CombinatorialPurgedCV import *


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params_range = {
    "tp": [0.005 + i*0.002 for i in range(3)],
    "sl": [-0.005 - i*0.002 for i in range(3)],
}

params_fixed = {
    "look_ahead_period": 20,
    "sma_fast": 30,
    "sma_slow":80,
    "rsi":14,
    "atr":5,
    "cost": 0.0001,
    "leverage": 5,
    "list_X": ["SMA_diff", "RSI", "ATR"],
    "train_mode": True,
}

CPCV = CombinatorialPurgedCV(data=df, TradingStrategy=BinLogReg, fixed_parameters=params_fixed,
                             parameters_range=params_range, N=10, k=2, purge_pct=0.10)
CPCV.get_index_samples()
CPCV.get_sub_samples()
CPCV.run_optimization()
CPCV.get_pbo()
CPCV.display_all_graph()