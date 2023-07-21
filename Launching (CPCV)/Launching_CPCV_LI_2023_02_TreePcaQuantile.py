from Strategies.LI_2023_02_TreePcaQuantile import *
from Quantreo.CombinatorialPurgedCV import *


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params_range = {
    "tp": [0.005 + i*0.002 for i in range(3)],
    "sl": [-0.005 - i*0.002 for i in range(3)],
}

params_fixed = {
    "look_ahead_period": 20,
    "sma_slow": 120,
    "sma_fast": 30,
    "rsi": 21,
    "atr": 15,
    "cost": 0.0001,
    "leverage": 5,
    "list_X": ["SMA_diff", "RSI", "ATR", "candle_way", "filling", "amplitude", "SPAN_A", "SPAN_B", "BASE", "STO_RSI",
               "STO_RSI_D", "STO_RSI_K", "previous_ret"],
    "train_mode": True,
}

CPCV = CombinatorialPurgedCV(data=df, TradingStrategy=TreePcaQuantile, fixed_parameters=params_fixed,
                             parameters_range=params_range, N=10, k=2, purge_pct=0.10)
CPCV.get_index_samples()
CPCV.get_sub_samples()
CPCV.run_optimization()
CPCV.get_pbo()
CPCV.display_all_graph()