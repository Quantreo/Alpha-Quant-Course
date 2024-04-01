from Quantreo.MonteCarlo import *
from Strategies.LI_2023_02_TreePcaQuantile import *
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params = {
    "tp": 0.0035,
    "sl": -0.0035,
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

MC = MonteCarlo(df, TreePcaQuantile, params, raw_columns=[], discount_calmar_ratio = 252*6)
MC.generate_paths(500, 2000)
MC.display_results()