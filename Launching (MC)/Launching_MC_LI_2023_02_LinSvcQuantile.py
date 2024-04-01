from Quantreo.MonteCarlo import *
from Strategies.LI_2023_02_LinSvcQuantile import *
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params = {
    "sma_slow": 80.,
    "atr": 10.,
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

MC = MonteCarlo(df, LinSvcQuantile, params, raw_columns=[], discount_calmar_ratio = 252*6)
MC.generate_paths(500, 2000)
MC.display_results()