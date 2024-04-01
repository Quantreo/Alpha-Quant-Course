from Quantreo.MonteCarlo import *
from Strategies.LI_2023_02_RsiSma import *
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params = {
    "tp": 0.005,
    "sl": -0.005,
    "fast_sma": 72,
    "slow_sma": 120,
    "rsi": 25,
    "cost": 0.0001,
    "leverage": 5
}

MC = MonteCarlo(df, RsiSma, params, raw_columns=[], discount_calmar_ratio = 252*6)
MC.generate_paths(500, 2000)
MC.display_results()