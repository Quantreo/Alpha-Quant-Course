from Quantreo.MonteCarlo import *
from Strategies.LI_2023_02_Ichimoku_1 import *
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params = {
    "ichi_window_1": 12.,
    "ichi_window_2": 39.,
    "tp": 0.005,
    "sl": -0.005,
    "cost": 0.0001,
    "leverage": 5
}

MC = MonteCarlo(df, Ichimoku_1, params, raw_columns=[], discount_calmar_ratio = 252*6)
MC.generate_paths(500, 2000)
MC.display_results()