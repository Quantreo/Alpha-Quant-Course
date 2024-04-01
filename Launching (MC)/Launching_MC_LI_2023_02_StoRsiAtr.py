from Quantreo.MonteCarlo import *
from Strategies.LI_2023_02_StoRsiAtr import *
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params = {
    "sto_period": 14.,
    "atr_period": 14.,
    "cost": 0.0001,
    "leverage": 5
}

MC = MonteCarlo(df, StoRsiAtr, params, raw_columns=[], discount_calmar_ratio = 252*6)
MC.generate_paths(10, 2000)
MC.display_results()