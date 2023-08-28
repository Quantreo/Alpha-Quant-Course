import MetaTrader5 as mt5
import pandas as pd
import numpy as np

import time
from Quantreo.MetaTrader5 import *
from datetime import datetime, timedelta
from Quantreo.LiveTradingSignal import *
import warnings
warnings.filterwarnings("ignore")

symbol = "AUDUSD-Z"
lot = 0.01
magic = 16
timeframe = timeframes_mapping["8-hours"]
pct_tp, pct_sl = 0.0035, 0.0075 # DONT PUT THE MINUS SYMBOL ON THE SL
mt5.initialize()

current_account_info = mt5.account_info()
print("------------------------------------------------------------------")
print(f"Login: {mt5.account_info().login} \tserver: {mt5.account_info().server}")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(
    f"Balance: {current_account_info.balance} USD, \t Equity: {current_account_info.equity} USD, \t Profit: {current_account_info.profit} USD")
print("------------------------------------------------------------------")

timeframe_condition = get_verification_time(timeframe[1])

while True:

    if datetime.now().strftime("%H:%M:%S") in timeframe_condition:
        print(datetime.now().strftime("%H:%M:%S"))

        # ! YOU NEED TO HAVE THE SYMBOL IN THE MARKET WATCH TO OPEN OR CLOSE A POSITION
        selected = mt5.symbol_select(symbol)
        if not selected:
            print(f"\nERROR - Failed to select '{symbol}' in MetaTrader 5 with error :", mt5.last_error())

        # Create the signals
        buy, sell = li_2023_02_TreePcaQuantile(symbol, timeframe[0], 30, 80, 14, 5,
                    "../models/saved/LI_2023_02_TreePcaQuantile_AUDUSD_model.jolib",
                    "../models/saved/LI_2023_02_TreePcaQuantile_AUDUSD_sc.jolib",
                    "../models/saved/LI_2023_02_TreePcaQuantile_AUDUSD_pca.jolib")

        # Import current open positions
        res = resume()

        # Here we have a tp-sl exit signal, and we can't open two position on the same asset for the same strategy
        if ("symbol" in res.columns) and ("volume" in res.columns):
            if not ((res["symbol"] == symbol) & (res["volume"] == lot)).any():
                # Run the algorithm
                run(symbol, buy, sell, lot, pct_tp=pct_tp, pct_sl=pct_sl, magic=magic)
        else:
            run(symbol, buy, sell, lot, pct_tp=pct_tp, pct_sl=pct_sl, magic=magic)

        # Generally you run several asset in the same time, so we put sleep to avoid to do again the
        # same computations several times and therefore increase the slippage for other strategies
        time.sleep(1)
