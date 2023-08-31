import numpy as np
from Quantreo.DataPreprocessing import *
from Quantreo.MetaTrader5 import *

from joblib import load


def random(symbol):
    values = [True, False]
    buy = np.random.choice(values)
    sell = not buy
    return buy, sell


def li_2023_02_RsiSma(symbol, timeframe, fast_sma_period, slow_sma_period, rsi_period):
    df = get_rates(symbol=symbol, number_of_data=500, timeframe=timeframe)
    df = sma(df, "close", fast_sma_period)
    df = sma(df, "close", slow_sma_period)
    df = rsi(df, "close", rsi_period)

    # def signal
    df["RSI_retarded"] = df[f"RSI"].shift(1)
    condition_1_buy = df[f"SMA_{fast_sma_period}"].iloc[-1] < df[f"SMA_{slow_sma_period}"].iloc[-1]
    condition_1_sell = df[f"SMA_{fast_sma_period}"].iloc[-1] > df[f"SMA_{slow_sma_period}"].iloc[-1]

    condition_2_buy = df[f"RSI"].iloc[-1] > df["RSI_retarded"].iloc[-1]
    condition_2_sell = df[f"RSI"].iloc[-1] < df["RSI_retarded"].iloc[-1]

    buy = condition_1_buy & condition_2_buy
    sell = condition_1_sell & condition_2_sell
    return buy, sell


def li_2023_02_LogRegQuantile(symbol, timeframe, sma_fast_period, slow_sma_period, rsi_period, atr_period, model_path):
    df = get_rates(symbol=symbol, number_of_data=500, timeframe=timeframe)

    df = sma_diff(df, "close", sma_fast_period, slow_sma_period)
    df = rsi(df, "close", rsi_period)
    df = atr(df, atr_period)

    df = df.dropna()
    model = load(model_path)

    list_X = ["SMA_diff", "RSI", "ATR"]
    X = df[list_X]

    predict_array = model.predict(X)
    prediction = predict_array[-1]

    buy = True if prediction == 1 else False
    sell = False
    return buy, sell


def li_2023_02_RsiSmaAtr(symbol, timeframe, fast_sma_period, slow_sma_period, rsi_period):
    df = get_rates(symbol=symbol, number_of_data=500, timeframe=timeframe)
    df = sma(df, "close", fast_sma_period)
    df = sma(df, "close", slow_sma_period)
    df = rsi(df, "close", rsi_period)

    # def signal
    df["RSI_retarded"] = df[f"RSI"].shift(1)
    condition_1_buy = df[f"SMA_{fast_sma_period}"].iloc[-1] < df[f"SMA_{slow_sma_period}"].iloc[-1]
    condition_1_sell = df[f"SMA_{fast_sma_period}"].iloc[-1] > df[f"SMA_{slow_sma_period}"].iloc[-1]

    condition_2_buy = df[f"RSI"].iloc[-1] > df["RSI_retarded"].iloc[-1]
    condition_2_sell = df[f"RSI"].iloc[-1] < df["RSI_retarded"].iloc[-1]

    buy = condition_1_buy & condition_2_buy
    sell = condition_1_sell & condition_2_sell

    # TP - SL
    df = atr(df, 15)
    df["TP_level"] = df["open"] + 3 * df["ATR"].shift(1)
    df["SL_level"] = df["open"] - 3 * df["ATR"].shift(1)
    df["TP_pct"] = (df["TP_level"] - df["open"]) / df["open"]
    df["SL_pct"] = (df["SL_level"] - df["open"]) / df["open"]

    pct_tp = df["TP_pct"].iloc[-1]
    pct_sl = df["SL_pct"].iloc[-1]
    return buy, sell, pct_tp, np.abs(pct_sl)


def li_2023_02_Ichimoku_1(symbol, timeframe, ichi_window_1, ichi_window_2):
    df = get_rates(symbol=symbol, number_of_data=500, timeframe=timeframe)
    df = ichimoku(df, ichi_window_1, ichi_window_2)

    condition_1_buy = df["SPAN_A"].iloc[-1] > df["SPAN_B"].iloc[-1]
    condition_1_sell = df["SPAN_A"].iloc[-1] < df["SPAN_B"].iloc[-1]

    condition_2_buy = (df["CONVERSION"].shift(1).iloc[-1] < df["BASE"].shift(1).iloc[-1]) & (df["BASE"].iloc[-1] < df["CONVERSION"].iloc[-1])
    condition_2_sell = (df["CONVERSION"].shift(1).iloc[-1] > df["BASE"].shift(1).iloc[-1]) & (df["BASE"].iloc[-1] < df["CONVERSION"].iloc[-1])

    buy = condition_1_buy & condition_2_buy
    sell = condition_1_sell & condition_2_sell
    return buy, sell


def li_2023_02_StoRsiAtr(symbol, timeframe, sto_period, atr_period):
    df = get_rates(symbol=symbol, number_of_data=500, timeframe=timeframe)
    df = sto_rsi(df, "close", sto_period)

    # def signal
    condition_1_buy = (df["STO_RSI_K"].iloc[-1] < 30) & (df["STO_RSI_D"].iloc[-1] < 30)
    condition_1_sell = (df["STO_RSI_K"].iloc[-1] > 70) & (df["STO_RSI_D"].iloc[-1] > 70)

    condition_2_buy = (df["STO_RSI_K"].iloc[-1] < df["STO_RSI_D"].iloc[-1]) & (
                df["STO_RSI_K"].shift(1).iloc[-1] > df["STO_RSI_D"].shift(1).iloc[-1])
    condition_2_sell = (df["STO_RSI_K"].iloc[-1] > df["STO_RSI_D"].iloc[-1]) & (
                df["STO_RSI_K"].shift(1).iloc[-1] < df["STO_RSI_D"].shift(1).iloc[-1])

    buy = condition_1_buy & condition_2_buy
    sell = condition_1_sell & condition_2_sell

    # TP - SL
    df = atr(df, atr_period)
    df["TP_level"] = df["open"] + 3 * df["ATR"].shift(1)
    df["SL_level"] = df["open"] - 3 * df["ATR"].shift(1)
    df["TP_pct"] = (df["TP_level"] - df["open"]) / df["open"]
    df["SL_pct"] = (df["SL_level"] - df["open"]) / df["open"]

    pct_tp = df["TP_pct"].iloc[-1]
    pct_sl = df["SL_pct"].iloc[-1]
    return buy, sell, pct_tp, np.abs(pct_sl)


def li_2023_02_LinSvcQuantile(symbol, timeframe, sma_fast_period, slow_sma_period, rsi_period, atr_period, model_path, sc_path):
    df = get_rates(symbol=symbol, number_of_data=500, timeframe=timeframe)

    df = sma_diff(df, "close", sma_fast_period, slow_sma_period)
    df = rsi(df, "close", rsi_period)
    df = atr(df, atr_period)

    df = df.dropna()
    model = load(model_path)
    sc = load(sc_path)

    list_X = ["SMA_diff", "RSI", "ATR"]
    X = df[list_X]
    X_sc = sc.transform(X)

    predict_array = model.predict(X_sc)
    prediction = predict_array[-1]

    buy = True if prediction == 1 else False
    sell = True if prediction == -1 else False
    return buy, sell


def li_2023_02_TreePcaQuantile(symbol, timeframe, sma_fast_period, slow_sma_period, rsi_period, atr_period, model_path,
                               sc_path, pca_path):
    df = get_rates(symbol=symbol, number_of_data=500, timeframe=timeframe)

    df = sma_diff(df, "close", sma_fast_period, slow_sma_period)
    df = rsi(df, "close", rsi_period)
    df = previous_ret(df, "close", 60)
    df = sto_rsi(df, "close", 14)
    df = ichimoku(df, 27, 78)
    df = candle_information(df)
    df = atr(df, atr_period)
    df = df.fillna(value=0)

    df = df.dropna()
    model = load(model_path)
    sc = load(sc_path)
    pca = load(pca_path)

    list_X = ["SMA_diff", "RSI", "ATR", "candle_way", "filling", "amplitude", "SPAN_A", "SPAN_B", "BASE", "STO_RSI",
               "STO_RSI_D", "STO_RSI_K", "previous_ret"]
    X = df[list_X]
    X_sc = sc.transform(X)
    X_pca = pca.transform(X_sc)

    predict_array = model.predict(X_pca)
    prediction = predict_array[-1]

    buy = True if prediction == 1 else False
    sell = True if prediction == -1 else False
    return buy, sell
