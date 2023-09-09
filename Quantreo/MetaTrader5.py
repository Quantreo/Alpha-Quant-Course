from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import MetaTrader5 as mt5

timeframes_mapping={
    "1-minute": [mt5.TIMEFRAME_M1, 1],
    "2-minutes": [mt5.TIMEFRAME_M2, 2],
    "3-minutes": [mt5.TIMEFRAME_M3, 3],
    "4-minutes": [mt5.TIMEFRAME_M4, 4],
    "5-minutes": [mt5.TIMEFRAME_M5, 5],
    "6-minutes": [mt5.TIMEFRAME_M6, 6],
    "10-minutes": [mt5.TIMEFRAME_M10, 10],
    "12-minutes": [mt5.TIMEFRAME_M12, 12],
    "15-minutes": [mt5.TIMEFRAME_M15, 15],
    "30-minutes": [mt5.TIMEFRAME_M30, 30],
    "1-hour": [mt5.TIMEFRAME_H1, 60],
    "2-hours": [mt5.TIMEFRAME_H2, 120],
    "3-hours": [mt5.TIMEFRAME_H3, 180],
    "4-hours": [mt5.TIMEFRAME_H4, 240],
    "6-hours": [mt5.TIMEFRAME_H6, 360],
    "8-hours": [mt5.TIMEFRAME_H8, 420],
    "12-hours": [mt5.TIMEFRAME_H12, 720],
    "1-day": [mt5.TIMEFRAME_D1, 1440]
}


def get_verification_time(timeframe:int):
    """
    You need to put the timeframe in minute.
    EX: 8H = 8*60 = 640| 2H = 2*60 = 120
    """
    start_time = datetime(year=2021,month=1,day=1,hour=0, minute=0) - timedelta(seconds=2)
    end_time = datetime(year=2021,month=1,day=1,hour=23, minute=59, second=59)

    time_list = [start_time.strftime("%H:%M:%S")]
    current_time = start_time
    while current_time <= end_time:
        current_time += timedelta(minutes=timeframe)
        time_list.append(current_time.strftime("%H:%M:%S"))
    del time_list[0]
    del time_list[-1]

    return time_list


def get_rates(symbol, number_of_data=10000, timeframe=mt5.TIMEFRAME_D1):
    # Compute now date
    from_date = datetime.now()

    # Extract n rates before now
    rates = mt5.copy_rates_from(symbol, timeframe, from_date, number_of_data)

    # Transform Tuple into a DataFrame
    df_rates = pd.DataFrame(rates)

    # Convert number format of the date into date format
    df_rates["time"] = pd.to_datetime(df_rates["time"], unit="s")

    # Put the "time" column as index
    df_rates = df_rates.set_index("time")

    return df_rates


def find_filling_mode(symbol):
    """
    The MetaTrader5 library doesn't find the filling mode correctly for a lot of brokers
    """
    for i in range(2):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": mt5.symbol_info(symbol).volume_min,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(symbol).ask,
            "type_filling": i,
            "type_time": mt5.ORDER_TIME_GTC}

        result = mt5.order_check(request)

        if result.comment == "Done":
            break

    return i


def send_order(symbol, lot, buy, sell, id_position=None, pct_tp=0.02, pct_sl=0.01, comment=" No specific comment",
               magic=0):
    # Initialize the bound between MT5 and Python
    mt5.initialize()

    # Extract filling_mode
    filling_type = find_filling_mode(symbol)

    """ OPEN A TRADE """
    if buy and id_position is None:
        ask_price = mt5.symbol_info_tick(symbol).ask
        tp_price = (1+pct_tp) * ask_price
        sl_price = (1-pct_sl) * ask_price

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY,
            "price": ask_price,
            "deviation": 10,
            "tp": tp_price,
            "sl": sl_price,
            "magic": magic,
            "comment": comment,
            "type_filling": filling_type,
            "type_time": mt5.ORDER_TIME_GTC}

        result = mt5.order_send(request)

        print(mt5.symbol_info_tick(symbol).ask, tp_price, sl_price)
        return result

    if sell and id_position is None:
        bid_price = mt5.symbol_info_tick(symbol).bid
        tp_price = (1 - pct_tp) * bid_price
        sl_price = (1 + pct_sl) * bid_price

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_SELL,
            "price": mt5.symbol_info_tick(symbol).bid,
            "deviation": 10,
            "tp": tp_price,
            "sl": sl_price,
            "magic": magic,
            "comment": comment,
            "type_filling": filling_type,
            "type_time": mt5.ORDER_TIME_GTC}

        result = mt5.order_send(request)

        print(mt5.symbol_info_tick(symbol).bid, tp_price, sl_price)
        return result

    """ CLOSE A TRADE """
    if buy and id_position is not None:
        bid_price = mt5.symbol_info_tick(symbol).bid
        request = {
            "position": id_position,
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_SELL,
            "price": bid_price,
            "deviation": 10,
            "magic": magic,
            "comment": comment,
            "type_filling": filling_type,
            "type_time": mt5.ORDER_TIME_GTC}

        result = mt5.order_send(request)
        return result

    if sell and id_position is not None:
        ask_price = mt5.symbol_info_tick(symbol).ask
        request = {
            "position": id_position,
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY,
            "price": ask_price,
            "deviation": 10,
            "magic": magic,
            "comment": comment,
            "type_filling": filling_type,
            "type_time": mt5.ORDER_TIME_GTC}

        result = mt5.order_send(request)
        return result


def resume():
    """ Return the current positions. Position=0 --> Buy """
    # Define the name of the columns that we will create
    columns_list = ["ticket", "position", "symbol", "volume", "magic", "profit", "price", "tp", "sl", "trade_size"]

    # Go take the current open trades
    positions_list = mt5.positions_get()

    # Create a empty dataframe
    summary = pd.DataFrame()

    # Loop to add each row in dataframe
    for element in positions_list:
        element_pandas = pd.DataFrame([element.ticket, element.type, element.symbol, element.volume, element.magic,
                                       element.profit, element.price_open, element.tp,
                                       element.sl, mt5.symbol_info(element.symbol).trade_contract_size],
                                      index=columns_list).transpose()
        summary = pd.concat((summary, element_pandas), axis=0)

    try:
        summary["profit %"] = summary.profit / (summary.price * summary.trade_size * summary.volume)
        summary = summary.reset_index(drop=True)
    except:
        pass
    return summary


def run(symbol, buy, sell, lot, pct_tp=0.02, pct_sl=0.01, comment="", magic=23400):

    # Initialize the connection
    mt5.initialize()

    # Choose your  symbol
    print("------------------------------------------------------------------")
    print("Date: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\tSYMBOL:", symbol)

    # Initialize the device
    orders = resume()

    # Buy or sell
    print(f"BUY: {buy} \t  SELL: {sell}")

    """ Close trade eventually """
    # Extraction type trade
    try:
        position_list = orders.loc[orders["symbol"] == symbol]
        identifier_list = orders.loc[orders["symbol"] == symbol]
        
        position = position_list.loc[position_list["magic"]==magic].values[0][1]
        identifier = identifier_list.loc[identifier_list["magic"] == magic].values[0][1]
        
    except:
        position = None
        identifier = None

    if position is not None:
        print(f"POSITION: {position} \t ID: {identifier}")

    # Verif trades
    if buy == True and position == 0:
        buy = False

    elif buy == False and position == 0:
        before = mt5.account_info().balance
        res = send_order(symbol, lot, True, False, id_position=identifier, pct_tp=pct_tp, pct_sl=pct_sl,
                             comment=comment, magic=magic)
        after = mt5.account_info().balance

        print(f"CLOSE BUY POSITION: {res.comment}")
        pct = np.round(100 * (after - before) / before, 3)

        if res.comment != "Request executed":
            print("WARNINGS", res.comment)

    elif sell == True and position == 1:
        sell = False

    elif sell == False and position == 1:
        before = mt5.account_info().balance
        res = send_order(symbol, lot, False, True, id_position=identifier, pct_tp=pct_tp, pct_sl=pct_sl,
                             comment=comment, magic=magic)
        print(f"CLOSE SELL POSITION: {res.comment}")
        after = mt5.account_info().balance

        pct = np.round(100 * (after - before) / before, 3)
        if res.comment != "Request executed":
            print("WARNINGS", res.comment)
    else:
        pass

    """ Buy or Sell """
    if buy == True:
        res = send_order(symbol, lot, True, False, id_position=None, pct_tp=pct_tp, pct_sl=pct_sl,
                             comment=comment, magic=magic)
        print(f"OPEN BUY POSITION: {res.comment}")
        if res.comment != "Request executed":
            print("WARNINGS", res.comment)

    if sell == True:
        res = send_order(symbol, lot, False, True, id_position=None, pct_tp=pct_tp, pct_sl=pct_sl,
                             comment=comment, magic=magic)
        print(f"OPEN SELL POSITION: {res.comment}")
        if res.comment != "Request executed":
            print("WARNINGS", res.comment)
    print("------------------------------------------------------------------")
