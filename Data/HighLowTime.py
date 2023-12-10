import numpy as np
from tqdm import tqdm
import pandas as pd

def find_timestamp_extremum(df, df_lower_timeframe):
    """
    :param: df_lowest_timeframe
    :return: self._data with three new columns: Low_time (TimeStamp), High_time (TimeStamp), High_first (Boolean)
    """
    df = df.copy()
    df = df.loc[df_lower_timeframe.index[0]:]

    # Set new columns
    df["low_time"] = np.nan
    df["high_time"] = np.nan

    # Loop to find out which of the high or low appears first
    for i in tqdm(range(len(df) - 1)):

        # Extract values from the lowest timeframe dataframe
        start = df.iloc[i:i + 1].index[0]
        end = df.iloc[i + 1:i + 2].index[0]
        row_lowest_timeframe = df_lower_timeframe.loc[start:end].iloc[:-1]

        # Extract Timestamp of the max and min over the period (highest timeframe)
        try:
            high = row_lowest_timeframe["high"].idxmax()
            low = row_lowest_timeframe["low"].idxmin()

            df.loc[start, "low_time"] = low
            df.loc[start, "high_time"] = high

        except Exception as e:
            print(e)
            df.loc[start, "low_time"] = None
            df.loc[start, "high_time"] = None

    # Verify the number of row without both TP and SL on same time
    percentage_good_row = len(df.dropna()) / len(df) * 100
    percentage_garbage_row = 100 - percentage_good_row

    # if percentage_garbage_row<95:
    print(f"WARNINGS: Garbage row: {'%.2f' % percentage_garbage_row} %")

    df = df.iloc[:-1]

    return df


df_low_tf = pd.read_csv("FixTimeBars/AUDUSD_30M_Admiral.csv", index_col="time", parse_dates=True)
df_high_tf = pd.read_csv("FixTimeBars/AUDUSD_4H_Admiral.csv", index_col="time", parse_dates=True)

df = find_timestamp_extremum(df_high_tf, df_low_tf)

print(df[["high_time", "low_time"]])
df.to_csv("FixTimeBars/AUDUSD_4H_Admiral_READY.csv")


