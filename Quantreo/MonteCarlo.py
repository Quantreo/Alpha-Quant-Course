from Quantreo.Backtest import *
from datetime import timedelta
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class MonteCarlo:
    """
    A class to simulate financial market paths based on historical data and assess strategies via Monte Carlo simulations.

    This class generates synthetic market data paths based on historical statistics, such as price movements and time intervals,
    to provide a broader base for strategy testing. It allows for backtesting trading strategies on numerous simulated
    paths to evaluate performance and resilience under varied market conditions.

    The simulation approach helps identify strategies with potential overfitting to historical data and tests strategy
    robustness in unseen market scenarios.

    Parameters
    ----------
    data : DataFrame
        The historical market data used as a basis for generating synthetic paths. This data should include at least
        open, low, high, and close prices, alongside any additional features required by the trading strategy.

    TradingStrategy : object
        An instance of the trading strategy class to be backtested. The strategy should define how trades are executed
        based on input data.

    parameters : dict
        A dictionary of parameters for the trading strategy. These parameters can include any strategy-specific
        settings that will remain constant throughout the simulation.

    raw_columns : list of str, optional
        A list of additional column names from the historical data that should be included in the synthetic data
        generation. These could be used by the trading strategy in decision-making.

    discount_calmar_ratio : int
        Number of candles in a year to create the right discount for the calmar ratio

    """

    def __init__(self, data, TradingStrategy, parameters, raw_columns=[], discount_calmar_ratio=252):
        # Set Initial parameters
        self.data = data
        self.TradingStrategy = TradingStrategy
        self.parameters = parameters
        self.raw_columns = raw_columns
        self.discount_calmar_ratio = discount_calmar_ratio
        self.paths = []
        self.returns, self.drawdowns = [], []

    def generate_path(self, number_observation=1000):
        df = self.data.copy()

        # Compute the variation from open to low,high,close
        df["pct_open_low"] = (df["low"] - df["open"]) / df["open"]
        df["pct_open_high"] = (df["high"] - df["open"]) / df["open"]
        df["pct_open_close"] = (df["close"] - df["open"]) / df["open"]

        # Compute the candles lenght in second (essential for non fix time bars)
        time_index = list((df.index[1:] - df.index[:-1]).total_seconds())

        # Replace the last value by the most frequent one
        mode = statistics.multimode(time_index)
        time_index.insert(0, mode[0])

        # Create columns with the length of the bar and how much second you need to touch the low and high time
        df["time_variation"] = time_index
        df["var_low_time"] = (pd.to_datetime(df["low_time"]) - df.index).map(lambda x: x.total_seconds())
        df["var_high_time"] = (pd.to_datetime(df["high_time"]) - df.index).map(lambda x: x.total_seconds())

        # Create a list that we will use to generate new data from our sample
        data = []
        for i in range(len(df)):
            row_values = [df["pct_open_low"].iloc[i], df["pct_open_high"].iloc[i], df["pct_open_close"].iloc[i],
                          df["time_variation"].iloc[i], df["var_low_time"].iloc[i], df["var_high_time"].iloc[i]]

            for col in self.raw_columns:
                row_values.append(df[col].iloc[i])
            data.append(row_values)

        # Grab the last open price and data
        start_price, start_date = df["open"].iloc[-1], df.index[-1]
        open_price, current_date = start_price, start_date

        # Create an empty list for our simulated data
        data_new = []
        for _ in range(number_observation):

            # Take our sample and extract one
            row_values = random.choice(data)
            pct_open_low, pct_open_high, pct_open_close = row_values[0], row_values[1], row_values[2]
            time_variation, var_low_time, var_high_time = row_values[3], row_values[4], row_values[5]

            # Extract raw data
            raw_data = [row_values[6 + i] for i in range(len(self.raw_columns))]

            # Compute date (index + low & high time)
            current_date += timedelta(seconds=time_variation)
            low_time = current_date + timedelta(seconds=var_low_time)
            high_time = current_date + timedelta(seconds=var_high_time)

            # Compute prices
            low_price = open_price * (1 + pct_open_low)
            high_price = open_price * (1 + pct_open_high)
            close_price = open_price * (1 + pct_open_close)

            # Verify low ≤ close & close ≤ high (We are never too careful)
            if close_price < low_price:
                low_price = close_price

            if high_price < close_price:
                high_price = close_price

            # Add our new variables into the list
            row_data_new = [open_price, low_price, high_price, close_price, current_date, low_time, high_time]

            # Add raw data
            row_data_new.extend(raw_data)
            data_new.append(row_data_new)

            # Update the next open price being the close price
            # You can add a gap based on the gap distribution of our asset if you want here
            open_price = close_price

        columns_list = ["open", "low", "high", "close", "time", "low_time", "high_time"]
        columns_list.extend(self.raw_columns)

        df_simulated = pd.DataFrame(data_new, columns=columns_list)
        df_simulated = df_simulated.set_index("time")

        return df_simulated


    def generate_paths(self, number_simulations=100, number_observation=1000):
        # We use the generate_path function to generate N different paths
        for _ in range(number_simulations):

            # Generate a path
            df_sim = self.generate_path(number_observation = number_observation)

            # Add it into our paths list
            self.paths.append(df_sim)


    def backtest_paths(self):
        # If we do not have any paths into the paths list we run the generate_paths function to create some
        if len(self.paths) == 0:
            self.generate_paths()

        # Run the backtest for each path
        for df_path in tqdm(self.paths):

            # Initialiaze the backtest
            BT = Backtest(data=df_path, TradingStrategy=self.TradingStrategy, parameters=self.parameters)

            # Compute the returns of the strategy (on this specific datasets and with these parameters)
            BT.run()

            # Calculation and storage of the criterion (Return over period over the maximum drawdown)
            ret, dd = BT.get_ret_dd()
            self.returns.append(ret)
            self.drawdowns.append(dd)


    def display_results(self):
        # If we do not have any return we run the backtest
        if len(self.returns) == 0:
            self.backtest_paths()

        # We compute the Calmar Ratio for each path
        ret_dd = [return_ / np.abs(dd) / (len(self.paths[0]) / self.discount_calmar_ratio) for return_, dd in zip(self.returns, self.drawdowns)]

        # We set up the figure (3 histograms)
        fig, axs = plt.subplots(3, 1, figsize=(15, 10))

        # Returns histogram
        axs[0].hist(self.returns, color="#289E41", bins=40, alpha=0.7, edgecolor="black")
        axs[0].set_title('Return Distribution %')
        axs[0].grid(axis='y', linestyle='-', alpha=0.5, color='lightgrey')

        # Drawdown histogram
        axs[1].hist(self.drawdowns, color="#9E2828", bins=40, alpha=0.7, edgecolor="black")
        axs[1].set_title('Drawdown Distribution %')
        axs[1].grid(axis='y', linestyle='-', alpha=0.5, color='lightgrey')

        # Calmar Ratio histogram
        axs[2].hist(ret_dd, color="#28709E", bins=40, alpha=0.7, edgecolor="black")
        axs[2].set_title('Calmar Ratio Distribution')
        axs[2].grid(axis='y', linestyle='-', alpha=0.5, color='lightgrey')
        plt.subplots_adjust(hspace=0.5)

        # Plot the graph
        plt.show()
