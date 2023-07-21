"""
Strategy Explanation:
- We use the Stochastic RSI to find the force of the market and the difference between D stochastic and K stochastic
to find short term movement

- We use the ATR to compute our Take-profit & Stop-loss dynamically
"""

from Quantreo.DataPreprocessing import *


class StoRsiAtr:

    def __init__(self, data, parameters):
        # Set parameters
        self.data = data
        self.sto_period, self.atr_period = parameters["sto_period"], parameters["atr_period"]
        self.tp, self.sl = None, None
        self.cost = parameters["cost"]
        self.leverage = parameters["leverage"]
        self.get_features()
        self.start_date_backtest = self.data.index[0]

        # Get Entry parameters
        self.buy, self.sell = False, False
        self.open_buy_price, self.open_sell_price = None, None
        self.entry_time, self.exit_time = None, None

        # Get exit parameters
        self.var_buy_high, self.var_sell_high = None, None
        self.var_buy_low, self.var_sell_low = None, None

        self.output_dictionary = parameters.copy()

    def get_features(self):
        self.data = sto_rsi(self.data, "close", self.sto_period)

        # Compute the TP-SL threshold in %
        self.data = atr(self.data, self.atr_period)
        self.data["TP_level"] = self.data["open"] + 2 * self.data["ATR"].shift(1)
        self.data["SL_level"] = self.data["open"] - 2 * self.data["ATR"].shift(1)
        self.data["TP_pct"] = (self.data["TP_level"] - self.data["open"]) / self.data["open"]
        self.data["SL_pct"] = (self.data["SL_level"] - self.data["open"]) / self.data["open"]

        # def signal
        self.data["signal"] = 0
        condition_1_buy = (self.data["STO_RSI_K"] < 30) & (self.data["STO_RSI_D"] < 30)
        condition_1_sell = (self.data["STO_RSI_K"] > 70) & (self.data["STO_RSI_D"] > 70)

        condition_2_buy = (self.data["STO_RSI_K"] < self.data["STO_RSI_D"]) & (self.data["STO_RSI_K"].shift(1) > self.data["STO_RSI_D"].shift(1))
        condition_2_sell = (self.data["STO_RSI_K"] > self.data["STO_RSI_D"]) & (self.data["STO_RSI_K"].shift(1) < self.data["STO_RSI_D"].shift(1))

        self.data.loc[condition_1_buy & condition_2_buy, "signal"] = 1
        self.data.loc[condition_1_sell & condition_2_sell, "signal"] = -1

    def get_entry_signal(self, time):
        """
        Entry signal
        :param time: TimeStamp of the row
        :return: Entry signal of the row and entry time
        """
        if len(self.data.loc[:time]) < 2:
            return 0, self.entry_time

        # Create entry signal --> -1,0,1
        entry_signal = 0
        if self.data.loc[:time]["signal"][-2] == 1:
            entry_signal = 1
        elif self.data.loc[:time]["signal"][-2] == -1:
            entry_signal = -1

        # Enter in buy position only if we want to, and we aren't already
        if entry_signal == 1 and not self.buy and not self.sell:
            self.buy = True
            self.open_buy_price = self.data.loc[time]["open"]
            self.tp, self.sl = self.data.loc[time]["TP_pct"], self.data.loc[time]["SL_pct"]
            print(time,self.tp*100, self.sl*100)
            self.entry_time = time

        # Enter in sell position only if we want to, and we aren't already
        elif entry_signal == -1 and not self.sell and not self.buy:
            self.sell = True
            self.open_sell_price = self.data.loc[time]["open"]
            self.tp, self.sl = self.data.loc[time]["TP_pct"], self.data.loc[time]["SL_pct"]
            print(time, self.tp*100, self.sl*100)
            self.entry_time = time

        else:
            entry_signal = 0

        return entry_signal, self.entry_time

    def get_exit_signal(self, time):
        """
        Take-profit & Stop-loss exit signal
        :param time: TimeStamp of the row
        :return: P&L of the position IF we close it

        **ATTENTION**: If you allow your bot to take a buy and a sell position in the same time,
        you need to return 2 values position_return_buy AND position_return_sell (and sum both for each day)
        """
        # Verify if we need to close a position and update the variations IF we are in a buy position
        if self.buy:
            self.var_buy_high = (self.data.loc[time]["high"] - self.open_buy_price) / self.open_buy_price
            self.var_buy_low = (self.data.loc[time]["low"] - self.open_buy_price) / self.open_buy_price

            # Let's check if AT LEAST one of our threshold are touched on this row
            if (self.tp < self.var_buy_high) and (self.var_buy_low < self.sl):

                # Close with a positive P&L if high_time is before low_time
                if self.data.loc[time]["high_time"] < self.data.loc[time]["low_time"]:
                    self.buy = False
                    self.open_buy_price = None
                    position_return_buy = (self.tp - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_buy, self.exit_time

                # Close with a negative P&L if low_time is before high_time
                elif self.data.loc[time]["low_time"] < self.data.loc[time]["high_time"]:
                    self.buy = False
                    self.open_buy_price = None
                    position_return_buy = (self.sl - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_buy, self.exit_time

                else:
                    self.buy = False
                    self.open_buy_price = None
                    position_return_buy = 0
                    self.exit_time = time
                    return position_return_buy, self.exit_time

            elif self.tp < self.var_buy_high:
                self.buy = False
                self.open_buy_price = None
                position_return_buy = (self.tp - self.cost) * self.leverage
                self.exit_time = time
                return position_return_buy, self.exit_time

            # Close with a negative P&L if low_time is before high_time
            elif self.var_buy_low < self.sl:
                self.buy = False
                self.open_buy_price = None
                position_return_buy = (self.sl - self.cost) * self.leverage
                self.exit_time = time
                return position_return_buy, self.exit_time

        # Verify if we need to close a position and update the variations IF we are in a sell position
        if self.sell:
            self.var_sell_high = -(self.data.loc[time]["high"] - self.open_sell_price) / self.open_sell_price
            self.var_sell_low = -(self.data.loc[time]["low"] - self.open_sell_price) / self.open_sell_price

            # Let's check if AT LEAST one of our threshold are touched on this row
            if (self.tp < self.var_sell_low) and (self.var_sell_high < self.sl):

                # Close with a positive P&L if high_time is before low_time
                if self.data.loc[time]["low_time"] < self.data.loc[time]["high_time"]:
                    self.sell = False
                    self.open_sell_price = None
                    position_return_sell = (self.tp - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_sell, self.exit_time

                # Close with a negative P&L if low_time is before high_time
                elif self.data.loc[time]["high_time"] < self.data.loc[time]["low_time"]:
                    self.sell = False
                    self.open_sell_price = None
                    position_return_sell = (self.sl - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_sell, self.exit_time

                else:
                    self.sell = False
                    self.open_sell_price = None
                    position_return_sell = 0
                    self.exit_time = time
                    return position_return_sell, self.exit_time

            # Close with a positive P&L if high_time is before low_time
            elif self.tp < self.var_sell_low:
                self.sell = False
                self.open_sell_price = None
                position_return_sell = (self.tp - self.cost) * self.leverage
                self.exit_time = time
                return position_return_sell, self.exit_time

            # Close with a negative P&L if low_time is before high_time
            elif self.var_sell_high < self.sl:
                self.sell = False
                self.open_sell_price = None
                position_return_sell = (self.sl - self.cost) * self.leverage
                self.exit_time = time
                return position_return_sell, self.exit_time

        return 0, None
