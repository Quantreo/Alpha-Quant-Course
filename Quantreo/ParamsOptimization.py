"""
The ParamsOptimization class is the first bloc of our future WalkForwardOptimization (the best way to train & test you algo)
"""

import pandas as pd
from Quantreo.Backtest import *
import itertools


class ParamsOptimization:

    def __init__(self, data, TradingStrategy, fixed_parameters, parameters_range):

        self.data = data
        self.TradingStrategy = TradingStrategy
        self.parameters_range = parameters_range
        self.fixed_parameters = fixed_parameters

        self.dictionaries = None
        self.get_combinations()

        self.BT, self.criterion = None, None

        self.best_params_sample_df, self.best_params_sample = None, None

        self.columns = list(self.parameters_range.keys())
        self.columns.append("criterion")

    def get_combinations(self):
        # Create a list of dictionaries with all the possible combination (ONLY with variable parameters)
        keys = list(self.parameters_range.keys())
        combinations = list(itertools.product(*[self.parameters_range[key] for key in keys]))
        self.dictionaries = [dict(zip(keys, combination)) for combination in combinations]

        # We add the fixed parameters on each dictionary
        for dictionary in self.dictionaries:
            dictionary.update(self.fixed_parameters)

    def get_criterion(self, sample, params):
        # Backtest initialization with a specif dataset and set of parameters
        self.BT = Backtest(data=sample, TradingStrategy=self.TradingStrategy, parameters=params)

        # Compute the returns of the strategy (on this specific datasets and with these parameters)
        self.BT.run()

        # Calculation and storage of the criterion (Return over period over the maximum drawdown)
        ret, dd = self.BT.get_ret_dd()
        self.criterion = ret / dd

    def get_best_params_train_set(self):
        # Store of the possible parameters combinations with the associated criterion
        # Here, we put the best criterion on the train set to find the best parameters BUT we will replace it
        # by the best criterion on the test set to be as close as possible to the reality
        storage_values_params = []

        for self.params_item in self.dictionaries:
            # Extract the variables parameters from the dictionary
            current_params = [self.params_item[key] for key in list(self.parameters_range.keys())]

            # Compute the criterion and add it to the list of params
            self.get_criterion(self.data, self.params_item)
            current_params.append(self.criterion)

            # We add the current_params list to the storage_values_params in order to create a dataframe
            storage_values_params.append(current_params)

        df_find_params = pd.DataFrame(storage_values_params, columns=self.columns)

        # Extract the dataframe line with the best parameters
        self.best_params_sample_df = df_find_params.sort_values(by="criterion", ascending=False).iloc[0:1, :]

        # Create a dictionary with the best params on the train set in order to test them on the test set later
        self.best_params_sample = dict(df_find_params.sort_values(by="criterion", ascending=False).iloc[0, :-1])
        self.best_params_sample.update(self.fixed_parameters)

