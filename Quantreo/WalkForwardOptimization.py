import itertools
from Quantreo.Backtest import *


class WalkForwardOptimization:


    def __init__(self, data, TradingStrategy, fixed_parameters, parameters_range, length_train_set=10_000,
                 pct_train_set=.80, anchored=True, title=None):

        self.data = data
        self.TradingStrategy = TradingStrategy
        self.parameters_range = parameters_range
        self.fixed_parameters = fixed_parameters

        self.dictionaries = None
        self.get_combinations()

        self.length_train_set, self.pct_train_set = length_train_set, pct_train_set
        self.train_samples, self.test_samples, self.anchored = [], [], anchored

        self.BT, self.criterion = None, None

        self.best_params_sample_df, self.best_params_sample = None, None

        self.smooth_result = pd.DataFrame()
        self.best_params_smoothed = list()

        self.columns = list(self.parameters_range.keys())
        self.columns.append("criterion")
        self.df_results = pd.DataFrame(columns=self.columns)

        self.title_graph = title

    def get_combinations(self):
        # Create a list of dictionaries with all the possible combination (ONLY with variable parameters)
        keys = list(self.parameters_range.keys())
        combinations = list(itertools.product(*[self.parameters_range[key] for key in keys]))
        self.dictionaries = [dict(zip(keys, combination)) for combination in combinations]

        # We add the fixed parameters on each dictionary
        for dictionary in self.dictionaries:
            dictionary.update(self.fixed_parameters)

    def get_sub_samples(self):
        # Compute the length of the test set
        length_test = int(self.length_train_set / self.pct_train_set - self.length_train_set)

        # Initialize size parameters
        start = 0
        # We don't initialize the end with length_train+length_test because we add it in the loop
        end = self.length_train_set

        # We split the data until we can't make more than 2 sub-samples
        while (len(self.data) - end) > 2 * length_test:
            end += length_test

            # If we are at the last sample we take the whole rest to not create a tiny last sample
            if (len(self.data) - end) < 2 * length_test:
                end = len(self.data)

            # Fill the samples list depending on if there are anchored or not
            if self.anchored:
                # We store the train and test set in 2 list to make future computations easier
                self.train_samples.append(self.data.iloc[:end - length_test,:])
                self.test_samples.append(self.data.iloc[end - length_test: end, :])
            else:
                # We store the train and test set in 2 list to make future computations easier
                self.train_samples.append(self.data.iloc[start:end - length_test, :])
                self.test_samples.append(self.data.iloc[end - length_test: end, :])

            start += length_test

    def get_criterion(self, sample, params):
        # Backtest initialization with a specif dataset and set of parameters
        self.BT = Backtest(data=sample, TradingStrategy=self.TradingStrategy, parameters=params)

        # Compute the returns of the strategy (on this specific datasets and with these parameters)
        self.BT.run()

        # Calculation and storage of the criterion (Return over period over the maximum drawdown)
        ret, dd = self.BT.get_ret_dd()

        # We add ret and dd because dd < 0
        self.criterion = ret + 2*dd

    def get_best_params_train_set(self):
        # Store of the possible parameters combinations with the associated criterion
        # Here, we put the best criterion on the train set to find the best parameters BUT we will replace it
        # by the best criterion on the test set to be as close as possible to the reality
        storage_values_params = []

        for self.params_item in np.random.choice(self.dictionaries, int(len(self.dictionaries)*0.55)):
            # Extract the variables parameters from the dictionary
            current_params = [self.params_item[key] for key in list(self.parameters_range.keys())]

            # Compute the criterion and add it to the list of params
            self.get_criterion(self.train_sample, self.params_item)
            current_params.append(self.criterion)

            # We add the current_params list to the storage_values_params in order to create a dataframe
            storage_values_params.append(current_params)

        df_find_params = pd.DataFrame(storage_values_params, columns=self.columns)

        # Extract the dataframe line with the best parameters
        self.best_params_sample_df = df_find_params.sort_values(by="criterion", ascending=False).iloc[0:1, :]

        # !! We put the last index value as index
        # because WITHOUT that when you replace the criterion value later you will replace all value with the same index
        self.best_params_sample_df.index = self.train_sample.index[-2:-1]

        # We add the best params to the dataframe which contains all the best params for each period
        self.df_results = pd.concat((self.df_results, self.best_params_sample_df), axis=0)

        # Create a dictionary with the best params on the train set in order to test them on the test set later
        self.best_params_sample = dict(df_find_params.sort_values(by="criterion", ascending=False).iloc[0, :-1])
        self.best_params_sample.update(self.fixed_parameters)

    def get_smoother_result(self):
        self.smooth_result = pd.DataFrame()
        # For each column, we will extract the exp mean or the mode

        for column in self.df_results.columns:

            # If the values are float we compute the exponential mean of the columns to have smoother modifications
            if isinstance(self.df_results[column][0], (float, np.float64)) :
                self.smooth_result[column] = self.df_results[column].ewm(com=1.5, ignore_na=True).mean()

            # If the values are not float we search the mode of the columns
            else:
                self.smooth_result[column] = self.df_results[column].mode()

        # Create a dictionary with the best params SMOOTHED by exponential mean or by the mode
        test_params = dict(self.smooth_result.iloc[-1,:-1])
        test_params.update(self.fixed_parameters)

        # Run on train set to keep the best model in memory and not the last
        # !! We take the best params sample to train the model NOT the smoothed params
        self.get_criterion(self.train_sample, self.best_params_sample)

        return test_params

    def test_best_params(self):
        # Extract smoothed best params
        smooth_best_params = self.get_smoother_result()

        try:
            # Incorporate right path to the ML algo weights
            smooth_best_params["model_path"] = self.BT.TradingStrategy.saved_model_path
        except:
            pass

        try:
            # Incorporate right path to the ML algo weights
            smooth_best_params["sc_path"] = self.BT.TradingStrategy.saved_sc_path
        except:
            pass

        try:
            # Incorporate right path to the ML algo weights
            smooth_best_params["pca_path"] = self.BT.TradingStrategy.saved_pca_path
        except:
            pass

        # Remove train mode
        smooth_best_params["train_mode"] = False

        # Compute the criterion on the test set, using the smoothed best params
        self.get_criterion(self.test_sample, smooth_best_params)

        # !! Not necessary, but we replace the criterion train value by the criterion test value to do not create
        # a look-ahead bias
        self.df_results.at[self.df_results.index[-1], 'criterion'] = self.criterion

        self.best_params_smoothed.append(smooth_best_params)
        print(self.best_params_smoothed)

    def run_optimization(self):
        # Create the sub-samples
        self.get_sub_samples()

        # Run the optimization
        for self.train_sample, self.test_sample in zip(self.train_samples, self.test_samples):
            self.get_best_params_train_set()
            self.test_best_params()

    def display(self):
        # Empty dataframe that will be filled by the result on each period
        df_test_result = pd.DataFrame()

        for params, test in zip(self.best_params_smoothed, self.test_samples):

            # !! Here, we can call directly the model without run again the model
            # because hte model is saved used the date so, we keep the best one thanks to get_smoother_result function
            # when we compute the weights for the best params smoothed
            self.BT = Backtest(data=test, TradingStrategy=self.TradingStrategy, parameters=params)
            self.BT.run()
            df_test_result = pd.concat((df_test_result, self.BT.data), axis=0)

        # Print the backtest for the period following the walk-forward method
        self.BT = Backtest(data=df_test_result, TradingStrategy=self.TradingStrategy, parameters=params, title=self.title_graph)
        self.BT.display()