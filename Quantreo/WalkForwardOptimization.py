import itertools
from Quantreo.Backtest import *


class WalkForwardOptimization:
    """
    A class for performing Walk-Forward Optimization on a trading strategy.

    This class is responsible for finding the optimal set of parameters for a
    trading strategy by dividing a dataset into multiple training and test sets
    and running the strategy on each one.

    This method of optimization helps prevent curve fitting by ensuring that the
    strategy performs well over many different time periods and under various market conditions.

    Parameters
    ----------
    data: DataFrame
        The input data to be used for the backtests.

    TradingStrategy: object
        The trading strategy to be optimized.

    fixed_parameters: dict
        The parameters of the strategy that should remain fixed throughout the optimization process.

    parameters_range: dict
        The range of values that the non-fixed parameters of the strategy should take on during the optimization process.

    length_train_set: int, default 10_000
        The size of the training set in number of data points.

    pct_train_set: float, default .80
        The proportion of the dataset to be used for training.

    anchored: bool, default True
        Whether the training set should be anchored, meaning it always begins at the start of the dataset.
        If False, the training set will move forward in time with the test set.

    title: str, default None
        The title of the backtest's plot.

    randomness: float, default 0.75
        A factor to determine the size of the sample space for parameter combinations to be tested.

    """

    def __init__(self, data, TradingStrategy, fixed_parameters, parameters_range, length_train_set=10_000,
                 pct_train_set=.80, anchored=True, title=None, randomness=0.75):
        # Set initial parameters
        self.data = data
        self.TradingStrategy = TradingStrategy
        self.parameters_range = parameters_range
        self.fixed_parameters = fixed_parameters
        self.randomness = randomness
        self.dictionaries = None
        self.get_combinations()

        # Necessary variables to create our sub-samples
        self.length_train_set, self.pct_train_set = length_train_set, pct_train_set
        self.train_samples, self.test_samples, self.anchored = [], [], anchored

        # Necessary variables to compute and store our criteria
        self.BT, self.criterion = None, None
        self.best_params_sample_df, self.best_params_sample = None, None
        self.smooth_result = pd.DataFrame()
        self.best_params_smoothed = list()

        # Create dataframe that will contain the optimal parameters  (ranging parameters + criterion)
        self.columns = list(self.parameters_range.keys())
        self.columns.append("criterion")
        self.df_results = pd.DataFrame(columns=self.columns)

        # Set the title of our Backtest plot
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
                # Fill the samples list depending on if there are anchored or not
                if self.anchored:
                    # We store the train and test set in 2 list to make future computations easier
                    self.train_samples.append(self.data.iloc[:end - length_test, :])
                    self.test_samples.append(self.data.iloc[end - length_test: len(self.data), :])
                else:
                    # We store the train and test set in 2 list to make future computations easier
                    self.train_samples.append(self.data.iloc[start:end - length_test, :])
                    self.test_samples.append(self.data.iloc[end - length_test: len(self.data), :])
                break

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

        for self.params_item in np.random.choice(self.dictionaries, size=int(len(self.dictionaries)*self.randomness), replace=False):
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

        # Replace the ranging parameters by the smoothed parameters
        for key in test_params.keys():
            self.best_params_sample[key] = test_params[key]

        # New way to keep the ML algo weights in memory
        # We initialize the strategy class to train the weights if it is necessary
        Strategy = self.TradingStrategy(self.train_sample, self.best_params_sample)

        # Extract the output dictionary parameters
        output_params = Strategy.output_dictionary

        return output_params

    def test_best_params(self):
        # Extract smoothed best params
        smooth_best_params = self.get_smoother_result()

        # Compute the criterion on the test set, using the smoothed best params
        self.get_criterion(self.test_sample, smooth_best_params)

        # We replace the criterion train value by the criterion test value to do not create
        self.df_results.at[self.df_results.index[-1], 'criterion'] = self.criterion
        self.best_params_smoothed.append(smooth_best_params)

    def run_optimization(self):
        # Create the sub-samples
        self.get_sub_samples()

        # Run the optimization
        for self.train_sample, self.test_sample in tqdm(zip(self.train_samples, self.test_samples)):
            self.get_best_params_train_set()
            self.test_best_params()

    def display(self):
        # Empty dataframe that will be filled by the result on each period
        df_test_result = pd.DataFrame()

        for params, test in zip(self.best_params_smoothed, self.test_samples):
            # !! Here, we can call directly the model without run again the model because the optimal weights are
            # computed already and stored into the output dictionary and so in the self.best_params_smoothed list
            self.BT = Backtest(data=test, TradingStrategy=self.TradingStrategy, parameters=params)
            self.BT.run()
            df_test_result = pd.concat((df_test_result, self.BT.data), axis=0)

        # Print the backtest for the period following the walk-forward method
        self.BT = Backtest(data=df_test_result, TradingStrategy=self.TradingStrategy, parameters=params)
        self.BT.display(self.title_graph)
