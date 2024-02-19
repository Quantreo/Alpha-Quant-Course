import pandas as pd
import numpy as np
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from Quantreo.Backtest import *
import itertools
import warnings
warnings.filterwarnings("ignore")


class CombinatorialPurgedCV:
    """
    Class for Combinatorial Purged Cross Validation (CPCV). This class splits a dataset
    into N partitions for training and testing, applying a purge to prevent overfitting
    based on temporal information.


    Parameters
    --------------------
    data : DataFrame
        The DataFrame containing the data to be split for training and testing.
    TradingStrategy : class
        The class defining the trading strategy to be evaluated.
    fixed_parameters : dict
        A dictionary with the parameters of the TradingStrategy that will remain fixed during the cross-validation.
    parameters_range : dict
        A dictionary with the parameters of the TradingStrategy that will be optimized. Each parameter should be
        associated with a range or a list of possible values.
    N : int, optional
        The number of partitions to create. Defaults to 10.
    k : int, optional
        The number of partitions to use for testing. Defaults to 2.
    purge_pct : float, optional
        The percentage of the data to be purged between training and testing sets. Defaults to 0.10.

    """

    def __init__(self, data, TradingStrategy, fixed_parameters, parameters_range, N=10, k=2, purge_pct=0.10):
        # Set initial parameters
        self.data = data
        self.TradingStrategy = TradingStrategy
        self.fixed_parameters = fixed_parameters
        self.parameters_range = parameters_range
        self.N = N
        self.k = k
        self.purge_pct = purge_pct

        # Necessary variables to compute and store our criteria
        self.BT, self.criterion = None, None
        self.dictionaries = None
        self.best_params_sample_df, self.best_params_sample = None, None
        self.dfs_list_pbo = []
        self.smooth_result = pd.DataFrame()
        self.best_params_smoothed = list()
        self.counter = 1
        self.lambdas = []
        self.train_sample, self.test_sample, self.output_params = None, None, None
        self.lists, self.df_lists = None, None
        self.lmb_series, self.pbo = None, None

        # Create dataframe that will contain the optimal parameters  (ranging parameters + criteria)
        self.columns = list(self.parameters_range.keys())
        self.columns.append("criterion_train")
        self.columns.append("criterion_test")
        self.df_results = pd.DataFrame(columns=self.columns)
        self.train_df_list, self.test_df_list = None, None
        self.plots = {}

    def get_combinations(self):
        # Create a list of dictionaries with all the possible combination (ONLY with variable parameters)
        keys = list(self.parameters_range.keys())
        combinations = list(itertools.product(*[self.parameters_range[key] for key in keys]))
        self.dictionaries = [dict(zip(keys, combination)) for combination in combinations]

        # We add the fixed parameters on each dictionary
        for dictionary in self.dictionaries:
            dictionary.update(self.fixed_parameters)

    def get_index_samples(self):
        # Index of samples
        nb_set = list(range(self.N))

        # Generate all the combinations between k among N
        combinations_test = list(combinations(nb_set, self.k))

        # Generate the complementary of the combinations (train set)
        combinations_train = [list(set(nb_set) - set(combinaisons_test)) for combinaisons_test in combinations_test]
        self.lists = []

        # Create a list with the test index and the train index
        for i in range(len(combinations_test)):
            self.lists.append([list(combinations_test[i]), combinations_train[i]])

    def get_sub_samples(self):
        # Create an equal division of the data (N samples)
        split_data = np.array_split(self.data, self.N)

        # List df train & test couple
        self.df_lists = []

        # STEP 1: Reorganize and add the purge into the sets
        for i in range(len(self.lists)):
            # Extract the idx for each sub-samples couple
            list_sets = self.lists[i]
            test_idx = list_sets[0]
            train_idx = list_sets[1]

            # Create list containing the su-samples for each period
            test_sets = [split_data[i] for i in test_idx]
            train_sets = []

            # Create the Purge & embargo periods
            for j in train_idx:
                train_df_ind = split_data[j]

                # Remove the beginning or the end if the train is across a test set
                if (j - 1 in test_idx) and (j + 1 in test_idx):
                    split_embargo = 2 * int(len(train_df_ind) * self.purge_pct)
                    split_purge = int(len(train_df_ind) * self.purge_pct)
                    train_df_ind = train_df_ind.iloc[split_embargo:-split_purge, :]
                elif j + 1 in test_idx:
                    split_purge = int(len(train_df_ind) * self.purge_pct)
                    train_df_ind = train_df_ind.iloc[:-split_purge, :]
                elif j - 1 in test_idx:
                    split_embargo = 2 * int(len(train_df_ind) * self.purge_pct)
                    train_df_ind = train_df_ind.iloc[split_embargo:, :]

                # Add the purged set to the list
                train_sets.append(train_df_ind)

            self.df_lists.append([train_sets, test_sets])

        # STEP 2: Concatenate consecutive sets
        # Create empty lists that will contain
        train_output_list = []
        test_output_list = []

        # We analyze for each couple (train, test) if we can concat consecutive sets
        for j in range(len(self.df_lists)):
            new_list_train = [self.df_lists[j][0][0]]
            new_list_test = [self.df_lists[j][1][0]]

            # We check separately if we need to concat some sets for the train sets and the test sets because it will
            # not increase so much the computation (1s) but it will decrease a lot the complexity of the code

            # Extract each list of train sets to check if a concatenation is needed
            for i in range(1, len(self.df_lists[j][0])):
                # Extract index for the last value of the previous set and the first value of the current set
                idx_end = self.df_lists[j][0][i - 1].index[-1]
                idx_start = self.df_lists[j][0][i].index[0]

                # Use the data from the beginning (the data variable) to check if idx_start is really the next index
                # If yes, it means that the two sets are consecutive
                sub_data = self.data.loc[idx_end:]
                normal_start_idx = sub_data.index[1]

                # Verify if we have index of the next set is the same of the index that follow the last index of the
                # previous set (using the data variable which is the input data)
                if idx_start == normal_start_idx:
                    # Take the last set
                    current_df = new_list_train[-1]

                    # Add this set to the previous one because they are consecutive
                    current_df_updated = pd.concat((current_df, self.df_lists[j][0][i]), axis=0)

                    # Replace the set by the updated set
                    new_list_train[-1] = current_df_updated
                else:
                    # When the last index of the previous set is not the first index of this set, we initiate another
                    # set
                    new_list_train.append(self.df_lists[j][0][i])

            # Add the list of train sets into a list (because the list of train sets is only one path)
            train_output_list.append(new_list_train)

            # Extract each list of train sets to check if a concatenation is needed
            for i in range(1, len(self.df_lists[j][1])):
                # Extract index for the last value of the previous set and the first value of the current set
                idx_end = self.df_lists[j][1][i - 1].index[-1]
                idx_start = self.df_lists[j][1][i].index[0]

                # Use the data from the beginning (the data variable) to check if idx_start is really the next index
                # If yes, it means that the two sets are consecutive
                sub_data = self.data.loc[idx_end:]
                normal_start_idx = sub_data.index[1]

                # Verify if we have index of the next set is the same of the index that follow the last index of the
                # previous set (using the data variable which is the input data)
                if idx_start == normal_start_idx:
                    # Take the last set
                    current_df = new_list_test[-1]

                    # Add this set to the previous one because they are consecutive
                    current_df_updated = pd.concat((current_df, self.df_lists[j][1][i]), axis=0)

                    # Replace the set by the updated set
                    new_list_test[-1] = current_df_updated
                else:
                    # When the last index of the previous set is not the first index of this set, we initiate another
                    # set
                    new_list_test.append(self.df_lists[j][1][i])

            # Add the list of test sets into a list (because the list of test sets is only one path)
            test_output_list.append(new_list_test)

        # We replace the list of train sets and test sets for each couple by the same set but with a concat when
        # it is possible
        for j in range(len(self.df_lists)):
            self.df_lists[j][0] = train_output_list[j]
            self.df_lists[j][1] = test_output_list[j]

    def get_returns(self, train=True):
        # We extract the data with the features to concatenate them later
        # Necessary to avoid creating the features and the ML weights on data with huge gap sometimes
        train_sample_list = []

        for train_sample in self.train_df_list:
            # Initialize the instance
            Strategy = self.TradingStrategy(train_sample, self.params_item)

            # Extract the attributes (the variables containing self. before)
            attributes = [attr for attr in dir(Strategy)]

            # Sometimes we need self.data_train when it is a ML model and when it is not we need self.data
            if "data_train" in attributes:
                train_sample_list.append(Strategy.data_train)
            else:
                train_sample_list.append(Strategy.data)

        # Concatenate each train sets to train the weights (ONLY) on the more data possible
        self.train_sample = pd.concat(train_sample_list, axis=0)

        # Prepare some weight (if it is necessary)
        Strategy = self.TradingStrategy(self.train_sample, self.params_item)

        # Extract weights that will allow us to run our algo (especially for strategies that need a training like ML)
        self.output_params = Strategy.output_dictionary

        # Create an empty list to store the returns (and concat them)
        list_return = []

        # Chose the right set for depending on the mode Train or Test
        if train:
            df_list = self.train_df_list
        else:
            df_list = self.test_df_list

        for tsample in df_list:
            # Compute the returns
            self.BT = Backtest(data=tsample, TradingStrategy=self.TradingStrategy, parameters=self.output_params)
            self.BT.run()

            # Add the returns into the list
            list_return.append(self.BT.data)

        # Concat the list to have the whole backtest (we need to do that to avoid aberrant values in the gap)
        sets = pd.concat(list_return, axis=0)

        # We initialize the Backtest class again (Without running the backtest) just to compute the criterion
        self.BT = Backtest(data=sets, TradingStrategy=self.TradingStrategy, parameters=self.params_item)

        # Calculation and storage of the criterion (Return over period over the maximum drawdown)
        ret, dd = self.BT.get_ret_dd()

        # We use the Calmar ratio as criterion
        self.criterion = ret / np.abs(dd)

    def get_best_params_set(self):
        storage_values_params = []

        for self.params_item in self.dictionaries:
            # Extract the variables parameters from the dictionary
            current_params = [self.params_item[key] for key in list(self.parameters_range.keys())]

            # Compute the criterion and add it to the list of params (criterion train)
            self.get_returns(train=True)
            current_params.append(self.criterion)

            # Compute the criterion and add it to the list of params (criterion test)
            self.get_returns(train=False)
            current_params.append(self.criterion)

            storage_values_params.append(current_params)

        # Extract the dataframe line with the best parameters
        df_find_params = pd.DataFrame(storage_values_params, columns=self.columns)
        self.dfs_list_pbo.append(df_find_params)

        # Extract the dataframe line with the best parameters
        self.best_params_sample_df = df_find_params.sort_values(by="criterion_train", ascending=False).iloc[0:1, :]

        # !! We put the last index value as index
        # because WITHOUT that when you replace the criterion value later you will replace all value with the same index
        self.best_params_sample_df.index = [self.counter]

        # We add the best params to the dataframe which contains all the best params for each period
        self.df_results = pd.concat((self.df_results, self.best_params_sample_df), axis=0)

        # Create a dictionary with the best params on the train set in order to test them on the test set later
        self.best_params_sample = dict(df_find_params.sort_values(by="criterion_train", ascending=False).iloc[0, :-2])
        self.best_params_sample.update(self.fixed_parameters)

    def run_optimization(self):
        # Create the sub-samples
        self.get_sub_samples()
        self.get_combinations()

        # Run the optimization
        for couple_list in tqdm(self.df_lists):
            # Extract the train and test sets in this couple
            self.train_df_list, self.test_df_list = couple_list[0], couple_list[1]

            self.get_best_params_set()
            self.counter += 1

    def get_combination_graph(self, ax):
        # Iterate on each sets couples to display the
        for i in range(len(self.df_lists)):
            # Extract a couple
            list_couple = self.df_lists[i]

            # Extract the train and test sets in this couple
            train_df_list, test_df_list = list_couple[0], list_couple[1]

            # Concatenate each sets into a True train & test periods
            df_test = pd.concat(test_df_list, axis=0)
            df_train = pd.concat(train_df_list, axis=0)

            # Plot the periods for each couple (we add i to the 1 series to can visualize each sets easily)
            ax.plot(df_train.index, np.ones(len(df_train)) + i, "o", color='#6F9FCA', linewidth=1)
            ax.plot(df_test.index, np.ones(len(df_test)) + i, "o", color='#CA7F6F', linewidth=1)

        # Some layout
        ax.set_title(f"Nb tests: {len(self.df_lists)}")
        plt.legend(["TRAIN", "TEST"], loc="upper left")
        plt.show()

    def get_pbo(self):
        for ind_df in self.dfs_list_pbo:
            # Order the dataframe using the criterion test column
            dfp_ordered = ind_df.sort_values(by="criterion_test", ascending=False)

            # Re-index the dataframe
            dfp_ordered.index = [len(dfp_ordered) + 1 - i for i in range(1, len(dfp_ordered) + 1)]

            # Order the dataframe a second time through the criterion train column
            dfp_rank = dfp_ordered.sort_values(by="criterion_train", ascending=False)

            # Extract the rank in the test of the best combination in the train set
            rank = dfp_rank.index[0]

            # Create the relative rank of the OOS performance
            wcb = rank / (len(dfp_ordered) + 1)

            # Create the logit
            lambda_c = np.log(wcb / (1 - wcb))

            # Add it to the logits list
            self.lambdas.append(lambda_c)
            print(dfp_rank.index[0], lambda_c)

        # Create a Series with the lambda list
        self.lmb_series = pd.Series(self.lambdas)

        # Compute the probability of overfitting
        self.pbo = 100 * len(self.lmb_series[self.lmb_series < 0]) / len(self.lmb_series)

    def get_pbo_graph(self, ax):
        ax.hist(self.lmb_series, color="#6F9FCA", bins=10, edgecolor='black', density=True)
        sns.kdeplot(self.lmb_series, color="#CA6F6F", ax=ax)
        ax.text(0.95, 0.90, f'Probability of Overfitting: {self.pbo:.2f} %', horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.9'))
        ax.set_title("Hist of Rank Logits")
        ax.set_xlabel("Logits")
        ax.set_ylabel("Frequency")

    def get_degration_graph(self, ax):
        x = self.df_results["criterion_train"]
        y = self.df_results["criterion_test"]

        # Regression line
        coeffs = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = coeffs[1] + coeffs[0] * line_x

        # P(SR_OOS < 0)
        ct_oos = self.df_results["criterion_test"]
        p_oos_pos = 100*len(ct_oos[ct_oos>0]) / len(ct_oos)

        ax.scatter(x, y)
        ax.plot(line_x, line_y, color='#CA6F6F')
        ax.text(0.95, 0.90, f'P(SR[00S] > 0): {p_oos_pos:.2f} %', horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.9'))
        ax.set_title(f"Criterion TEST = {coeffs[1]:.2f} + TRAIN * {coeffs[0]:.2f} + epsilon")
        ax.set_xlabel("Criterion Train")
        ax.set_ylabel("Criterion Test")

    def display_all_graph(self):
        fig, axes = plt.subplot_mosaic('AB;CC', figsize=(15, 8))

        self.get_pbo_graph(axes["A"])
        self.get_degration_graph(axes["B"])
        self.get_combination_graph(axes["C"])

        fig.subplots_adjust(hspace=0.9)

        plt.show()
