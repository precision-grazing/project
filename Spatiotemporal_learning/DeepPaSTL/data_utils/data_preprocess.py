from os import listdir
import sys
import numpy as np
import pickle

import pandas as pd
import gc

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf

DAYS_IN_A_YEAR = 366  # Account for leap year, instead of discarding it.


# Read data from file and process only the relevant information
def load_data(args):
    # Store values on number of days in each dataset fed to the network
    # To be use for creating sliding windows
    n_train_days, n_test_days = [], []
    all_df = []

    if args.xdim * args.ydim != args.num_features:
        assert "Incorrect Feature Dimensions"

    filepaths = [f for f in sorted(listdir(args.data_folder)) if f.endswith(args.dataset)]
    
    for i, f in enumerate(filepaths):
        print(f"Reading {f}")
        df = pd.read_csv(args.data_folder + f)
        df['data_index'] = i
        # Count number of days before merging and store in a list already,
        # it's not efficient, but it works
        train_days, test_days = split_timeseries_data(args, df, return_len=True)
        n_train_days.append(train_days)
        n_test_days.append(test_days)
        all_df.append(df)

    fert_df = pd.concat(all_df, axis=0, ignore_index=True)
    del all_df

    """
    Common Features
    """
    fert_df["date"] = pd.to_datetime(fert_df["date"])
    # Drop the single day of 2010
    fert_df.drop(fert_df[fert_df.date == '2010-01-01'].index, inplace=True)
    height_list = ["h" + str(i + 1) for i in range(args.num_features)]

    # Split the data for training, and testing before computing the correlation
    train_data, test_data = split_timeseries_data(args, fert_df)

    # Now we perform scaling/normalization on the training and omit the validation/target set.
    scale_data_train = train_data[train_data['date'] < args.validation_start_date]
    scale_data_test = test_data

    """
    Scaling Train and Test Data Independently
    """
    print("Scaling Data")
    scale_map_train = {}
    scaled_data_train = pd.DataFrame()
    scaled_data_train = pd.concat([scaled_data_train, scale_data_train], ignore_index=True)

    scale_map_test = {}
    scaled_data_test = pd.DataFrame()
    scaled_data_test = pd.concat([scaled_data_test, scale_data_test], ignore_index=True)

    # To capture the yearly trend of the fertilizer height we also standardize and compute the yearly
    # auto-correlation for each height.
    print("Normalizing Data")

    for h in height_list:
        scaled_data_train[h] = (scaled_data_train[h] - scale_data_train[h].min()) / \
                                (scale_data_train[h].max() - scale_data_train[h].min())
        scale_map_train[h] = {'min_train': scale_data_train[h].min(), 'max_train': scale_data_train[h].max()}
        # print('\n')
        # Standardize for Test
        scaled_data_test[h] = (scaled_data_test[h] - scale_data_test[h].min()) / (scale_data_test[h].max() - scale_data_test[h].min())
        scale_map_test[h] = {'min_test': scale_data_test[h].min(), 'max_test': scale_data_test[h].max()}

    # Drop unnecessary features from Train and Test Data
    scaled_data_train.drop(['date_of_year', 'year'], axis=1, inplace=True)
    scaled_data_test.drop(['date_of_year', 'year'], axis=1, inplace=True)

    print("Saving Scaled Data, Scaling, Number of Days as Pickle")
    print(scaled_data_train.head())
    scaled_data_train.to_pickle(args.data_folder + args.process_folder + args.model + '_train_processed_data' + '.pkl')
    scaled_data_test.to_pickle(args.data_folder + args.process_folder + args.model + '_test_processed_data' + '.pkl')
    pickle.dump(scale_map_train, open(args.data_folder + args.process_folder + args.model + '_scale_map_train.pkl', 'wb'))
    pickle.dump(scale_map_test, open(args.data_folder + args.process_folder + args.model + '_scale_map_test.pkl', 'wb'))

    with open(args.data_folder + args.process_folder + args.model + '_n_train_days' + '.pkl', 'wb') as fp:
        pickle.dump(n_train_days, fp)
    with open(args.data_folder + args.process_folder + args.model + '_n_test_days' + '.pkl', 'wb') as fp:
        pickle.dump(n_test_days, fp)


def split_timeseries_data(args, data, return_len=False):
    # Let's split the data into the following parts
    # Train: 1980-01-01 ~ 2003-12-31
    # Validation: 2004-01-01 ~ 2007-12-31
    # Test: 2008-01-01 ~ 2009-12-31
    if return_len:
        data["date"] = pd.to_datetime(data["date"])
        # Drop the single day of 2010
        data.drop(data[data.date == '2010-01-01'].index, inplace=True)

    print("Timeline of input data: ")
    print(data['date'].min(), " to ", data['date'].max())
    train_data = data[data['date'] < args.testing_start_date]
    test_data = data[(data['date'] >= args.testing_start_date) & (data['date'] <= args.testing_end_date)]

    print("Train Data Timeline: ")
    print(train_data['date'].min(), " to ", train_data['date'].max())
    print("Test Data Timeline: ")
    print(test_data['date'].min(), " to ", test_data['date'].max())

    train_days = len(train_data.index)
    test_days = len(test_data.index)
    print(f'Length of dataset for Train: {train_days}, Test: {test_days}')
    print('\n')

    if return_len:
        return train_days, test_days
    else:
        return train_data, test_data


def get_yearly_autocorr(data):
    ac = acf(data, nlags=366)
    # print(np.shape(ac))
    return (0.5 * ac[365]) + (0.25 * ac[364]) + (0.25 * ac[366])


def last_year_lag(col):
    return (col.shift(364) * 0.25) + (col.shift(365) * 0.5) + (col.shift(366) * 0.25)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

def split_test_data(args, data, return_len=False):
    data["date"] = pd.to_datetime(data["date"])
    # Drop the single day of 2010
    data.drop(data[data.date == '2010-01-01'].index, inplace=True)

    print("Timeline of input data: ")
    test_data = data[(data['date'] >= args.testing_start_date) & (data['date'] <= args.testing_end_date)]
    print("Test Data Timeline: ")
    print(test_data['date'].min(), " to ", test_data['date'].max())
    test_days = len(test_data.index)
    
    if return_len:
        return test_days
    else:
        return test_data

def process_testing_data(args):
    n_test_days = []
    all_df = []

    if args.xdim * args.ydim != args.num_features:
        assert "Incorrect Feature Dimensions"

    filepaths = [f for f in sorted(listdir(args.data_folder)) if f.endswith(args.dataset)]
    
    for i, f in enumerate(filepaths):
        print(f"Reading {f}")
        df = pd.read_csv(args.data_folder + f)
        df['data_index'] = i
        # Count number of days before merging and store in a list already,
        # it's not efficient, but it works
        test_days = split_test_data(args, df, return_len=True)
        n_test_days.append(test_days)
        all_df.append(df)

    fert_df = pd.concat(all_df, axis=0, ignore_index=True)
    del all_df

    fert_df["date"] = pd.to_datetime(fert_df["date"])
    # Drop the single day of 2010
    fert_df.drop(fert_df[fert_df.date == '2010-01-01'].index, inplace=True)
    height_list = ["h" + str(i + 1) for i in range(args.num_features)]
    test_data = split_test_data(args, fert_df)

    scale_data_test = test_data
    scale_map_test = {}
    scaled_data_test = pd.DataFrame()
    scaled_data_test = pd.concat([scaled_data_test, scale_data_test], ignore_index=True)
    
    print("Normalizing Data")
    for h in height_list:
        scaled_data_test[h] = (scaled_data_test[h] - scale_data_test[h].min()) / (scale_data_test[h].max() - scale_data_test[h].min())
        scale_map_test[h] = {'min_test': scale_data_test[h].min(), 'max_test': scale_data_test[h].max()}

    # Drop unnecessary features from Test Data
    scaled_data_test.drop(['date_of_year', 'year'], axis=1, inplace=True)

    print("Saving Scaled Data, Scaling, Number of Days as Pickle")
    print(scaled_data_test.head())
    scaled_data_test.to_pickle(args.data_folder + args.process_folder + args.model + '_test_predictions_processed_data' + '.pkl')
    pickle.dump(scale_map_test, open(args.data_folder + args.process_folder + args.model + '_scale_map_test_predictions.pkl', 'wb'))
    with open(args.data_folder + args.process_folder + args.model + '_n_test_days' + '.pkl', 'wb') as fp:
        pickle.dump(n_test_days, fp)
