import pandas as pd
import os
from datetime import datetime


def transform_timeseries_data(data, n_in=1, n_out=1, drop_NaN=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        drop_NaN: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if drop_NaN:
        agg.dropna(inplace=True)

    # y = agg['var1(t)']
    # X = agg.drop(columns='var1(t)')
    return agg


def check_if_dates_are_consecutive(df):
    print("Checking dates")
    cons_dates = pd.DataFrame()
    dataset = pd.DataFrame()
    for index, row in df.iterrows():
        if cons_dates.empty:
            cons_dates = cons_dates.append(row, ignore_index=True)
        else:
            # if datetime.strptime(row['date'], "%Y-%m-%d").toordinal() - datetime.strptime(df.loc[index - 1]['date'],
            #                                                                              "%Y-%m-%d").toordinal() == 1:
            if datetime.strptime(index, "%Y-%m-%d").toordinal() - datetime.strptime(df.loc[df.index[df.index.to_list().index(index) - 1]].name,
                                                                                          "%Y-%m-%d").toordinal() == 1:
                cons_dates = cons_dates.append(row, ignore_index=True)
            else:
                print("found non-consecutive dates")
                transformed_data = transform_timeseries_data(cons_dates, n_per_in, n_per_out)
                dataset = pd.concat([dataset, transformed_data])
                dataset = pd.concat([dataset, check_if_dates_are_consecutive(df.iloc[df.index.to_list().index(index)+1:, ::])])
    transformed_data = transform_timeseries_data(cons_dates, n_per_in, n_per_out)
    return pd.concat([dataset, transformed_data])


n_per_in = 5
n_per_out = 1
n_features = 38

# The variable in which the aggregated dataset will be stored
dataset = pd.DataFrame()
# The directory where all the individual all_data_aggregated are
dataset_folder = os.fsencode('individual_all_data')
# Iterating through the individual all_data_aggregated directory
for csv in os.listdir(dataset_folder):
    # The individual's all_data_aggregated csv
    df = pd.read_excel(os.path.join(dataset_folder, csv).decode('utf-8'), engine='openpyxl')
    # Setting the date column as the index of the dataframe
    df = df.drop_duplicates()
    df = df.set_index(df.columns[0])
    df = check_if_dates_are_consecutive(df)
    # Normalizing/Scaling the Data
    # scaler = MinMaxScaler()
    # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    # Transforming the time series all_data_aggregated in a dataframe appropriate for supervised learning
    # df = transform_timeseries_data(df, n_per_in, n_per_out)
    # In each iteration the individual dataset is added to the aggregated dataset
    dataset = pd.concat([dataset, df])
pd.set_option('display.max_columns', None)
path = os.fsencode('all_data_aggregated')
dataset.to_excel(os.path.join(path.decode("utf-8"),
                              'aggregated_dataset_in_'+str(n_per_in)+'_out_'+str(n_per_out)+'_new.xlsx'))


# n_per_in = 14
# n_per_out = 1
# n_features = 38
#
# # The variable in which the aggregated dataset will be stored
# dataset = pd.DataFrame()
# # The directory where all the individual all_data_aggregated are
# dataset_folder = os.fsencode('individual_preprocessed_activity_data')
# # Iterating through the individual all_data_aggregated directory
# for csv in os.listdir(dataset_folder):
#     # The individual's all_data_aggregated csv
#     df = pd.read_csv(os.path.join(dataset_folder, csv).decode('utf-8'))
#     # Setting the date column as the index of the dataframe
#     df = df.set_index(df.columns[0])
#     df = transform_timeseries_data(df, n_per_in, n_per_out)
#     # In each iteration the individual dataset is added to the aggregated dataset
#     dataset = pd.concat([dataset, df])
# path = os.fsencode('all_data_aggregated')
# dataset.to_excel(os.path.join(path.decode("utf-8"),
#                               'aggregated_activity_only_dataset_in_'+str(n_per_in)+'_out_'+str(n_per_out)+'.xlsx'))
