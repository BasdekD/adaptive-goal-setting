import pandas as pd
import os
from datetime import datetime
import pickle


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


def create_data_with_consecutive_days(df, file_name, counter_of_breaks_in_dates):
    print("Checking dates")
    directory = 'individual_all_data_cons_days'
    cons_dates = pd.DataFrame(columns=df.columns)
    found_non_cons_dates = False
    for index, row in df.iterrows():
        date_1 = datetime.strptime(index, "%Y-%m-%d")
        date_2 = datetime.strptime(df.loc[df.index[df.index.to_list().index(index) - 1]].name, "%Y-%m-%d")
        if cons_dates.empty:
            cons_dates = cons_dates.append(row)
        else:
            if date_1.toordinal() - date_2.toordinal() == 1 or \
                    (date_2.month == 2 and date_2.day == 28 and date_1.toordinal() - date_2.toordinal() == 2):
                cons_dates = cons_dates.append(row)
            else:
                found_non_cons_dates = True
                print("Writing consecutive dates of file %s" % file_name)
                cons_dates.to_excel(directory+'\\'+file_name+'_%d.xlsx' % counter_of_breaks_in_dates)
                # pickle.dump(cons_dates, open(file_name+'_%d.sav' % counter_of_breaks_in_dates, 'wb'))
                counter_of_breaks_in_dates += 1
                print("Found non-consecutive dates %s follows %s" % (date_1, date_2))
                print("This is break no: %d in file %s" % (counter_of_breaks_in_dates, file_name))
                create_data_with_consecutive_days(df.iloc[df.index.to_list().index(index):, ::],
                                                  file_name, counter_of_breaks_in_dates)
                break
    if not found_non_cons_dates:
        print("Writing final dates of file %s" % file_name)
        cons_dates.to_excel(directory+'\\'+file_name + '_%d.xlsx' % counter_of_breaks_in_dates)
        # pickle.dump(cons_dates, open(file_name+'_%d.sav' % counter_of_breaks_in_dates, 'wb'))


n_per_in = 5
n_per_out = 1
n_features = 38


def create_aggregated_dataset(in_periods, out_periods):
    # The variable in which the aggregated dataset will be stored
    dataset = pd.DataFrame()
    consecutive_data_folder = os.fsencode('individual_all_data_cons_days')
    for csv in os.listdir(consecutive_data_folder):
        print("Processing file %s" % csv.decode('utf-8'))
        # The individual's all_data_aggregated csv
        df = pd.read_excel(os.path.join(consecutive_data_folder, csv).decode('utf-8'),
                           engine='openpyxl', header=0)
        # Setting the date column as the index of the dataframe
        df = df.set_index(df.columns[0])
        df.index.name = 'date'
        # Transforming the time series all_data_aggregated in a dataframe appropriate for supervised learning
        df = transform_timeseries_data(df, n_per_in, n_per_out)
        # In each iteration the individual dataset is added to the aggregated dataset
        dataset = pd.concat([dataset, df])
    path = os.fsencode('all_data_aggregated')
    print('Writing file aggregated_dataset_in_'+str(n_per_in)+'_out_'+str(n_per_out)+'_new.xlsx')
    dataset.to_excel(os.path.join(path.decode("utf-8"),
                                  'aggregated_dataset_in_'+str(n_per_in)+'_out_'+str(n_per_out)+'_new.xlsx'))


dataset_folder = os.fsencode('individual_all_data')
# Iterating through the individual all_data_aggregated directory
for csv in os.listdir(dataset_folder):
    df = pd.read_excel(os.path.join(dataset_folder, csv).decode('utf-8'), engine='openpyxl', header=0, index_col='date')
    print("Removing non-wear days from file %s" % csv.decode('utf-8'))
    filtered_data = df.loc[df['Steps'] >= 500]
    print("Removed %d days from dataset" % (len(df) - len(filtered_data)))
    dif = df.merge(filtered_data, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
    print(dif)
    print("Checking and handling non-consecutive days in dataset")
    create_data_with_consecutive_days(filtered_data, csv.decode('utf-8')[:-4], 0)
create_aggregated_dataset(n_per_in, n_per_out)
