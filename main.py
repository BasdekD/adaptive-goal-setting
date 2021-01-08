import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics, linear_model, tree, ensemble
from matplotlib import pyplot as plt
import numpy as np


def concat_individual_datasets():
    """
    Create an aggregated dataset from a number of individual smaller datasets
    """

    n_per_in = 7
    n_per_out = 1
    n_features = 1

    # The variable in which the aggregated dataset will be stored
    dataset = pd.DataFrame()
    # The directory where all the individual data are
    individual_data = os.fsencode('individual data')
    # Iterating through the individual data directory
    for csv in os.listdir(individual_data):
        # The individual's data csv
        df = pd.read_csv(os.path.join(individual_data, csv).decode('utf-8'))
        # Setting the date column as the index of the dataframe
        df = df.set_index(df.columns[0])
        # Normalizing/Scaling the Data
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        # Transforming the time series data in a dataframe appropriate for supervised learning
        df = transform_timeseries_data(df, n_per_in, n_per_out, n_features)
        # In each iteration the individual dataset is added to the aggregated dataset
        dataset = pd.concat([dataset, df])
    pd.set_option('display.max_columns', None)
    print(dataset.head())
    print(dataset.columns)
    print(len(dataset))
    pd.options.display.width = 0
    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # model = linear_model.LinearRegression()
    # model = tree.DecisionTreeRegressor(criterion="mae", max_depth=12)
    model = ensemble.RandomForestRegressor(criterion="mae")
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, n_features)).tolist()
    y_predicted = scaler.inverse_transform(np.array(y_predicted).reshape(-1, n_features)).tolist()
    print("The MAE is {}".format(metrics.mean_absolute_error(y_test, y_predicted)))

    last_week = np.array(X.iloc[-20:-1, :])
    yhat = model.predict(last_week)
    # Transforming values back to their normal prices
    yhat = scaler.inverse_transform(np.array(yhat).reshape(-1, n_features)).tolist()
    yhat = pd.DataFrame(yhat)
    actual = np.array(y.iloc[-20:-1])
    actual = scaler.inverse_transform(actual.reshape((-1, 1)))
    actual = pd.DataFrame(actual)
    print(metrics.mean_absolute_error(yhat, actual))

    # Printing and plotting those predictions
    print("Predicted Step Count:\n", yhat.loc[::, ::])

    plt.plot(yhat.loc[::, ::], label='Predicted')

    # Printing and plotting the actual values
    print("\nActual Step Count:\n", actual.loc[::, ::])
    plt.plot(actual.loc[::, ::], label='Actual')

    plt.title(f"Predicted vs Actual Daily Step Counts")
    plt.ylabel("Steps")
    plt.legend()
    plt.show()


def transform_timeseries_data(data, n_in=1, n_out=1, n_features=1, drop_NaN=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        n_features: Number of features to base the transaction on.
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

    # X = agg.iloc[::, :(n_in * n_features)]
    # y = agg.iloc[::, (n_in * n_features):]
    return agg


if __name__ == '__main__':
    concat_individual_datasets()
