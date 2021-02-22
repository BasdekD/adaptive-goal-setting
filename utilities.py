from statistics import mean

import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


# get the dataset
def get_dataset(n_per_in, n_per_out):
    print('Reading dataset....')
    dataset = pd.read_excel(
        '01. aggregated_and_timeseries_transformed_datasets\\'
        'aggregated_dataset_in_5_out_1_activity_date_ids_no_values_bellow_500.xlsx',
        header=0,
        engine='openpyxl',
        index_col='date'
    )
    y = dataset['var1(t)']
    X = dataset.drop(columns='var1(t)')

    print('Scaling dataset....')
    scaler = RobustScaler()
    x_rescaled = scaler.fit_transform(X)
    return x_rescaled, y


def get_models(names):
    all_models = dict()
    models_to_return = dict()
    # ri
    estimator = Ridge()
    # grid_params = [{'fit_intercept': [True]}]
    grid_params = [{
        'alpha': [0.1, 1],
        'solver': ['auto', 'lsqr', 'saga']
    }]
    all_models['ri'] = {'estimator': estimator, 'grid_params': grid_params}
    # dt
    estimator = DecisionTreeRegressor()
    # grid_params = [{
    #     'criterion': ['mae'],
    #     'max_depth': [3, 5, 7],
    #     'max_features': ['auto']
    # }]
    grid_params = [{
        'criterion': ['mae'],
        'max_depth': [7, 12, 20],
        'max_features': ['auto', 'sqrt', 'log2']
    }]
    all_models['dt'] = {'estimator': estimator, 'grid_params': grid_params}
    # rf
    estimator = RandomForestRegressor()
    grid_params = [{
        'n_estimators': [50, 100, 150],
        'criterion': ['mae'],
        'max_depth': [7, 12, 20],
        'max_features': ['sqrt', 'log2']
    }]
    # grid_params = [{
    #     'n_estimators': [150, 200, 250],
    #     'criterion': ['mae'],
    #     'max_depth': [12],
    #     'max_features': ['sqrt']
    # }]
    # grid_params = [{
    #     'n_estimators': [250, 300, 400],
    #     'criterion': ['mae'],
    #     'max_depth': [40],
    #     'max_features': ['sqrt']
    # }]
    all_models['rf'] = {'estimator': estimator, 'grid_params': grid_params}
    # gbr
    estimator = GradientBoostingRegressor()
    grid_params = [{
        'loss': ['ls', 'huber'],
        'criterion': ['mae'],
        'max_depth': [7, 12, 20],
        'max_features': ['sqrt', 'log2']
    }]
    # grid_params = [{
    #     'loss': ['ls'],
    #     'criterion': ['mae'],
    #     'max_depth': [20, 40, 60],
    #     'max_features': ['sqrt']
    # }]
    all_models['gbr'] = {'estimator': estimator, 'grid_params': grid_params}
    for name, model in all_models.items():
        if name in names:
            models_to_return[name] = model
        else:
            pass
    return models_to_return


def get_feature_selector(model, n_per_in):
    print("Loading optimal number of features for window size %d" % n_per_in)
    rfecv = pickle.load(open('rfecv_%s_%d.sav' % (model, n_per_in), 'rb'))
    return rfecv


def step_error_as_percentage(y_true, y_predicted):
    y_true = y_true.to_list()
    error_list = list()
    for i in range(len(y_true)):
        if y_true[i] == 0:
            pass
        else:
            error = round(abs((y_true[i] - y_predicted[i])/y_true[i] * 100), 2)
            error_list.append(error)
    return mean(error_list)
