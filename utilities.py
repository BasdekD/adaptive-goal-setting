import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# get the dataset
def get_dataset(n_per_in, n_per_out):
    print('Reading dataset....')
    dataset = pd.read_excel(
        'all_data_aggregated\\aggregated_dataset_in_' + str(n_per_in) + '_out_' + str(n_per_out) + '.xlsx',
        header=0,
        engine='openpyxl',
        index_col='date'
    )
    y = dataset['var1(t)']
    X = dataset.drop(columns='var1(t)')

    print('Scaling dataset....')
    scaler = RobustScaler()
    print("Normalizing data.....")
    x_rescaled = scaler.fit_transform(X)
    return x_rescaled, y


def get_models(names):
    all_models = dict()
    models_to_return = dict()
    # lr
    estimator = LinearRegression()
    grid_params = [{'fit_intercept': [True]}]
    all_models['lr'] = {'estimator': estimator, 'grid_params': grid_params}
    # dt
    estimator = DecisionTreeRegressor()
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
    all_models['rf'] = {'estimator': estimator, 'grid_params': grid_params}
    # gbr
    estimator = GradientBoostingRegressor()
    grid_params = [{
        'loss': ['ls', 'huber'],
        'criterion': ['mae'],
        'max_depth': [7, 12, 20],
        'max_features': ['sqrt', 'log2']
    }]
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

