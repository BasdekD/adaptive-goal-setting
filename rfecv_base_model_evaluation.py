# explore the algorithm wrapped by RFECV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot
import pandas as pd
import pickle


n_per_in = 7
n_per_out = 1


# get the dataset
def get_dataset():
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
    X_rescaled = scaler.fit_transform(X)
    return X_rescaled, y


# get a list of models to evaluate
def get_selectors(X, y):
    selectors = dict()
    # lr
    print('Training Linear Regression wrapper....')
    rfecv = RFECV(estimator=LinearRegression(), step=7, cv=KFold(shuffle=False),
                  scoring='neg_mean_absolute_error',
                  min_features_to_select=7)
    rfecv.fit(X, y)
    filename = 'rfecv_lr_'+str(n_per_in)+'.sav'
    pickle.dump(rfecv, open(filename, 'wb'))
    selectors['lr'] = rfecv
    # dt
    print('Training Decision Tree wrapper....')
    rfecv = RFECV(estimator=DecisionTreeRegressor(), step=7, cv=KFold(shuffle=False),
                  scoring='neg_mean_absolute_error',
                  min_features_to_select=7)
    rfecv.fit(X, y)
    filename = 'rfecv_dt_'+str(n_per_in)+'.sav'
    pickle.dump(rfecv, open(filename, 'wb'))
    selectors['dt'] = rfecv
    # rf
    print('Training Random Forest wrapper....')
    rfecv = RFECV(estimator=RandomForestRegressor(), step=7, cv=KFold(shuffle=False),
                  scoring='neg_mean_absolute_error',
                  min_features_to_select=7)
    rfecv.fit(X, y)
    filename = 'rfecv_lr_'+str(n_per_in)+'.sav'
    pickle.dump(rfecv, open(filename, 'wb'))
    selectors['rf'] = rfecv
    # gbr
    print('Training GBRegressor wrapper....')
    rfecv = RFECV(estimator=GradientBoostingRegressor(), step=7, cv=KFold(shuffle=False),
                  scoring='neg_mean_absolute_error',
                  min_features_to_select=7)
    rfecv.fit(X, y)
    filename = 'rfecv_gbr_'+str(n_per_in)+'.sav'
    pickle.dump(rfecv, open(filename, 'wb'))
    selectors['gbr'] = rfecv
    return selectors


# define dataset
X, y = get_dataset()
# get the models to evaluate
selectors = get_selectors(X, y)
# evaluate the models and store results
results, names = list(), list()
for name, selector in selectors.items():
    score = selector.grid_scores_.max()
    n_features = selector.n_features_
    results.append(score)
    names.append(name)
    print('>%s %.3f %d' % (name, score, n_features))
# plot model performance for comparison
pyplot.bar(names, results)
pyplot.show()
