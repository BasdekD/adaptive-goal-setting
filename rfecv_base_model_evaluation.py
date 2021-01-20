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


n_per_in = 5
n_per_out = 1
initial_n_features = 38


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
    rfecv = RFECV(estimator=LinearRegression(), step=n_per_in, cv=KFold(shuffle=False),
                  scoring='neg_mean_absolute_error',
                  min_features_to_select=n_per_in)
    rfecv.fit(X, y)
    filename = 'rfecv_lr_'+str(n_per_in)+'.sav'
    pickle.dump(rfecv, open(filename, 'wb'))
    selectors['lr'] = rfecv
    # dt
    print('Training Decision Tree wrapper....')
    rfecv = RFECV(estimator=DecisionTreeRegressor(), step=n_per_in, cv=KFold(shuffle=False),
                  scoring='neg_mean_absolute_error',
                  min_features_to_select=n_per_in)
    rfecv.fit(X, y)
    filename = 'rfecv_dt_'+str(n_per_in)+'.sav'
    pickle.dump(rfecv, open(filename, 'wb'))
    selectors['dt'] = rfecv
    # rf
    print('Training Random Forest wrapper....')
    rfecv = RFECV(estimator=RandomForestRegressor(), step=n_per_in, cv=KFold(shuffle=False),
                  scoring='neg_mean_absolute_error',
                  min_features_to_select=n_per_in)
    rfecv.fit(X, y)
    filename = 'rfecv_rf_'+str(n_per_in)+'.sav'
    pickle.dump(rfecv, open(filename, 'wb'))
    selectors['rf'] = rfecv
    # gbr
    print('Training GBRegressor wrapper....')
    rfecv = RFECV(estimator=GradientBoostingRegressor(), step=n_per_in, cv=KFold(shuffle=False),
                  scoring='neg_mean_absolute_error',
                  min_features_to_select=n_per_in)
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
    results.append(-score)
    names.append(name)
    print('>%s %.3f %d' % (name, score, n_features))
# plot model performance for comparison
pyplot.figure()
pyplot.title('BEST MAE PER BASE MODEL FOR RFE')
pyplot.xlabel("Base model used")
pyplot.ylabel("Best MAE")
pyplot.bar(names, results)
pyplot.show()

i = (initial_n_features * n_per_in) + (initial_n_features - 1)
n_of_features = []
while i > n_per_in:
    n_of_features.append(i)
    i -= n_per_in
n_of_features.append(n_per_in)
n_of_features.reverse()
for name, selector in selectors.items():
    pyplot.figure()
    pyplot.title('%s | Best CV MAE: %.3f | Features: %d' % (name.upper(), selector.grid_scores_.max(), selector.n_features_))
    pyplot.xlabel("Number of features selected")
    pyplot.ylabel("Cross validation score")
    pyplot.plot(n_of_features, selector.grid_scores_)
    pyplot.show()
