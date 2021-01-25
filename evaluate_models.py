import utilities
import pickle
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error


def get_models():
    models = dict()
    # lr
    estimator = LinearRegression()
    grid_params = [{'fit_intercept': [True]}]
    models['lr'] = {'estimator': estimator, 'grid_params': grid_params}
    # dt
    estimator = DecisionTreeRegressor()
    grid_params = [{
        'criterion': ['mae'],
        'max_depth': [7, 12, 20],
        'max_features': ['auto', 'sqrt', 'log2']
    }]
    models['dt'] = {'estimator': estimator, 'grid_params': grid_params}
    # rf
    estimator = RandomForestRegressor()
    grid_params = [{
        'n_estimators': [50, 100, 150],
        'criterion': ['mae'],
        'max_depth': [7, 12, 20],
        'max_features': ['sqrt', 'log2']
    }]
    models['rf'] = {'estimator': estimator, 'grid_params': grid_params}
    # gbr
    estimator = GradientBoostingRegressor()
    grid_params = [{
        'loss': ['ls', 'huber'],
        'criterion': ['mae'],
        'max_depth': [7, 12, 20],
        'max_features': ['sqrt', 'log2']
    }]
    models['gbr'] = {'estimator': estimator, 'grid_params': grid_params}
    return models


def get_n_features(n_per_in):
    print("Loading optimal number of features for window size %d" % n_per_in)
    rfecv = pickle.load(open('rfecv_gbr_'+str(n_per_in)+'.sav', 'rb'))
    n_features = rfecv.n_features_
    return n_features


n_per_in = 5
n_per_out = 1
X, y = utilities.get_dataset(n_per_in, n_per_out)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# create pipeline
rfe = RFE(estimator=GradientBoostingRegressor(criterion="mae"), n_features_to_select=get_n_features(n_per_in))
models = get_models()
results, names = list(), list()
for name, model in models.items():
    cv = KFold(n_splits=5, shuffle=False)
    predictor = GridSearchCV(
        estimator=model['estimator'],
        param_grid=model['grid_params'],
        scoring='neg_mean_absolute_error',
        cv=cv,
        refit=True
    )
    pipeline = Pipeline(steps=[('feature_selector', rfe), ('model', predictor)])
    print("Creating pipeline with %s model..." % name)
    pipeline.fit(x_train, y_train)
    filename = 'model_%s_%d.sav' % (name, n_per_in)
    print("Writing %s pipeline to file..." % name)
    pickle.dump(pipeline, open(filename, 'wb'))
    # evaluate model
    print("Getting predictions for validation set")
    y_predicted = pipeline.predict(x_test)
    print('MAE in validation set for %s: %.3f' % (name, mean_absolute_error(y_test, y_predicted)))
    score = pipeline['model'].best_score_
    results.append(score)
    names.append(name)
    # report performance
    print('>%s %.3f' % (name, score))
    pyplot.plot(y_predicted, label='Predicted')
    # Printing and plotting the actual values
    pyplot.plot(y_test, label='Actual')
    pyplot.title(f"Predicted vs Actual Daily Step Counts")
    pyplot.ylabel("Steps")
    pyplot.legend()
    pyplot.show()

pyplot.figure()
pyplot.title('CV MAE PER MODEL')
pyplot.xlabel("Algorithms")
pyplot.ylabel("MAE")
pyplot.bar(names, results)
pyplot.show()

# plot model performance for comparison
pyplot.figure()
pyplot.title('CV MAE PER MODEL')
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
