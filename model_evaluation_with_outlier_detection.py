import pickle
import utilities
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot
import numpy as np

n_per_in = 5
n_per_out = 1
X, y = utilities.get_dataset(n_per_in, n_per_out)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
print('Shape before outlier removal')
print(x_train.shape, y_train.shape)

# lr = pickle.load(open('models/no_rfe_model_lr_%d.sav' % n_per_in, 'rb'))
# dt = pickle.load(open('models/no_rfe_model_dt_%d.sav' % n_per_in, 'rb'))
# rf = pickle.load(open('models/no_rfe_model_rf_%d.sav' % n_per_in, 'rb'))
gbr = pickle.load(open('models/no_rfe_model_gbr_%d.sav' % n_per_in, 'rb'))

# models_with_outliers = {'lr': lr, 'dt': dt, 'rf': rf, 'gbr': gbr}
models_with_outliers = {'gbr': gbr}
# iso = IsolationForest(contamination=0.1)
# yhat = iso.fit_predict(x_train)
# mask = yhat != -1
# x_train, y_train = x_train[mask, :], y_train[mask]
# print('Shape after outlier removal')
# print(x_train.shape, y_train.shape)

cv = KFold(n_splits=5, shuffle=False)
results_with_outliers, results_without_outliers, names = list(), list(), list()

models_without_outliers = utilities.get_models(['gbr'])
for name, model in models_without_outliers.items():
    best_params = models_with_outliers[name].best_params_
    for param_name, param_value in best_params.items():
        best_params[param_name] = [param_value]
    print("Training %s model..." % name)
    predictor = GridSearchCV(
        estimator=model['estimator'],
        param_grid=best_params,
        scoring='neg_mean_absolute_error',
        cv=cv,
        refit=True
    )
    predictor.fit(x_train, y_train)
    filename = 'no_outliers_no_rfe_model_%s_%d.sav' % (name, n_per_in)
    print("Writing %s model to file..." % name)
    # pickle.dump(predictor, open(filename, 'wb'))
    # evaluate model
    print("Getting predictions for validation set")
    y_predicted = predictor.predict(x_test)
    print('MAE in validation set for %s: %.3f' % (name, mean_absolute_error(y_test, y_predicted)))
    score = predictor.best_score_
    results_without_outliers.append(score)
    results_with_outliers.append(models_with_outliers[name].best_score_)
    names.append(name)
    # report performance
    print('>%s %.3f' % (name, score))
    pyplot.plot(y_predicted[:20], label='Predicted')
    # Printing and plotting the actual values
    pyplot.plot(y_test[:20], label='Actual')
    pyplot.title(f"Predicted vs Actual Daily Step Counts")
    pyplot.ylabel("Steps")
    pyplot.legend()
    pyplot.show()

pyplot.figure()
x = np.arange(len(results_without_outliers))
pyplot.bar(x, results_with_outliers, color='b', width=0.25)
pyplot.bar(x, results_without_outliers, color='r', width=0.25)
pyplot.legend(['Outliers in Dataset', 'Outliers Removed'])
pyplot.xticks([i + 0.25 for i in range(len(names))], names)
pyplot.title('MEAN CV MAE WITH VS WITHOUT OUTLIERS IN DATASET')
pyplot.xlabel('ALGORITHMS')
pyplot.ylabel('MAE IN NUMBER OF STEPS')
pyplot.show()

