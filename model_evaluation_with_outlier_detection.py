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

ri = pickle.load(open('models/no_rfe_model_ri_%d.sav' % n_per_in, 'rb'))
dt = pickle.load(open('models/no_rfe_model_dt_%d.sav' % n_per_in, 'rb'))
rf = pickle.load(open('models/no_rfe_model_rf_%d.sav' % n_per_in, 'rb'))
gbr = pickle.load(open('models/no_rfe_model_gbr_%d.sav' % n_per_in, 'rb'))

models_with_outliers = {'ri': ri, 'dt': dt, 'rf': rf, 'gbr': gbr}
iso = IsolationForest(contamination="auto", n_estimators=250, bootstrap=True)
yhat = iso.fit_predict(x_train)
mask = yhat != -1
x_train_iso, y_train_iso = x_train[mask, :], y_train[mask]
print('Shape after outlier removal')
print(x_train_iso.shape, y_train_iso.shape)
outliers = y_train[yhat == -1]
print("Outliers Detected: %d" % len(outliers))

pyplot.figure()
pyplot.title("Daily Step Counts in Dataset\nOutliers Found: %d" % len(outliers))
pyplot.scatter([i for i in range(len(outliers))], outliers, c="red")
pyplot.scatter([i for i in range(len(y_train_iso))], y_train_iso, c="blue")
pyplot.xlabel("Days in dataset")
pyplot.ylabel("Daily step count")
pyplot.legend(['Outliers', 'Non-no_of_outliers'], loc="upper right")
pyplot.show()

pyplot.figure()
pyplot.title("Outliers in Dataset: %d" % len(outliers))
pyplot.scatter([i for i in range(len(outliers))], outliers, c="red")
pyplot.xlabel("Days in dataset")
pyplot.ylabel("Daily step count")
pyplot.legend('Outliers', loc="upper right")
pyplot.show()

cv = KFold(n_splits=5, shuffle=False)
results_with_outliers, results_without_outliers, names = list(), list(), list()

models_without_outliers = utilities.get_models(['ri', 'dt', 'rf', 'gbr'])
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
    predictor.fit(x_train_iso, y_train_iso)
    filename = 'no_outliers_no_rfe_model_%s_%d.sav' % (name, n_per_in)
    print("Writing %s model to file..." % name)
    pickle.dump(predictor, open(filename, 'wb'))
    # evaluate model
    print("Getting predictions for validation set")
    y_predicted = predictor.predict(x_test)
    print('MAE in validation set for %s: %.3f' % (name, mean_absolute_error(y_test, y_predicted)))
    print("Error as percentage of actual step count: %d" % utilities.step_error_as_percentage(y_test, y_predicted))
    score = predictor.best_score_
    results_without_outliers.append(score * -1)
    results_with_outliers.append(models_with_outliers[name].best_score_ * -1)
    names.append(name)
    # report performance
    print('>%s %.3f' % (name, score))
    pyplot.plot(y_predicted[:20], label='Predicted')
    # Printing and plotting the actual values
    pyplot.plot(y_test[:20], label='Actual')
    pyplot.xticks([i for i in range(len(y_predicted[:20]))], [i for i in range(len(y_predicted[:20]))])
    pyplot.title("Predicted vs Actual Daily Step Counts\nAlgo: %s" % name)
    pyplot.ylabel("Steps")
    pyplot.legend()
    pyplot.show()

pyplot.figure()
x = np.arange(len(results_without_outliers))
pyplot.bar(x+0.125, results_with_outliers, color='b', width=0.25)
pyplot.bar(x+0.375, results_without_outliers, color='r', width=0.25)
pyplot.legend(['Before Isolation Forest', 'After Isolation Forest'], loc="lower right")
pyplot.xticks([i + 0.25 for i in range(len(names))], names)
pyplot.title('BASELINE MEAN CV MAE VS CV MAE AFTER APPLYING\nISOLATION FOREST OUTLIER REMOVAL METHOD')
pyplot.xlabel('ALGORITHMS')
pyplot.ylabel('MAE IN NUMBER OF STEPS')
pyplot.show()

