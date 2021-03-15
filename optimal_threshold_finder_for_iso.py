import random
import utilities
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot
from statistics import mean, stdev

n_per_in = 5
n_per_out = 1
X, y = utilities.get_dataset(n_per_in, n_per_out)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

contamination = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5]
mae_cv_results, mae_test_set_results, mape_cv_results, mape_test_set_results = list(), list(), list(), list()
for contamination_value in contamination:
    iso = IsolationForest(contamination=contamination_value, n_estimators=250, bootstrap=True)
    yhat = iso.fit_predict(x_train)
    mask = yhat != -1
    x_train_iso, y_train_iso = x_train[mask, :], y_train[mask]
    outliers = y_train[yhat == -1]
    print("Outliers Detected: %d" % len(outliers))

    pyplot.figure()
    pyplot.title("Daily Step Counts in Dataset\nOutliers Found: %d for Contamination: %d%%"
                 % (len(outliers), contamination_value * 100))
    pyplot.scatter([i for i in random.sample(range(0, len(y_train_iso)), len(outliers))], outliers, c="red")
    pyplot.scatter([i for i in range(len(y_train_iso))], y_train_iso, c="blue")
    pyplot.xlabel("Step Count Index in Dataset")
    pyplot.ylabel("Daily Step Count")
    pyplot.legend(['Outliers', 'Non-outliers'], loc="upper right")
    pyplot.show()
    params = {'criterion': 'mae', 'loss': 'ls', 'max_depth': 7, 'max_features': 'log2'}
    custom_scorer = make_scorer(utilities.step_error_as_percentage, greater_is_better=False)
    gbr = GradientBoostingRegressor(**params)
    cv = KFold(n_splits=5, shuffle=False)
    scoring = {'mae': 'neg_mean_absolute_error', 'perc_scorer': custom_scorer}
    print("Evaluating model for contamination: %d%%" % (contamination_value * 100))
    scores = cross_validate(
        gbr,
        x_train_iso,
        y_train_iso,
        scoring=scoring,
        cv=cv,
        return_estimator=True
    )
    print("Fitting Model to training data, contamination: %d%%" % (contamination_value * 100))
    gbr.fit(x_train_iso, y_train_iso)
    y_pred = gbr.predict(x_test)
    test_set_mae = mean_absolute_error(y_test, y_pred)
    test_set_mape = utilities.step_error_as_percentage(y_test, y_pred)
    print("Outliers: %d, Contamination: %d%%, Mean CV MAE: %d, STDEV: %d\n"
          "Mean Percentage Error: %2f%%" % (len(outliers), (contamination_value * 100), mean(scores['test_mae']),
                                            stdev(scores['test_mae']), mean(scores['test_perc_scorer'])))
    print("MAE in validation set: %d" % test_set_mae)
    mae_cv_results.append(mean(scores['test_mae']) * -1)
    mae_test_set_results.append(test_set_mae)
    mape_cv_results.append(mean(scores['test_perc_scorer']))
    mape_test_set_results.append(test_set_mape)

pyplot.figure()
pyplot.title("Plot of MAE and Different Percentages of Outliers Removed")
pyplot.plot(mae_cv_results, label="Mean CV MAE")
pyplot.plot(mae_test_set_results, label="MAE in Test Set")
pyplot.xticks([i + 0.25 for i in range(len(contamination))],
              [str(threshold) for threshold in contamination])
pyplot.xlabel("Percentage of Samples Removed as Outliers")
pyplot.ylabel("MAE in Steps")
pyplot.legend()
pyplot.show()

pyplot.figure()
pyplot.title("Plot of MAPE and Different Percentages of Outliers Removed")
pyplot.plot(mape_cv_results, label="Mean CV MAPE")
pyplot.plot(mape_test_set_results, label="MAPE in Test Set")
pyplot.xticks([i + 0.25 for i in range(len(contamination))],
              [str(threshold) for threshold in contamination])
pyplot.xlabel("Percentage of Samples Removed as Outliers")
pyplot.ylabel("MAPE (%)")
pyplot.legend()
pyplot.show()
