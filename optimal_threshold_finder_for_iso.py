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
cv_results, val_results, perc_results = list(), list(), list()
for contamination_value in contamination:
    iso = IsolationForest(contamination=contamination_value, n_estimators=250, bootstrap=True)
    yhat = iso.fit_predict(x_train)
    mask = yhat != -1
    x_train_iso, y_train_iso = x_train[mask, :], y_train[mask]
    outliers = y_train[yhat == -1]
    print("Outliers Detected: %d" % len(outliers))

    pyplot.figure()
    pyplot.title("Daily Step Counts in Dataset\nOutliers Found: %d for Contamination: %.2f%%"
                 % (len(outliers), contamination_value))
    pyplot.scatter([i for i in random.sample(range(0, len(y_train_iso)), len(outliers))], outliers, c="red")
    pyplot.scatter([i for i in range(len(y_train_iso))], y_train_iso, c="blue")
    pyplot.xlabel("Step Count Index in Dataset")
    pyplot.ylabel("Daily Step Count")
    pyplot.legend(['Outliers', 'Non-outliers'], loc="upper right")
    pyplot.show()
    params = {'criterion': 'mae', 'loss': 'ls', 'max_depth': 7, 'max_features': 'sqrt'}
    custom_scorer = make_scorer(utilities.step_error_as_percentage, greater_is_better=False)
    gbr = GradientBoostingRegressor(**params)
    cv = KFold(n_splits=5, shuffle=False)
    scoring = {'mae': 'neg_mean_absolute_error', 'perc_scorer': custom_scorer}
    print("Evaluating model for contamination: %.2f%%" % contamination_value)
    scores = cross_validate(
        gbr,
        x_train_iso,
        y_train_iso,
        scoring=scoring,
        cv=cv,
        return_estimator=True
    )
    print("Fitting Model to training data, contamination: %.2f%%" % contamination_value)
    gbr.fit(x_train_iso, y_train_iso)
    y_pred = gbr.predict(x_test)
    val_score = mean_absolute_error(y_test, y_pred)
    print("Outliers: %d, Contamination: %.2f%%, Mean CV MAE: %d, STDEV: %d\n"
          "Mean Percentage Error: %2f%%" % (len(outliers), contamination_value, mean(scores['test_mae']),
                                            stdev(scores['test_mae']), mean(scores['test_perc_scorer'])))
    print("MAE in validation set: %d" % val_score)
    cv_results.append(mean(scores['test_mae']) * -1)
    val_results.append(val_score)
    perc_results.append(mean(scores['test_perc_scorer']))

pyplot.figure()
pyplot.title("Plot of MAE and Different Percentages of Outliers Removed")
pyplot.plot(cv_results, label="Mean CV MAE")
pyplot.plot(val_results, label="MAE in Validation Set")
pyplot.xticks([i + 0.25 for i in range(len(contamination))],
              [str(threshold) + '%' for threshold in contamination])
pyplot.xlabel("Percentage of Samples Removed as Outliers")
pyplot.ylabel("MAE in Steps")
pyplot.legend()
pyplot.show()

pyplot.figure()
pyplot.title("Plot of Error as Percentage with\nDifferent Percentages of Outliers Removed")
pyplot.plot(perc_results, label="Error in Validation set\nas percentage of actual step count")
pyplot.xticks([i + 0.25 for i in range(len(contamination))],
              [str(threshold) + '%' for threshold in contamination])
pyplot.xlabel("Percentage of Samples Removed as Outliers")
pyplot.ylabel("Error as percentage of actual step count")
pyplot.legend()
pyplot.show()
