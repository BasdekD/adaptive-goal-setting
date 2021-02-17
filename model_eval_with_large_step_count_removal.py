import pandas as pd
import random
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from statistics import mean, stdev
from matplotlib import pyplot

import utilities

n_per_in = 5
n_per_out = 1
print('Reading dataset....')
data = pd.read_excel(
        '01. aggregated_and_timeseries_transformed_datasets\\aggregated_dataset_in_' + str(n_per_in) + '_out_' + str(n_per_out) + '_new.xlsx',
        header=0,
        engine='openpyxl',
        index_col='date'
    )

y = data['var1(t)']
X = data.drop(columns='var1(t)')
print('Scaling dataset....')
scaler = RobustScaler()
x_rescaled = scaler.fit_transform(X.values)
x_rescaled_df = pd.DataFrame(x_rescaled, index=X.index, columns=X.columns)

x_train, x_test, y_train, y_test = train_test_split(x_rescaled_df, y, test_size=0.10, random_state=42)

train_data = pd.concat([x_train, y_train], axis=1)
train_data = train_data.sort_values(by='var1(t)', ascending=False)
percentage_threshold = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5]
cv_results, val_results, perc_results = list(), list(), list()
for threshold in percentage_threshold:
    no_of_outliers = round(len(train_data) * threshold)
    clipped_data = train_data.iloc[no_of_outliers:, ::]
    outliers = train_data.iloc[:no_of_outliers, ::]

    pyplot.figure()
    pyplot.title("Daily Step Counts in Dataset\nOutliers Found: %d for threshold: %.2f%%" % (no_of_outliers, threshold))
    pyplot.scatter([i for i in random.sample(range(0, len(clipped_data)), len(outliers))], outliers['var1(t)'], c="red")
    pyplot.scatter([i for i in range(len(clipped_data))], clipped_data['var1(t)'], c="blue")
    pyplot.xlabel("Index of Sample in Dataset")
    pyplot.ylabel("Daily step count")
    pyplot.legend(['Outliers', 'Non-Outliers'], loc="upper right")
    pyplot.show()
    params = {'criterion': 'mae', 'loss': 'ls', 'max_depth': 7, 'max_features': 'sqrt'}
    gbr = GradientBoostingRegressor(**params)
    print("Obtaining Cross Validation Scores for Threshold %.2f%%" % threshold)
    cv = KFold(shuffle=False)
    clipped_data = clipped_data.sample(frac=1)
    clipped_y = clipped_data['var1(t)']
    clipped_X = clipped_data.drop(columns='var1(t)')
    scores = cross_val_score(
        gbr,
        clipped_X,
        clipped_y,
        scoring='neg_mean_absolute_error',
        cv=cv
    )
    print("Fitting Model to training data, threshold %.2f" % threshold)
    gbr.fit(clipped_X, clipped_y)
    y_pred = gbr.predict(x_test)
    val_score = mean_absolute_error(y_test, y_pred)
    print("Outliers: %d, Threshold: %.2f%%, MEAN CV MAE: %d, STDEV: %d" %
          (no_of_outliers, threshold, mean(scores), stdev(scores)))
    print("MAE in validation set: %d" % val_score)
    cv_results.append(mean(scores) * -1)
    val_results.append(val_score)
    perc_results.append(utilities.step_error_as_percentage(y_test, y_pred))

pyplot.figure()
pyplot.title("Plot of MAE and Different Percentages of Outliers Removed")
pyplot.plot(cv_results, label="Mean CV MAE")
pyplot.plot(val_results, label="MAE in Validation Set")
pyplot.xticks([i+0.25 for i in range(len(percentage_threshold))], [str(threshold)+'%' for threshold in percentage_threshold])
pyplot.xlabel("Percentage of Samples Removed as Outliers")
pyplot.ylabel("MAE in Steps")
pyplot.legend()
pyplot.show()

pyplot.figure()
pyplot.title("Plot of Error as Percentage with\nDifferent Percentages of Outliers Removed")
pyplot.plot(perc_results, label="Error in Validation set\nas percentage of actual step count")
pyplot.xticks([i+0.25 for i in range(len(percentage_threshold))], [str(threshold)+'%' for threshold in percentage_threshold])
pyplot.xlabel("Percentage of Samples Removed as Outliers")
pyplot.ylabel("Error as percentage of actual step count")
pyplot.legend()
pyplot.show()