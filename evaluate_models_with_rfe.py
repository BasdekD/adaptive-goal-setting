from statistics import mean
from sklearn.ensemble import GradientBoostingRegressor
import utilities
import pickle
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.pipeline import Pipeline
from matplotlib import pyplot


n_per_in = 5
n_per_out = 1
X, y = utilities.get_dataset_without_outliers(n_per_in, n_per_out)
rfe = pickle.load(open('models\\03. models_after_rfe\\01. core_models\\rfecv_gbr_%d.sav' % n_per_in, 'rb'))
# x_reduced = rfe.transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
params = {'criterion': 'mae', 'loss': 'ls', 'max_depth': 7, 'max_features': 'log2'}
gbr = GradientBoostingRegressor(**params)
pipeline = Pipeline(steps=[('select', rfe), ('predictor', gbr)])
custom_scorer = make_scorer(utilities.step_error_as_percentage, greater_is_better=False)
scoring = {'mae': 'neg_mean_absolute_error', 'perc_scorer': custom_scorer}
print("Obtaining Cross Validation Scores")
cv = KFold(shuffle=False)
scores = cross_validate(
    pipeline,
    x_train,
    y_train,
    scoring=scoring,
    cv=cv
)
print("Fitting Model to training data")
pipeline.fit(x_train, y_train)
filename = 'rfe_no_outlier_model_pipe_%d.sav' % n_per_in
print("Writing GBR model to file...")
pickle.dump(pipeline, open(filename, 'wb'))
y_pred = pipeline.predict(x_test)
test_set_mae = mean_absolute_error(y_test, y_pred)
test_set_mape = utilities.step_error_as_percentage(y_test, y_pred)
print("MEAN CV MAE: %d, MEAN CV MAPE: %2f%%" % (mean(scores['test_mae']), mean(scores['test_perc_scorer'])))
print("MAE in test set: %d" % test_set_mae)
print("MAPE in test set: %.2f%%" % test_set_mape)
cv_mae = mean(scores['test_mae']) * -1
cv_mape = mean(scores['test_perc_scorer']) * -1

# report performance
pyplot.plot(y_pred[:20], label='Predicted')
# Printing and plotting the actual values
pyplot.plot(y_test[:20], label='Actual')
pyplot.title(f"Predicted vs Actual Daily Step Counts\nAlgo: GBR")
pyplot.xticks([i for i in range(len(y_pred[:20]))], [i for i in range(len(y_pred[:20]))])
pyplot.ylabel("Steps")
pyplot.legend()
pyplot.show()
