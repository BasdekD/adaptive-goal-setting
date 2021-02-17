import utilities
import pickle
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, make_scorer
from matplotlib import pyplot


n_per_in = 5
n_per_out = 1
X, y = utilities.get_dataset(n_per_in, n_per_out)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


models = utilities.get_models(['ri', 'dt', 'rf', 'gbr'])
cv = KFold(n_splits=5, shuffle=False)
custom_scorer = make_scorer(utilities.step_error_as_percentage, greater_is_better=False)
scoring = {'mae': 'neg_mean_absolute_error', 'perc_scorer': custom_scorer}
results,percentage_error, names = list(), list(), list()

for name, model in models.items():
    print("Looking for best parameters for %s model..." % name)
    predictor = GridSearchCV(
        estimator=model['estimator'],
        param_grid=model['grid_params'],
        scoring=scoring,
        cv=cv,
        refit='mae'
    )
    predictor.fit(x_train, y_train)
    filename = 'no_rfe_model_%s_%d.sav' % (name, n_per_in)
    print("Writing %s model to file..." % name)
    pickle.dump(predictor, open(filename, 'wb'))
    # evaluate model
    print("Getting predictions for validation set")
    y_predicted = predictor.predict(x_test)
    print('MAE in validation set for %s: %.3f' % (name, mean_absolute_error(y_test, y_predicted)))
    error_as_percentage = predictor.cv_results_['mean_test_perc_scorer'][predictor.best_index_] * -1
    score = predictor.best_score_ * -1
    results.append(score)
    percentage_error.append(error_as_percentage)
    names.append(name.upper())
    # report performance
    print('>%s CV MAE: %.3f, PERC_ERROR: %.2f%%' % (name, score, error_as_percentage))
    pyplot.plot(y_predicted[:20], label='Predicted')
    # Printing and plotting the actual values
    pyplot.plot(y_test[:20], label='Actual')
    pyplot.title(f"Predicted vs Actual Daily Step Counts\nAlgo: %s" % name.upper())
    pyplot.xticks([i for i in range(len(y_predicted[:20]))], [i for i in range(len(y_predicted[:20]))])
    pyplot.ylabel("Steps")
    pyplot.legend()
    pyplot.show()

pyplot.figure()
pyplot.title('CV MAE PER MODEL')
pyplot.xlabel("Algorithms")
pyplot.ylabel("MAE")
pyplot.bar(names, results)
pyplot.show()

pyplot.figure()
pyplot.title('Error as Percentage per Model')
pyplot.xlabel("Algorithms")
pyplot.ylabel("Error as Percentage")
pyplot.bar(names, percentage_error)
pyplot.show()
