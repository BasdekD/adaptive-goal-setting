import utilities
import pickle
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot


n_per_in = 5
n_per_out = 1
X, y = utilities.get_dataset(n_per_in, n_per_out)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


models = utilities.get_models(['ri', 'dt', 'rf', 'gbr'])
cv = KFold(n_splits=5, shuffle=False)
results, names = list(), list()

for name, model in models.items():
    print("Looking for best parameters for %s model..." % name)
    predictor = GridSearchCV(
        estimator=model['estimator'],
        param_grid=model['grid_params'],
        scoring='neg_mean_absolute_error',
        cv=cv,
        refit=True
    )
    predictor.fit(x_train, y_train)
    filename = 'no_rfe_model_%s_%d.sav' % (name, n_per_in)
    print("Writing %s model to file..." % name)
    pickle.dump(predictor, open(filename, 'wb'))
    # evaluate model
    print("Getting predictions for validation set")
    y_predicted = predictor.predict(x_test)
    print('MAE in validation set for %s: %.3f' % (name, mean_absolute_error(y_test, y_predicted)))
    score = predictor.best_score_
    results.append(score)
    names.append(name)
    # report performance
    print('>%s %.3f' % (name, score))
    pyplot.plot(y_predicted[:20], label='Predicted')
    # Printing and plotting the actual values
    pyplot.plot(y_test[:20], label='Actual')
    pyplot.title(f"Predicted vs Actual Daily Step Counts")
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
