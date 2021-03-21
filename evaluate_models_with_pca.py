from statistics import mean
from sklearn.ensemble import GradientBoostingRegressor
import utilities
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from matplotlib import pyplot


# get a list of models to evaluate
def get_PCA_models(input_data, predictor):
    models = dict()
    for i in range(len(input_data.columns), n_per_in, -n_per_in):
        steps = [('pca', PCA(n_components=i)), ('p', predictor)]
        models[str(i)] = Pipeline(steps=steps)
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y, scoring):
    cv = KFold(shuffle=False)
    scores = cross_validate(
        model,
        X,
        y,
        scoring=scoring,
        cv=cv
    )
    return scores


n_per_in = 5
n_per_out = 1
X, y = utilities.get_dataset_without_outliers(n_per_in, n_per_out)
params = {'criterion': 'mae', 'loss': 'ls', 'max_depth': 7, 'max_features': 'log2'}
gbr = GradientBoostingRegressor(**params)
models = get_PCA_models(X, gbr)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
custom_scorer = make_scorer(utilities.step_error_as_percentage, greater_is_better=False)
scoring = {'mae': 'neg_mean_absolute_error', 'perc_scorer': custom_scorer}
print("Obtaining Cross Validation Scores")
# evaluate the models and store results
cv_mae_results, cv_mape_results, names = list(), list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y, scoring)
    cv_mae_results.append(mean(scores['test_mae']) * -1)
    cv_mape_results.append(mean(scores['test_perc_scorer'] * -1))
    names.append(name)
    print('>%s MAE: %d, MAPE: %.2f%% ' % (name, mean(scores['test_mae']) * -1, mean(scores['test_perc_scorer'] * -1)))

# plot model performance for comparison
pyplot.boxplot(cv_mae_results, labels=names, showmeans=True)
pyplot.xticks(rotation=45)
pyplot.show()
