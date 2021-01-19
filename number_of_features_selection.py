import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
import pandas as pd

n_per_in = 7
n_per_out = 1

dataset = pd.read_excel(
    'all_data_aggregated\\aggregated_dataset_in_'+str(n_per_in)+'_out_'+str(n_per_out)+'.xlsx',
    header=0,
    engine='openpyxl',
    index_col='date'
)

y = dataset['var1(t)']
X = dataset.drop(columns='var1(t)')

scaler = RobustScaler()
print("Normalizing data.....")
X_rescaled = scaler.fit_transform(X)

estimator = DecisionTreeRegressor(criterion="mae")
min_features_to_select = n_per_in

print("Conducting feature selection (estimator Decision tree).....")
rfecv = RFECV(estimator=estimator, step=n_per_in, cv=KFold(shuffle=False),
              scoring='neg_mean_absolute_error',
              min_features_to_select=min_features_to_select)
rfecv.fit(X_rescaled, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()
