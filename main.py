import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics, linear_model, tree, ensemble
from matplotlib import pyplot as plt
import numpy as np
import time

tic = time.perf_counter()

n_per_in = 7
n_per_out = 1
n_features = 38

dataset = pd.read_excel(
    'all_data_aggregated\\aggregated_dataset_in_'+str(n_per_in)+'_out_'+str(n_per_out)+'.xlsx',
    header=0,
    engine='openpyxl',
    index_col='date'
)
# dataset = pd.read_excel(
#     'all_data_aggregated\\aggregated_activity_only_dataset_in_'+str(n_per_in)+'_out_'+str(n_per_out)+'.xlsx',
#     header=0,
#     engine='openpyxl',
#     index_col='date'
# )

# Normalizing/Scaling the Data
# scaler = MinMaxScaler()
scaler = RobustScaler()
# dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns, index=dataset.index)


pd.options.display.width = 0
y = dataset['var1(t)']
X = dataset.drop(columns='var1(t)')
X_rescaled = scaler.fit_transform(X)

# print(X.head())
# print(len(X))
# print(y.head())

x_train, x_test, y_train, y_test = train_test_split(X_rescaled, y, test_size=0.20, random_state=42)

# model = linear_model.LinearRegression()
# model = linear_model.SGDRegressor()
# model = tree.DecisionTreeRegressor(criterion="mae", max_depth=10)
model = ensemble.RandomForestRegressor(criterion="mae")
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
# y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, n_features)).tolist()
# y_predicted = scaler.inverse_transform(np.array(y_predicted).reshape(-1, n_features)).tolist()
print("The MAE is {}".format(metrics.mean_absolute_error(y_test, y_predicted)))

# test_period = np.array(X_rescaled.iloc[-20:-1, :])
test_period = X_rescaled[-21:-1]
yhat = model.predict(test_period)
# Transforming values back to their normal prices
# yhat = scaler.inverse_transform(np.array(yhat).reshape(-1, n_features)).tolist()
yhat = pd.DataFrame(yhat)
actual = np.array(y.iloc[-21:-1])
# actual = scaler.inverse_transform(actual.reshape((-1, 1)))
actual = pd.DataFrame(actual)
print(metrics.mean_absolute_error(yhat, actual))

# Printing and plotting those predictions
print("Predicted Step Count:\n", yhat.loc[::, ::])

plt.plot(yhat.loc[::, ::], label='Predicted')

# Printing and plotting the actual values
print("\nActual Step Count:\n", actual.loc[::, ::])
plt.plot(actual.loc[::, ::], label='Actual')

plt.title(f"Predicted vs Actual Daily Step Counts")
plt.ylabel("Steps")
plt.legend()
plt.show()

toc = time.perf_counter()
print(f"Run in {toc - tic:0.4f} seconds")

