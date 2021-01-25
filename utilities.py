import pandas as pd
from sklearn.preprocessing import RobustScaler


# get the dataset
def get_dataset(n_per_in, n_per_out):
    print('Reading dataset....')
    dataset = pd.read_excel(
        'all_data_aggregated\\aggregated_dataset_in_' + str(n_per_in) + '_out_' + str(n_per_out) + '.xlsx',
        header=0,
        engine='openpyxl',
        index_col='date'
    )
    y = dataset['var1(t)']
    X = dataset.drop(columns='var1(t)')

    print('Scaling dataset....')
    scaler = RobustScaler()
    print("Normalizing data.....")
    x_rescaled = scaler.fit_transform(X)
    return x_rescaled, y
