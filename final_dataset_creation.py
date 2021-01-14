import pandas as pd
import os

questionary_features = pd.read_excel(
    'questionary/questionary_features.xlsx',
    header=0,
    engine='openpyxl',
    index_col='hash'
)
questionary_features = questionary_features.iloc[:23, :31]

activity_features = os.fsencode('preprocessed_date_data')
for csv in os.listdir(activity_features):
    df = pd.DataFrame()
    activity_sample_hash = csv.decode('utf-8')[:-4]
    for questionary_sample_hash in questionary_features.index:
        if activity_sample_hash == questionary_sample_hash:
            activity_data = pd.read_csv(os.path.join(activity_features, csv).decode('utf-8'))
            questionary_row = questionary_features.loc[activity_sample_hash]
            for i in range(len(activity_data)):
                df = df.append(questionary_row)
            activity_data.reset_index(drop=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
            dataset = pd.concat([activity_data, df], axis=1)
            print(len(dataset) == len(activity_data))
            dataset = dataset.set_index(dataset.columns[0])
            path = os.fsencode('final_datasets')
            print("Writing file {}.......".format(csv.decode("utf-8")))
            dataset.to_excel(os.path.join(path.decode("utf-8"), activity_sample_hash+'.xlsx'))
