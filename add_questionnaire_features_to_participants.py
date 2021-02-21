import pandas as pd
import os

questionary_features = pd.read_excel(
    '07. additional_data/02. questionary_data/questionary_features.xlsx',
    header=0,
    engine='openpyxl',
    index_col='hash'
)
questionary_features = questionary_features.iloc[:23, :31]

activity_features = os.fsencode('05. activity_and_date_features')
participant_id = 0
for csv in os.listdir(activity_features):
    df = pd.DataFrame()
    ids = []
    # For csv
    activity_sample_hash = csv.decode('utf-8')[:-4]
    # For xlsx
    # activity_sample_hash = csv.decode('utf-8')[:-5]
    for questionary_sample_hash in questionary_features.index:
        if activity_sample_hash == questionary_sample_hash:
            activity_data = pd.read_csv(os.path.join(activity_features, csv).decode('utf-8'))
            # activity_data = pd.read_excel(os.path.join(activity_features, csv).decode('utf-8'), engine='openpyxl', header=0,
            #                    index_col='date')
            questionary_row = questionary_features.loc[activity_sample_hash]
            for i in range(len(activity_data)):
                df = df.append(questionary_row)
                ids.append(participant_id)
            participant_id += 1
            activity_data.reset_index(drop=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
            ids = pd.Series(ids, name='participant_id')
            dataset = pd.concat([activity_data, ids], axis=1)
            print(len(dataset) == len(activity_data))
            dataset = dataset.set_index(dataset.columns[0])
            # dataset = dataset.rename(columns={'index': 'hash_id'})
            path = os.fsencode('09. activity_date_ids')
            print("Writing file {}.......".format(csv.decode("utf-8")))
            dataset.to_excel(os.path.join(path.decode("utf-8"), activity_sample_hash+'.xlsx'))
