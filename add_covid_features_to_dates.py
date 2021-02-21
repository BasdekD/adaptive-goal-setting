import pandas as pd
import os
from datetime import datetime as dt

covid_features = pd.read_csv("07. additional_data\\01. covid_data\\covid_features.csv")
covid_features = covid_features.set_index(covid_features.columns[0])
covid_features.index = covid_features.index.astype('datetime64[ns]')

activity_data_directory = os.fsencode('05. activity_and_date_features')
path = os.fsencode('03. activity_date_and_covid_features')

for csv in os.listdir(activity_data_directory):
    print("Processing file %s" % csv.decode('utf-8')[:-4])
    df = pd.read_csv(os.path.join(activity_data_directory, csv).decode('utf-8'))
    df = df.set_index(df.columns[0])
    df.index.name = 'date'
    df.index = df.index.astype('datetime64[ns]')
    new_df = pd.merge(df, covid_features, how='left', on=['date'])
    new_df = new_df.fillna(5)
    new_df = new_df.reset_index()
    new_df['date'] = new_df['date'].dt.strftime('%Y-%m-%d')
    new_df = new_df.set_index(new_df.columns[0])
    new_df.to_excel(os.path.join(path.decode("utf-8"), csv.decode('utf-8')[:-4]+'.xlsx'))
