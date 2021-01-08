import holidays
from datetime import datetime
import pandas as pd
import os


def preprocessing():
    # dataset_path = "C:\\Users\\Dimitris.DESKTOP-VO6DORB\\OneDrive\\Thesis\\06. Data\\Preprocessed Data"
    # dataset_dir = os.fsencode(dataset_path)
    # csv = os.listdir(dataset_dir)[0]
    gr_holidays = holidays.GR()
    individual_data = os.fsencode('individual data')
    # Iterating through the individual data directory
    for csv in os.listdir(individual_data):
        df = pd.read_csv(os.path.join(individual_data, csv).decode('utf-8'))
        is_holiday_df = pd.DataFrame(columns=['is_holiday'])
        day_of_week = pd.DataFrame(columns=['day_of_week'])
        is_weekend = pd.DataFrame(columns=['is_weekend'])
        month = pd.DataFrame(columns=['month'])
        day_in_month = pd.DataFrame(columns=['day_in_month'])

        for index, row in df.iterrows():
            no_of_day_in_week = datetime.strptime(row['date'], '%Y-%m-%d').weekday()
            if no_of_day_in_week == 5 or no_of_day_in_week == 6:
                is_weekend = is_weekend.append({'is_weekend': True}, ignore_index=True)
            else:
                is_weekend = is_weekend.append({'is_weekend': False}, ignore_index=True)
            no_of_month = datetime.strptime(row['date'], '%Y-%m-%d').month
            month = month.append({'month': no_of_month}, ignore_index=True)
            no_of_day = datetime.strptime(row['date'], '%Y-%m-%d').day
            day_in_month = day_in_month.append({'day_in_month': no_of_day}, ignore_index=True)
            is_holiday_df = is_holiday_df.append({'is_holiday': row['date'] in gr_holidays}, ignore_index=True)
            day_of_week = day_of_week.append({'day_of_week': no_of_day_in_week}, ignore_index=True)

        df = pd.concat([df, is_holiday_df, day_of_week, is_weekend, month, day_in_month], axis=1)
        df = df.set_index(df.columns[0])
        path = os.fsencode('preprocessed_date_data')
        print("Writing file {}.......".format(csv.decode("utf-8")))
        df.to_csv(os.path.join(path.decode("utf-8"), csv.decode("utf-8")))


if __name__ == '__main__':
    preprocessing()
