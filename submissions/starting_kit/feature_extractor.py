import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df

        path = os.path.dirname(__file__)

# import data_weather and merge it
        ext_data = pd.read_csv(os.path.join(path, 'external_data.csv'))

        external_data = ext_data[['DateOfDeparture', 'Departure', 'Arrival', 'Distance',
                      'dep_encod', 'ar_encod',
                      'Events'
                      ]]

        X_encoded = pd.merge(X_encoded, external_data, how='left',
            left_on=['DateOfDeparture', 'Departure', 'Arrival'],
            right_on=['DateOfDeparture','Departure', 'Arrival'],
            sort=False)
#start

#One hot encoder for departure and arrival
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure']))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded.drop(['Arrival','Departure'], axis=1, inplace=True)

#Mean encoding
        X_encoded.loc[:, 'ATL': 'SFO'] = X_encoded.loc[:, 'ATL': 'SFO'].mul(X_encoded.loc[:, 'dep_encod'], axis='rows')
        arr = X_encoded.loc[:, 'a_ATL': 'a_SFO'].mul(- 1 * X_encoded.loc[:, 'ar_encod'], axis='rows')

        arr.columns = X_encoded.loc[:, 'ATL': 'SFO'].columns
        X_encoded.loc[:, 'ATL': 'SFO'] = X_encoded.loc[:, 'ATL': 'SFO'].add(arr, axis=1)

        X_encoded.drop(['dep_encod', 'ar_encod'], axis=1, inplace=True)
        X_encoded.drop(X_encoded.loc[:, 'a_ATL': 'a_SFO'].columns, axis=1, inplace=True)

#One hot encoder for dates
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

        X_encoded = encode(X_encoded, 'month', 12)
        X_encoded = encode(X_encoded, 'week', 52)
        X_encoded = encode(X_encoded, 'day', 365)
        X_encoded = encode(X_encoded, 'weekday', 6)

        X_encoded.drop(['DateOfDeparture', 'year', 'month', 'day', 'weekday', 'week', 'std_wtd'], axis=1, inplace=True)

#meteo

        X_encoded['Events'].fillna(0, inplace=True)
        X_encoded['Events'] = X_encoded['Events'].replace(['Rain','Fog'], 1)
        X_encoded['Events'] = X_encoded['Events'].replace(['Rain-Thunderstorm','Fog-Rain-Thunderstorm', \
                                                'Rain-Snow','Snow','Fog-Rain','Thunderstorm','Fog-Snow', \
                                                'Fog-Rain-Snow','Fog-Rain-Snow-Thunderstorm', \
                                                'Rain-Snow-Thunderstorm','Rain-Hail-Thunderstorm', \
                                                'Fog-Rain-Hail-Thunderstorm', \
                                                'Rain-Thunderstorm-Tornado'], 2)

        X_encoded['Weeks_to_dep_int'] = pd.qcut(X_encoded['WeeksToDeparture'], 4)
        X_encoded.loc[X_encoded['WeeksToDeparture'] <= 9.524, 'WeeksToDeparture'] = 0
        X_encoded.loc[(X_encoded['WeeksToDeparture'] > 9.524) & (X_encoded['WeeksToDeparture'] <= 11.3), 'WeeksToDeparture'] = 1
        X_encoded.loc[(X_encoded['WeeksToDeparture'] > 11.3) & (X_encoded['WeeksToDeparture'] <= 13.24), 'WeeksToDeparture'] = 2
        X_encoded.loc[ X_encoded['WeeksToDeparture'] > 13.24, 'WeeksToDeparture'] = 3
        X_encoded['WeeksToDeparture'] = X_encoded['WeeksToDeparture'].astype(int)
        X_encoded.drop(['Weeks_to_dep_int'], axis=1, inplace=True)

        X_array = X_encoded.values
        return X_array
