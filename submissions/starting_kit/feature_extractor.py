
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler


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
                                'Mean TemperatureC', 'dep_encod', 'ar_encod',
                                'MeanDew PointC', 'Mean Humidity', 'Min VisibilitykM', 'Max Wind SpeedKm/h',
                                'Precipitationmm','CloudCover','Events']]

        X_encoded = pd.merge(X_encoded, external_data, how='left',
            left_on=['DateOfDeparture', 'Departure', 'Arrival'],
            right_on=['DateOfDeparture','Departure', 'Arrival'],
            sort=False)
#start

#One hot encoder for departure and arrival
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)

#Mean encoding
        #X_encoded.loc[:, 'd_ATL': 'd_SFO'] = X_encoded.loc[:, 'd_ATL': 'd_SFO'].mul(X_encoded.loc[:, 'dep_encod'], axis='rows')
        #X_encoded.loc[:, 'a_ATL': 'a_SFO'] = X_encoded.loc[:, 'a_ATL': 'a_SFO'].mul(X_encoded.loc[:, 'ar_encod'], axis='rows')

        #X_encoded.drop(['dep_encod', 'ar_encod'], axis=1, inplace=True)

#One hot encoder for dates
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))

        X_encoded.drop(['DateOfDeparture', 'year', 'month', 'day', 'weekday', 'week', 'n_days'], axis=1, inplace=True)

#reservations
        #X_encoded.loc[:, 'WeeksToDeparture' : 'std_wtd'] = StandardScaler().fit_transform(X_encoded.loc[:, 'WeeksToDeparture' : 'std_wtd'])

        X_encoded.drop(['std_wtd'], axis=1, inplace=True)

#Create three categories for Events
        X_encoded['Events'].fillna(0, inplace=True)
        X_encoded['Events'] = X_encoded['Events'].replace(['Rain','Fog'], 1)
        X_encoded['Events'] = X_encoded['Events'].replace(['Rain-Thunderstorm','Fog-Rain-Thunderstorm', \
                                                        'Rain-Snow','Snow','Fog-Rain','Thunderstorm','Fog-Snow', \
                                                        'Fog-Rain-Snow','Fog-Rain-Snow-Thunderstorm', \
                                                        'Rain-Snow-Thunderstorm','Rain-Hail-Thunderstorm', \
                                                        'Fog-Rain-Hail-Thunderstorm', \
                                                        'Rain-Thunderstorm-Tornado'], 2)
#Missingvalues Precipitationmm
        X_encoded['Precipitationmm'] = X_encoded['Precipitationmm'].replace('T', np.nan)
        X_encoded['Precipitationmm'] = pd.to_numeric(X_encoded['Precipitationmm'])
        X_encoded['Precipitationmm'] = X_encoded['Precipitationmm'].fillna(X_encoded['Precipitationmm'].mean())

#create columns
        #X_encoded['Precipitationmm*CloudCover'] = X_encoded['Precipitationmm'] * X_encoded['CloudCover']
        #X_encoded['Temperature*Humidity'] = X_encoded['Mean TemperatureC'] * X_encoded['Mean Humidity']

# Regroup values
        X_encoded['Weeks_to_dep_int'] = pd.qcut(X_encoded['WeeksToDeparture'], 4)
        X_encoded.loc[X_encoded['WeeksToDeparture'] <= 9.524, 'WeeksToDeparture'] = 0
        X_encoded.loc[(X_encoded['WeeksToDeparture'] > 9.524) & (X_encoded['WeeksToDeparture'] <= 11.3), 'WeeksToDeparture'] = 1
        X_encoded.loc[(X_encoded['WeeksToDeparture'] > 11.3) & (X_encoded['WeeksToDeparture'] <= 13.24), 'WeeksToDeparture'] = 2
        X_encoded.loc[ X_encoded['WeeksToDeparture'] > 13.24, 'WeeksToDeparture'] = 3
        X_encoded['WeeksToDeparture'] = X_encoded['WeeksToDeparture'].astype(int)
        X_encoded.drop(['Weeks_to_dep_int'], axis=1, inplace=True)

#Create column weekend
        #X_encoded['weekend'] = X_encoded['wd_5'] + X_encoded['wd_6']
        #X_encoded.drop(['wd_0', 'wd_1', 'wd_2', 'wd_3', 'wd_4', 'wd_5','wd_6'], axis=1, inplace=True)
# StandardScaler
        #X_encoded.loc[:, 'Precipitationmm*CloudCover' : 'Temperature*Humidity'] = StandardScaler().fit_transform(
                                                            #X_encoded.loc[:, 'Precipitationmm*CloudCover' : 'Temperature*Humidity'])
#drop what we don't use for now
        X_encoded.drop(['Mean TemperatureC', 'Mean Humidity', 'MeanDew PointC', 'Precipitationmm', 'CloudCover'], axis=1, inplace=True)

#end

        X_array = X_encoded.values
        return X_array
