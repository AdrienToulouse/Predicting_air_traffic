
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df

        path = os.path.dirname(__file__)

# import data_weather and merge it
        ext_data = pd.read_csv(os.path.join(path, 'external_data.csv'), sep=';')

        external_data = ext_data[['DateOfDeparture', 'Departure', 'Arrival', 'Distance',
                      'dep_encod', 'ar_encod',
                      'Mean TemperatureC', 'MeanDew PointC', 'Mean Humidity', 'Min VisibilitykM',
                      'Max Wind SpeedKm/h', 'Precipitationmm', 'Events'
                      ]]

        X_encoded = pd.merge(X_encoded, external_data, how='left',
            left_on=['DateOfDeparture', 'Departure', 'Arrival'],
            right_on=['DateOfDeparture','Departure', 'Arrival'],
            sort=False)
#start

        #X_encoded.drop(['std_wtd'], axis=1, inplace=True)

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

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))

        X_encoded.drop(['DateOfDeparture', 'year', 'month', 'day', 'weekday', 'week', 'n_days'], axis=1, inplace=True)

#PCA reservation
        reserv = preprocessing.scale(X_encoded.loc[:, 'WeeksToDeparture' : 'std_wtd'])
        pca = PCA(n_components=1)
        reserv = pca.fit_transform(reserv)
        X_encoded['reserv'] = reserv

#meteo
        X_encoded['Precipitationmm'] = X_encoded['Precipitationmm'].replace('T', np.nan)
        X_encoded['Precipitationmm'] = pd.to_numeric(X_encoded['Precipitationmm'])
        X_encoded['Precipitationmm'] = X_encoded['Precipitationmm'].fillna(X_encoded['Precipitationmm'].mean())

        X_encoded['Events'].fillna(0, inplace=True)
        X_encoded['Events'] = X_encoded['Events'].replace(['Rain','Fog'], 1)
        X_encoded['Events'] = X_encoded['Events'].replace(['Rain-Thunderstorm','Fog-Rain-Thunderstorm', \
                                                'Rain-Snow','Snow','Fog-Rain','Thunderstorm','Fog-Snow', \
                                                'Fog-Rain-Snow','Fog-Rain-Snow-Thunderstorm', \
                                                'Rain-Snow-Thunderstorm','Rain-Hail-Thunderstorm', \
                                                'Fog-Rain-Hail-Thunderstorm', \
                                                'Rain-Thunderstorm-Tornado'], 2)

        meteo = preprocessing.scale(X_encoded.loc[:, 'Mean TemperatureC' : 'Max Wind SpeedKm/h'])
        pca = PCA(n_components=2)
        meteo = pca.fit_transform(meteo)
        meteo = pd.DataFrame(data = meteo, columns = ['Princ. Comp. ' + str(i) for i in range(2)])

        X_encoded = pd.concat([X_encoded, meteo], axis = 1)

        X_encoded.drop(['Mean TemperatureC', 'Mean Humidity', 'Min VisibilitykM', 'MeanDew PointC',
         'Max Wind SpeedKm/h'], axis=1, inplace=True)
#end

        X_array = X_encoded.values
        return X_array
