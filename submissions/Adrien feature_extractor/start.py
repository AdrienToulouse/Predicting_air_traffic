import pandas as pd
import os
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

        data_weather = pd.read_csv(os.path.join(path, 'external_data.csv'))
        X_weather = data_weather[['Date', 'AirPort', 'Mean TemperatureC',
                                'MeanDew PointC', 'Mean Humidity', 'Mean VisibilityKm', 'Max Wind SpeedKm/h',
                                'Precipitationmm','CloudCover','Events']]
        X_weather = X_weather.rename(
            columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
        X_encoded = pd.merge(
            X_encoded, X_weather, how='left',
            left_on=['DateOfDeparture', 'Arrival'],
            right_on=['DateOfDeparture', 'Arrival'],
            sort=False)
#start

    #One hot encoder for departure and arrival
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)

    #One hot encoder for dates
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])

        X_encoded['year']    = X_encoded['DateOfDeparture'].dt.year
        X_encoded['month']   = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day']     = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week']    = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days']  = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))

        X_encoded.drop(['year', 'month', 'day', 'weekday', 'week', 'n_days','DateOfDeparture'], axis=1, inplace=True)
    #Create two categories for Events
        X_encoded['Events'].fillna(0, inplace=True)
        X_encoded['Events'] = X_encoded['Events'].replace(['Rain','Rain-Thunderstorm','Fog','Fog-Rain-Thunderstorm', \
                                                         'Rain-Snow','Snow','Fog-Rain','Thunderstorm','Fog-Snow', \
                                                         'Fog-Rain-Snow','Fog-Rain-Snow-Thunderstorm', \
                                                         'Rain-Snow-Thunderstorm','Rain-Hail-Thunderstorm', \
                                                         'Fog-Rain-Hail-Thunderstorm', \
                                                         'Rain-Thunderstorm-Tornado'], 1)
    #Missingvalues Precipitationmm
        X_encoded['Precipitationmm'] = X_encoded['Precipitationmm'].replace(['T'], 0.00)
    #Create column weekend
        X_encoded['weekend'] = X_encoded['wd_5'] + X_encoded['wd_6']
    # StandardScaler
        X_encoded.loc[:, 'WeeksToDeparture' : 'CloudCover'] = StandardScaler().fit_transform(
                                                                X_encoded.loc[:, 'WeeksToDeparture' : 'CloudCover'])

#end
        X_array = X_encoded.values
        return X_array
