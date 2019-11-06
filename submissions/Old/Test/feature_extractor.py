import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        path = os.path.dirname(__file__)
        data_weather = pd.read_csv(os.path.join(path, 'external_data.csv'))
        external_data = data_weather[['Date', 'AirPort', 'Max TemperatureC', 'Mean TemperatureC', 'Min TemperatureC',
                                       'MeanDew PointC', 'Mean Humidity', 'Mean VisibilityKm', 'Mean Wind SpeedKm/h',
                                       'CloudCover']]
        external_data = external_data.rename(
            columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
        X_encoded = pd.merge(
            X_encoded, external_data, how='left',
            left_on=['DateOfDeparture', 'Arrival'],
            right_on=['DateOfDeparture', 'Arrival'],
            sort=False)

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)

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

        X_encoded = X_encoded.drop(['year', 'month', 'day', 'weekday', 'week', 'n_days'], axis=1)

        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(X_encoded.loc[:, : 'CloudCover'])
        data_scaled = pd.DataFrame(data_scaled, columns=('WeeksToDeparture', 'std_wtd', 'Max TemperatureC',
                                                 'Mean TemperatureC', 'Min TemperatureC', 'MeanDew PointC',
                                                 'Mean Humidity', 'Mean VisibilityKm', 'Mean Wind SpeedKm/h',
                                                 'CloudCover'))

        X_encoded.drop(['WeeksToDeparture', 'std_wtd', 'Max TemperatureC',
                'Mean TemperatureC', 'Min TemperatureC', 'MeanDew PointC',
                'Mean Humidity', 'Mean VisibilityKm', 'Mean Wind SpeedKm/h',
                'CloudCover'], axis=1)
        X_encoded[['WeeksToDeparture', 'std_wtd', 'Max TemperatureC', 'Mean TemperatureC',
           'Min TemperatureC', 'MeanDew PointC', 'Mean Humidity', 'Mean VisibilityKm',
           'Mean Wind SpeedKm/h', 'CloudCover']] = data_scaled.loc[:,'WeeksToDeparture': 'CloudCover']

        X_array = X_encoded.values
        return X_array
