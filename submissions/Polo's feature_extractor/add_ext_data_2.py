import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def _encode(X_df):
    X_encoded = X_df
    path = os.path.dirname(__file__)
    ext_data = pd.read_csv(os.path.join(path, 'external_data.csv'))
    external_data = ext_data[['DateOfDeparture', 'Departure', 'Arrival', 'dep_encod', 'ar_encod', 'Distance']]
                                #'Mean TemperatureC', 'MeanDew PointC', 'Mean Humidity', 'Min VisibilitykM',
                                #'Max Wind SpeedKm/h', 'Precipitationmm','CloudCover','Events']]
    X_encoded = pd.merge(X_encoded, external_data, how='left',
            left_on=['DateOfDeparture', 'Departure', 'Arrival'],
            right_on=['DateOfDeparture','Departure', 'Arrival'],
            sort=False)

    X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
    X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
    X_encoded = X_encoded.drop('Departure', axis=1)
    X_encoded = X_encoded.drop('Arrival', axis=1)

    #X_encoded.loc[:, 'd_ATL': 'd_SFO'] = X_encoded.loc[:, 'd_ATL': 'd_SFO'].mul(X_encoded.loc[:, 'dep_encod'], axis='rows')
    #X_encoded.loc[:, 'a_ATL': 'a_SFO'] = X_encoded.loc[:, 'a_ATL': 'a_SFO'].mul(X_encoded.loc[:, 'ar_encod'], axis='rows')

    X_encoded = X_encoded.drop(['dep_encod', 'ar_encod'], axis=1)

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

    X_encoded = X_encoded.drop(['DateOfDeparture', 'year', 'month', 'day', 'weekday', 'week', 'n_days'], axis=1)

    X_encoded.loc[:, 'WeeksToDeparture' : 'std_wtd'] = StandardScaler().fit_transform(X_encoded.loc[:, 'WeeksToDeparture' : 'std_wtd'])

    X_encoded = X_encoded.drop(['std_wtd'], axis=1)

    #X_encoded['Events'].fillna(0, inplace=True)
    #X_encoded['Events'] = X_encoded['Events'].replace(['Rain','Fog'], 1)
    #X_encoded['Events'] = X_encoded['Events'].replace(['Rain-Thunderstorm','Fog-Rain-Thunderstorm', \
                                            #'Rain-Snow','Snow','Fog-Rain','Thunderstorm','Fog-Snow', \
                                            #'Fog-Rain-Snow','Fog-Rain-Snow-Thunderstorm', \
                                            #'Rain-Snow-Thunderstorm','Rain-Hail-Thunderstorm', \
                                            #'Fog-Rain-Hail-Thunderstorm', \
                                            #'Rain-Thunderstorm-Tornado'], 2)
#Missingvalues Precipitationmm
    #X_encoded['Precipitationmm'] = X_encoded['Precipitationmm'].replace('T', np.nan)
    #X_encoded['Precipitationmm'] = pd.to_numeric(X_encoded['Precipitationmm'])
    #X_encoded['Precipitationmm'] = X_encoded['Precipitationmm'].fillna(X_encoded['Precipitationmm'].mean())


    return X_encoded


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        X_encoded = _encode(X_df)
        self.columns = X_encoded.columns
        return self

    def transform(self, X_df):
        X_encoded = _encode(X_df)
        X_empty = pd.DataFrame(columns=self.columns)
        X_encoded = pd.concat([X_empty, X_encoded], axis=0, sort=False)
        X_encoded = X_encoded.fillna(0)

        # Reorder/Pick columns from train
        X_encoded = X_encoded[list(self.columns)]
        # Check that columns of test set are the same than train set
        assert list(X_encoded.columns) == list(self.columns)
        X_array = X_encoded.values
        return X_array