
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

        external_data = ext_data[['DateOfDeparture', 'Departure', 'Arrival', 'Mean TemperatureC', 'Distance',
                      'MeanDew PointC', 'Mean Humidity', 'Mean VisibilityKm', 'Mean Sea Level PressurehPa', 'CloudCover',
                      'Max Wind SpeedKm/h', 'Precipitationmm', 'Events', 'dep_encod', 'ar_encod', 'Revenue', 'Number_hab',
                      'Oil_price']]

        X_encoded = pd.merge(X_encoded, external_data, how='left',
            left_on=['DateOfDeparture', 'Departure', 'Arrival'],
            right_on=['DateOfDeparture','Departure', 'Arrival'],
            sort=False)
#start

        X_encoded.drop(['std_wtd'], axis=1, inplace=True)
#One hot encoder for departure and arrival
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop(['Arrival','Departure'], axis=1)

#Mean encoding
        X_encoded.loc[:, 'd_ATL': 'd_SFO'] = X_encoded.loc[:, 'd_ATL': 'd_SFO'].mul(X_encoded.loc[:, 'dep_encod'], axis='rows')
        X_encoded.loc[:, 'a_ATL': 'a_SFO'] = X_encoded.loc[:, 'a_ATL': 'a_SFO'].mul(X_encoded.loc[:, 'ar_encod'], axis='rows')

        X_encoded.drop(['dep_encod', 'ar_encod'], axis=1, inplace=True)

#One hot encoder for dates
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        #X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        #X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))

        X_encoded.drop(['DateOfDeparture', 'year', 'day', 'weekday', 'week', 'n_days'], axis=1, inplace=True)

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
        #X_encoded['Weeks_to_dep_int'] = pd.qcut(X_encoded['WeeksToDeparture'], 4)
        #X_encoded.loc[X_encoded['WeeksToDeparture'] <= 9.524, 'WeeksToDeparture'] = 0
        #X_encoded.loc[(X_encoded['WeeksToDeparture'] > 9.524) & (X_encoded['WeeksToDeparture'] <= 11.3), 'WeeksToDeparture'] = 1
        #X_encoded.loc[(X_encoded['WeeksToDeparture'] > 11.3) & (X_encoded['WeeksToDeparture'] <= 13.24), 'WeeksToDeparture'] = 2
        #X_encoded.loc[ X_encoded['WeeksToDeparture'] > 13.24, 'WeeksToDeparture'] = 3
        #X_encoded['WeeksToDeparture'] = X_encoded['WeeksToDeparture'].astype(int)
        #X_encoded.drop(['Weeks_to_dep_int'], axis=1, inplace=True)

#Date feature enginereeing
        X_encoded['winter'] = X_encoded['m_1'] + X_encoded['m_2']+ X_encoded['m_3']+ X_encoded['m_11']+X_encoded['m_12']
        X_encoded.drop(['m_1', 'm_2', 'm_3', 'm_11', 'm_12'], axis=1, inplace=True)
        #X_encoded.drop(['m_1', 'm_2', 'm_3', 'm_4', 'm_5','m_6','m_7', 'm_8', 'm_9', 'm_10', 'm_11', 'm_12'], axis=1, inplace=True)
        X_encoded['holiday_week'] = X_encoded['w_1'] + X_encoded['w_27'] + X_encoded['w_47'] + X_encoded['w_52']
        #X_encoded.drop(['w_1', 'w_2', 'w_3', 'w_4', 'w_5','w_6','w_7', 'w_8', 'w_9', 'w_10',
        #                'w_11', 'w_12', 'w_13', 'w_14', 'w_15','w_16','w_17', 'w_18', 'w_19', 'w_20',
        #                'w_21', 'w_22', 'w_23', 'w_24', 'w_25','w_26','w_27', 'w_28', 'w_29', 'w_30',
        #                'w_31', 'w_32', 'w_33', 'w_34', 'w_35','w_36','w_37', 'w_38', 'w_39', 'w_40',
        #                'w_41', 'w_42', 'w_43', 'w_44', 'w_45','w_46','w_47', 'w_48', 'w_49', 'w_50', 'w_51', 'w_52',
        #                ], axis=1, inplace=True)
        X_encoded.drop(['w_1', 'w_27', 'w_47', 'w_52'], axis=1, inplace=True)
        X_encoded['weekend'] = X_encoded['wd_0'] + X_encoded['wd_5'] + X_encoded['wd_6']
        X_encoded.drop(['wd_0', 'wd_1', 'wd_2', 'wd_3', 'wd_4', 'wd_5','wd_6'], axis=1, inplace=True)
# StandardScaler
        #X_encoded.loc[:, 'Precipitationmm*CloudCover' : 'Temperature*Humidity'] = StandardScaler().fit_transform(
                                                            #X_encoded.loc[:, 'Precipitationmm*CloudCover' : 'Temperature*Humidity'])

        #X_encoded.loc[:, 'MeanDew PointC' : 'Max Wind SpeedKm/h'] = StandardScaler().fit_transform(
                                                            #X_encoded.loc[:, 'MeanDew PointC' : 'Max Wind SpeedKm/h'])

        #X_encoded.loc[:, 'Revenue' : 'Oil_price'] = StandardScaler().fit_transform(
                                                            #X_encoded.loc[:, 'Revenue' : 'Oil_price'])
#drop what we don't use for now
        #X_encoded.drop(['Mean TemperatureC', 'Mean Humidity', 'Precipitationmm', 'CloudCover'], axis=1, inplace=True)
        #X_encoded.drop(['MeanDew PointC', 'Mean Sea Level PressurehPa'], axis=1, inplace=True)
        #X_encoded.drop(['dep_encod','ar_encod','Revenue'], axis=1, inplace=True)

#end
        X_array = X_encoded.values
        return X_array
