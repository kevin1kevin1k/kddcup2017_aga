
# coding: utf-8

# # TODO
# 
# use 7 numbers to indicate the counts of each vehicle_model
# 
# use mean of interpolation instead of zero (either filling X or y, especially y)

# In[ ]:

import pandas as pd
import datetime
import math
import numpy as np
from sklearn import preprocessing


# In[ ]:

vol_tolls = (1, 1, 2, 3, 3)
vol_dires = (0, 1, 0, 0, 1)
toll_dire = zip(vol_tolls, vol_dires)

tra_intes = ('A', 'A', 'B', 'B', 'C', 'C')
tra_tolls = (2, 3, 1, 3, 1, 3)
inte_toll = zip(tra_intes, tra_tolls)

intervals_train = (
    ('06:00:00', '08:00:00'),
    ('15:00:00', '17:00:00')
)

intervals_predict = (
    ('08:00:00', '10:00:00'),
    ('17:00:00', '19:00:00')
)

short_dates = ('2016-09-20', '2016-09-26')
long_dates = ('2016-09-19', '2016-10-10')
valid_dates = ('2016-10-11', '2016-10-17')
test1_dates = ('2016-10-18', '2016-10-24')

VERBOSE = False


# In[ ]:

def parser_date(strs):
    ans = []
    for s in strs:
        t = datetime.datetime.strptime(s, '%Y-%m-%d')
        minute = int(math.floor(t.minute / 20) * 20)
        t = datetime.datetime(t.year, t.month, t.day, t.hour, minute, 0)
        ans.append(t)
    return ans

def parser_datetime(strs):
    ans = []
    for s in strs:
        t = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        minute = int(math.floor(t.minute / 20) * 20)
        t = datetime.datetime(t.year, t.month, t.day, t.hour, minute, 0)
        ans.append(t)
    return ans

# split datetime to date and time
def split_datetime(df):
    df_date = df['datetime'].apply(lambda x: x.date()).to_frame()
    df_time = df['datetime'].apply(lambda x: x.time()).to_frame()
    df_date.rename(index=str, columns={'datetime': 'date'}, inplace=True)
    df_time.rename(index=str, columns={'datetime': 'time'}, inplace=True)
    df = pd.concat([df_date, df_time, df], axis=1)
    df.drop('datetime', axis=1, inplace=True)
    df_date = None
    df_time = None
    return df

# Cut the specified dates in [begin, end]
def cut_date(df, begin, end):
    date_begin = datetime.datetime.strptime(begin, '%Y-%m-%d').date()
    date_end = datetime.datetime.strptime(end, '%Y-%m-%d').date()
    mask = (df['date'] >= date_begin) & (df['date'] <= date_end)
    return df[mask]

# Cut the specified time in [begin, end)
def cut_time(df, interval):
    time_begin = datetime.datetime.strptime(interval[0], '%H:%M:%S').time()
    time_end = datetime.datetime.strptime(interval[1], '%H:%M:%S').time()
    mask = (df['time'] >= time_begin) & (df['time'] < time_end)
    return df[mask]

def onehot(n, i):
    x = np.zeros(n)
    x[i] = 1
    return x

def concat(axis=0):
    return lambda x, y: np.concatenate([x, y], axis=axis)

def missing_idx(df, dates, ampm, intervals, name='', verbose=False):
    timedates = [(df.values[i][0], df.values[i][1]) for i in range(df.values.shape[0])]
    cnt = 0
    miss = []
    for date in pd.date_range(*dates):
        idx = 0 if ampm == 'am' else 1
        start = datetime.datetime.strptime(intervals[idx][0], '%H:%M:%S')
        end = datetime.datetime.strptime(intervals[idx][1], '%H:%M:%S')
        while start < end:
            if (date.date(), start.time()) not in timedates:
                miss.append(cnt)
                if verbose:
                    print 'Warning: %s missing' % name, date.date(), start.time()
            start += datetime.timedelta(minutes=20)
            cnt += 1
    if verbose:
        print '%s totally %d missing values' % (name, len(miss))
    return miss

def my_mape(pred, label, return_total=False):
    mape = 0.0
    total = 0
    
    for p, l in zip(pred, label):
        if l != 0:
            mape += abs((p-l) / l)
            total += 1
    
    mape /= total
    return (mape, total) if return_total else mape    


# In[ ]:

class Features:
    def __init__(self, pathname, filename_wea, filename_vol, filename_tra):
        self.df_wea = None
        self.df_vol = None
        self.pathname = pathname
        self.filename_wea = filename_wea
        self.filename_vol = filename_vol
        self.filename_tra = filename_tra
        print 'Reading files...'
        self.read_all()
        print 'Finish reading files.'
        
    def read_wea(self):
        self.df_wea = pd.read_csv(self.pathname + self.filename_wea, parse_dates=[0], date_parser=parser_date)
    
    def read_vol(self):
        self.df_vol = pd.read_csv(self.pathname + self.filename_vol, parse_dates=[0], date_parser=parser_datetime)

        mean = self.df_vol['vehicle_type'].mean()
        self.df_vol['vehicle_type'].fillna(mean, inplace=True)

        self.df_vol.rename(index=str, columns={'time': 'datetime'}, inplace=True)
        self.df_vol.sort_values(by=['datetime'], inplace=True)
        
        self.df_vol = split_datetime(self.df_vol)
    
    def read_tra(self):
        self.df_tra = pd.read_csv(self.pathname + self.filename_tra, parse_dates=[3], date_parser=parser_datetime)
        
        self.df_tra.rename(index=str, columns={'starting_time': 'datetime'}, inplace=True)
        self.df_tra.drop(['vehicle_id', 'travel_seq'], axis=1, inplace=True)
        self.df_tra = split_datetime(self.df_tra)
        self.df_tra.sort_values(by=['date', 'time', 'intersection_id', 'tollgate_id'], inplace=True)

    def read_all(self):
        self.read_wea()
        self.read_vol()
        self.read_tra()
    
    def get_wea(self, dates, ampm):
        if not isinstance(dates, list) and not isinstance(dates, tuple):
            dates = (dates, dates)
        
        df = cut_date(self.df_wea, *dates)
        hour = 6 if ampm == 'am' else 15
        df = df[df['hour'] == hour]

        timedates = [[df.values[i][0], df.values[i][1]] for i in range(df.values.shape[0])]
        cnt = 0
        miss = []
        for date in pd.date_range(*dates):
            idx = 0 if ampm == 'am' else 1
            if [date, hour] not in timedates:
                miss.append(cnt)
                if VERBOSE:
                    print 'Warning: %s missing' % 'wea', date.date(), hour
            cnt += 1
        if VERBOSE:
            print '%s totally %d missing values' % ('wea', len(miss))

        df.drop(['date', 'hour'], axis=1, inplace=True)
        mean = df.mean().to_frame().transpose()
        w = df.values
        for i in miss:
            w = np.insert(w, i, mean, axis=0)
        
        # shape: (number of days, 7)
        
        df = None
        return w

    def get_vol(self, dates, ampm, toll, dire, intervals):
        if not isinstance(dates, list) and not isinstance(dates, tuple):
            dates = (dates, dates)
        
        df = cut_date(self.df_vol, *dates)
        idx = 0 if ampm == 'am' else 1
        df = cut_time(df, intervals[idx])
        mask = (df['tollgate_id'] == toll) & (df['direction'] == dire)
        return df[mask]
    
    def get_vol_X_tolldire(self, dates, ampm, toll, dire, normalize=True):
        if not isinstance(dates, list) and not isinstance(dates, tuple):
            dates = (dates, dates)
        
        def one_tolldire(_toll, _dire):
            df = self.get_vol(dates, ampm, _toll, _dire, intervals_train)
            group = df.groupby(['date', 'time', 'tollgate_id', 'direction'])
            df = group.agg([np.sum, np.mean]).reset_index()
            df.fillna(0, inplace=True)
            car_info = df[['vehicle_model', 'has_etc', 'vehicle_type']].values

            miss = missing_idx(df=df, dates=dates, ampm=ampm, intervals=intervals_train, name='vol', verbose=VERBOSE)
            for i in miss:
                # zero may be bad
                car_info = np.insert(car_info, i, 0, axis=0)
        
            shape = car_info.shape
            car_info = car_info.reshape([shape[0] / 6, shape[1] * 6])
            if normalize:
                min_max_scaler = preprocessing.MinMaxScaler()
                car_info = min_max_scaler.fit_transform(car_info)
            return car_info

        car_info = reduce(
            concat(axis=1),
            [one_tolldire(toll, dire) for toll, dire in toll_dire]
        )
        
        weekday = np.array([onehot(7, date.weekday()) for date in pd.date_range(*dates)])
        weather = self.get_wea(dates=dates, ampm=ampm)
        onehot_tolldire = np.tile(onehot(len(toll_dire), toll_dire.index((toll, dire))), (weekday.shape[0], 1))
        X = np.concatenate([weekday, weather, car_info, onehot_tolldire], axis=1)
        I6 = np.eye(6)
        one6 = np.ones(6)
        X = reduce(
            concat(axis=0),
            np.array([np.concatenate([np.outer(one6, X[i]), I6], axis=1) for i in range(X.shape[0])])
        )
        
        # shape: (number of days * 6, 205)
        # 6 = predicting windows per 2 hours
        # 205 = 7(weekday onehot)
        #     + 7(weather)
        #     + 3(model, etc, type)*2(sum, mean)*6(windows per 2 hours)*5(all tolldire pairs)
        #     + 5(tolldire onehot)
        #     + 6(window onehot)
        
        df = None
        return X
    
    def get_vol_X(self, dates, ampm, normalize=True):
        return reduce(
            concat(axis=0),
            [self.get_vol_X_tolldire(dates=dates, ampm=ampm, toll=toll, dire=dire, normalize=normalize) for toll, dire in toll_dire]
        )
    
    def get_vol_y(self, dates, ampm, toll, dire, normalize=True):
        df = self.get_vol(dates, ampm, toll, dire, intervals_predict)
        group = df.groupby(['date', 'time', 'tollgate_id', 'direction'])
        df = group.count().reset_index()
        y = df['vehicle_model'].values

        miss = missing_idx(df=df, dates=dates, ampm=ampm, intervals=intervals_predict, name='vol', verbose=VERBOSE)
        for i in miss:
            # zero may be bad
            y = np.insert(y, i, 0, axis=0)
        
        df = None
        return y
    
    def get_vol_Xy(self, dates, ampm, normalize=True):
        X = self.get_vol_X(dates, ampm, normalize=normalize)
        y = reduce(
            concat(axis=0),
            [self.get_vol_y(dates=dates, ampm=ampm, toll=toll, dire=dire, normalize=normalize) for toll, dire in toll_dire]
        )
        
        return X, y
    
    
    def get_tra(self, dates, ampm, inte, toll, intervals):
        if not isinstance(dates, list) and not isinstance(dates, tuple):
            dates = (dates, dates)
        
        df = cut_date(self.df_tra, dates[0], dates[1])
        idx = 0 if ampm == 'am' else 1
        df = cut_time(df, intervals[idx])
        mask = (df['intersection_id'] == inte) & (df['tollgate_id'] == toll)
        return df[mask]
    
    def get_tra_X_intetoll(self, dates, ampm, inte, toll, normalize=True):
        if not isinstance(dates, list) and not isinstance(dates, tuple):
            dates = (dates, dates)

        def one_intetoll(_inte, _toll):
            df = self.get_tra(dates, ampm, _inte, _toll, intervals_train)

            group = df.groupby(['date', 'time', 'intersection_id', 'tollgate_id'])
            df = group.agg([np.sum, np.mean]).reset_index()
            df.fillna(0, inplace=True)
            car_info = df['travel_time'].values

            miss = missing_idx(df=df, dates=dates, ampm=ampm, intervals=intervals_train, name='tra', verbose=VERBOSE)
            for i in miss:
                # zero may be bad
                car_info = np.insert(car_info, i, 0, axis=0)
            
            shape = car_info.shape
            car_info = car_info.reshape([shape[0] / 6, shape[1] * 6])
            if normalize:
                min_max_scaler = preprocessing.MinMaxScaler()
                car_info = min_max_scaler.fit_transform(car_info)
            return car_info

        car_info = reduce(
            concat(axis=1),
            [one_intetoll(inte, toll) for inte, toll in inte_toll]
        )

        weekday = np.array([onehot(7, date.weekday()) for date in pd.date_range(*dates)])
        weather = self.get_wea(dates=dates, ampm=ampm)
        onehot_intetoll = np.tile(onehot(len(inte_toll), inte_toll.index((inte, toll))), (weekday.shape[0], 1))
        X = np.concatenate([weekday, weather, car_info, onehot_intetoll], axis=1)
        I6 = np.eye(6)
        one6 = np.ones(6)
        X = reduce(
            concat(axis=0),
            np.array([np.concatenate([np.outer(one6, X[i]), I6], axis=1) for i in range(X.shape[0])])
        )
        
        # shape: (number of days * 6, 98)
        # 6 = predicting windows per 2 hours
        # 98 = 7(weekday onehot)
        #    + 7(weather)
        #    + 1(travel_time)*2(sum, mean)*6(windows per 2 hours)*6(all intetoll pairs)
        #    + 6(intetoll onehot)
        #    + 6(window onehot)  
        
        df = None
        return X

    def get_tra_X(self, dates, ampm, normalize=True):
        return reduce(
            concat(axis=0),
            [self.get_tra_X_intetoll(dates=dates, ampm=ampm, inte=inte, toll=toll, normalize=normalize) for inte, toll in inte_toll]
        )
    
    def get_tra_y(self, dates, ampm, inte, toll, normalize=True):
        df = self.get_tra(dates, ampm, inte, toll, intervals_predict)
        group = df.groupby(['date', 'time', 'intersection_id', 'tollgate_id'])
        df = group.agg(np.mean).reset_index()
        y = df['travel_time'].values

        miss = missing_idx(df=df, dates=dates, ampm=ampm, intervals=intervals_predict, name='tra', verbose=VERBOSE)
        for i in miss:
            # zero may be bad
            y = np.insert(y, i, 0, axis=0)
        
        df = None
        return y
    
    def get_tra_Xy(self, dates, ampm, normalize=True):
        X = self.get_tra_X(dates, ampm, normalize=normalize)
        y = reduce(
            concat(axis=0),
            [self.get_tra_y(dates=dates, ampm=ampm, inte=inte, toll=toll, normalize=normalize) for inte, toll in inte_toll]
        )
        
        return X, y


# In[ ]:

if __name__ == '__main__':
    

# Training example:
    
    feat = Features(
        '../dataSets/training/',
        'weather (table 7)_training.csv',
        'volume(table 6)_training.csv',
        'trajectories(table 5)_training.csv'
    )

#     X, y = feat.get_vol_Xy(dates=long_dates, ampm='am')
#     print X.shape, y.shape

#     X_valid, y_valid = feat.get_vol_Xy(dates=valid_dates, ampm='am')
#     print X_valid.shape, y_valid.shape

    X, y = feat.get_tra_Xy(dates=long_dates, ampm='am')
    print X.shape, y.shape

    X_valid, y_valid = feat.get_tra_Xy(dates=valid_dates, ampm='am')
    print X_valid.shape, y_valid.shape

