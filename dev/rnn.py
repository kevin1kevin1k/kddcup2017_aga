import pandas as pd
import math
import sys
from src import functions as func
from src import feagen as feag
import datetime
import numpy as np
from datetime import datetime,timedelta
from keras.models import Sequential
from keras.layers import Input, Dense,LSTM,Activation
from keras.layers.core import Flatten,Dropout
from sklearn import preprocessing
import keras

def create_dataset(dataset,lookback = 1):
    dataX,dataY = [],[]
    for day in dataset:
        for i in range(len(day)-look_back):
            a = day[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append([day[i + look_back, 0]])
    return np.array(dataX), np.array(dataY)
 

def my_mape(label,pred):
    c = 0.0
    mae = 0.0
    for (l,p) in zip(label.flatten(),pred.flatten()):
        if l != 0:
            mae += abs(l-p)/l
            c += 1
    return mae/c,c

def eval_model(valX,valY,scaler,model):
    predY = []
    for day in range(7):
        val = np.reshape(valX[day*6,:,:],(1,1,6))
        for win in range(6):
            pred = model.predict(val)
            val = np.append(val[0,0,1:],pred)
            val = np.reshape(val,(1,1,6))
            predY.append(pred)
    predY = np.array(predY).reshape(1, -1)
    predY = scaler.inverse_transform(predY)
    raw_valY = scaler.inverse_transform(valY)
    
    return my_mape(raw_valY,predY)

def record(model,scaler,task,target,test,training_weeks,windows,use_sample_weight):
    #make prediction file
    with open('./rnn_record/'+target+'_'+task+'_'+str(training_weeks)+'w_'+use_sample_weight+'.csv','w') as record:
        record.write('\"intersection_id\",\"tollgate_id\",\"time_window\",\"avg_travel_time\"\n')
        flag = 0
        for day in range(7):
            ttX = np.array(test[day,:,:])
            ttX = np.reshape(ttX,(1,1,6))
            for i in range(6):
                record.write(target.split('-')[0]+',')
                record.write(target.split('-')[1]+',')
                record.write(windows[flag]+',')
                flag +=1
                #predict
                pred = model.predict(ttX)
                ttX = np.append(ttX[0,0,1:],pred)
                ttX = np.reshape(ttX,(1,1,6))
                record.write(str(scaler.inverse_transform(pred).flatten()[0])+'\n')

week2date = {}
week2date[3] = '2016-09-20'
week2date[4] = '2016-09-13'
week2date[5] = '2016-09-06'
week2date[6] = '2016-08-30'
week2date[7] = '2016-08-23'
win_6 = (6-6)*3
win_8 = (8-6)*3
win_10 = (10-6)*3
win_15 = (15-6)*3
win_17 = (17-6)*3
win_19 = (19-6)*3

if len(sys.argv) != 5:
    sys.exit("Usage : python rnn.py <task> <target> <training weeks> <use sample weight>")
else:
    task = sys.argv[1]
    target = sys.argv[2]
    training_weeks = int(sys.argv[3])
    use_sample_weight = sys.argv[4]
    total_weeks = training_weeks + 2
    testing_weeks = training_weeks + 1
    num_instance = total_weeks*7


if task == 'am':
    window_start = datetime.strptime('2016-10-18 08:00:00', "%Y-%m-%d %H:%M:%S")
    x_range = range(win_6,win_8)
    y_range = range(win_8,win_10)
    xy_range = range(win_6,win_10)
elif task == 'pm':
    x_range = range(win_15,win_17)
    y_range = range(win_17,win_19)
    xy_range = range(win_15,win_19)
    window_start = datetime.strptime('2016-10-18 17:00:00', "%Y-%m-%d %H:%M:%S")
else:
    sys.exit("Usage : Only am or pm for <task>")
    
windows = []
for d in range(7):
    for t in range(6):
        windows.append('\"['+str(window_start +timedelta(minutes=20*t))+','+ str(window_start +timedelta(minutes=20*(t+1)))+')\"')
    window_start  += timedelta(days=1)

#Read training data
path = '../dataSets/training/'
file_suffix = '.csv'
in_file = 'trajectories(table 5)_training'
travel_times = func.read_file_to_travel_times(path, in_file, file_suffix)


#Read testing data
path = '../dataSets/testing_phase1/'
file_suffix = '.csv'
in_file = 'trajectories(table 5)_test1'
testing_times = func.read_file_to_travel_times(path, in_file, file_suffix)

#merge them into one dict
routes =  travel_times.keys()
for r in routes:
    travel_times[r].update(testing_times[r])

num_window = 13*3 # 6 ~ 19
routes = travel_times.keys()
print routes
window_size = timedelta(minutes=20)
window_start = datetime.strptime(week2date[training_weeks]+' 06:00:00', "%Y-%m-%d %H:%M:%S")


cube_2d = np.zeros((num_instance*num_window))

d=0
while d < num_instance:
    window_scanner = window_start
    for t in range(num_window):
        try:
            cube_2d[d*39+t] = np.mean(travel_times[target][window_scanner])
        except:
            cube_2d[d*39+t] = 0
        window_scanner += window_size
    window_start += timedelta(days=1)
    d += 1
# Normalize
cube_2d = cube_2d.reshape(-1, 1)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
normed = scaler.fit_transform(cube_2d)
#normed = preprocessing.normalize(cube_2d,axis=0)

cube_3d = np.zeros((num_instance,num_window,1))
for d in range(num_instance):
    for t in range(num_window):
        cube_3d[d][t][0] = normed[d*39+t]

train = cube_3d[:training_weeks*7,xy_range,:]
valid = cube_3d[training_weeks*7:testing_weeks*7,xy_range,:]

test = cube_3d[testing_weeks*7:,x_range,:]


look_back = 6

trX, trY = create_dataset(train, look_back)
valX,valY = create_dataset(valid,look_back)

if use_sample_weight == 'sw':
    sample_w = np.zeros(trY.shape[0])
    for i in range(len(trY)):
        sample_w[i] = 0 if trY[i,0] == 0 else 1/trY[i,0]



trX = np.reshape(trX, (trX.shape[0], 1, trX.shape[1]))
valX = np.reshape(valX, (valX.shape[0], 1, valX.shape[1]))

print 'Train data:',trX.shape,trY.shape
print 'Valid data:',valX.shape,valY.shape
print 'Test data:',test.shape
print 'Valid non-zero',np.count_nonzero(valY.flatten())
print 'Test non-zero:',np.count_nonzero(test)

epoch = 300
diplay = 1
step = epoch/diplay
best = 0.3
best_c = 0
best_dim = 0
best_epoch = 0

for LSTM_out in range(4,21):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(LSTM_out, input_shape=(1, look_back)))
    model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(loss='mae', optimizer=adam)
    print "Doing route ",target,' ',task,' LSTM_out ',LSTM_out
    for i in range(step):
        if use_sample_weight == 'sw':
            model.fit(trX, trY, epochs=diplay, verbose=0,sample_weight=sample_w)
        else:
            model.fit(trX, trY, epochs=diplay, verbose=0)
        mape,count = eval_model(valX,valY,scaler,model)
        #print 'Step ',(i+1)*diplay,' valid score : ',mape
        if best > mape:
            best = mape
            best_c = count
            best_dim = LSTM_out
            best_epoch = (i+1)*diplay
            record(model,scaler,task,target,test,training_weeks,windows,use_sample_weight)
    print 'Best mape = ',best

with open('./rnn_record/'+task+'_record.txt','a') as record:
    record.write(target+' '+str(training_weeks)+'weeks best:'+str(best)+'('+str(best_c)+') ')
    record.write('epoch:'+str(best_epoch)+' dim:'+str(best_dim)+'\n')



