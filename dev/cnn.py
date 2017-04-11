from src import functions as func
from src import feagen as feag
import sys
import datetime
import numpy as np
from datetime import datetime,timedelta
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense,Activation
from keras.layers.core import Flatten,Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import preprocessing
import keras

week2date = {}
week2date[3] = '2016-09-20'
week2date[4] = '2016-09-13'
week2date[5] = '2016-09-06'
week2date[6] = '2016-08-30'
week2date[7] = '2016-08-23'
inss = ['B','B','A','A','C','C']
tols = ['3','1','3','2','3','1']

if len(sys.argv) != 4:
    sys.exit("Usage : python cnn.py <task> <training weeks> <10/1~7? y/n>")
else:
    task = sys.argv[1]
    training_weeks = int(sys.argv[2])
    with_vacation = sys.argv[3]
    if with_vacation == 'n':
        training_weeks -= 1
    elif with_vacation != 'y':
        sys.exit("Usage : Only y/n for <include 10/1~7?>")

    total_weeks = training_weeks + 2
    testing_weeks = training_weeks + 1
    num_instance = total_weeks*7
    



if task == 'am':
    window_start = datetime.strptime('2016-10-18 08:00:00', "%Y-%m-%d %H:%M:%S")
elif task == 'pm':
    window_start = datetime.strptime('2016-10-18 17:00:00', "%Y-%m-%d %H:%M:%S")
else:
    sys.exit("Usage : Only am or pm for <task>")

windows = []
for d in range(7):
    for t in range(6):
        windows.append('\"['+str(window_start +timedelta(minutes=20*t))+','+ str(window_start +timedelta(minutes=20*(t+1)))+')\"')
    window_start  += timedelta(days=1)




def my_mape(label,pred):
    c = 0.0
    mae = 0.0
    for (l,p) in zip(label.flatten(),pred.flatten()):
        if l != 0:
            mae += abs(l-p)/l
            c += 1
    return mae/c

def record(best,conv_out_dim,hidden_dim,step,task,training_weeks,predY,windows,with_vacation):
    #record mape
    with open('./cnn_record/'+str(training_weeks)+'w_'+with_vacation+'1001_'+task+'.txt','w') as record:
        record.write('Mape :'+str(best)+'\n')
        record.write('Epoch:'+str(step)+'\n')
        record.write('filter out :'+str(conv_out_dim)+'\n')
        record.write('hidden layer:'+str(hidden_dim)+'\n')
    #make prediction file
    with open('./cnn_record/'+str(training_weeks)+'w_'+with_vacation+'1001_'+task+'.csv','w') as record:
        record.write('\"intersection_id\",\"tollgate_id\",\"time_window\",\"avg_travel_time\"\n')
        predY = predY.flatten()
        flag = 0
        for win in windows:
            for ins,tol in zip(inss,tols):
                record.write(ins+',')
                record.write(tol+',')
                record.write(win+',')
                record.write(str(predY[flag])+'\n')
                flag += 1

win_6 = (6-6)*3
win_8 = (8-6)*3
win_10 = (10-6)*3
win_15 = (15-6)*3
win_17 = (17-6)*3
win_19 = (19-6)*3

if task == 'am':
    x_range = range(win_6,win_8)
    y_range = range(win_8,win_10)
    epoch = 4000
    epoch_display = 50
    step = epoch/epoch_display
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)

else:
    x_range = range(win_15,win_17)
    y_range = range(win_17,win_19)
    epoch = 4000
    epoch_display = 50
    step = epoch/epoch_display
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)

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

num_routes = 6
num_window = 13*3 # 6 ~ 19
routes = travel_times.keys()
window_size = timedelta(minutes=20)
starting = (week2date[training_weeks]+' 06:00:00') if with_vacation == 'y' else (week2date[training_weeks+1]+' 06:00:00')
window_start = datetime.strptime(starting, "%Y-%m-%d %H:%M:%S")

cube_2d = np.zeros((num_instance*num_window,num_routes))

d = 0
while d < num_instance:
    window_scanner = window_start
    for t in range(num_window):
        if with_vacation == 'n' and str(window_start.date()) >= '2016-10-01' and str(window_start.date()) <= '2016-10-07':
            d -= 1
            break
        for r in range(num_routes):
            try:
                cube_2d[d*39+t][r] = np.mean(travel_times[routes[r]][window_scanner])
            except:
                cube_2d[d*39+t][r] = 0
        window_scanner += window_size
    window_start += timedelta(days=1)
    d += 1

# Normalize
normed = preprocessing.normalize(cube_2d,axis=0)

# Make X,Y for CNN
cube_3d = np.zeros((num_instance,num_window,num_routes))
for d in range(num_instance):
    for t in range(num_window):
        for r in range(num_routes):
            cube_3d[d][t][r] = cube_2d[d*39+t][r] if t in y_range else normed[d*39+t][r]

trX = cube_3d[:training_weeks*7,x_range,:]
trY = cube_3d[:training_weeks*7,y_range,:]
valX = cube_3d[training_weeks*7:testing_weeks*7,x_range,:]
valY = cube_3d[training_weeks*7:testing_weeks*7,y_range,:]
ttX = cube_3d[testing_weeks*7:,x_range,:]

print 'Doing task ',task,' with ',training_weeks,' training data.'
print 'Train data:',trX.shape
print 'Valid data:',valX.shape
print 'Test data:',ttX.shape

best = 0.21

for conv_out_dim in range(4,20):
    for hidden_dim in range(4,20):
        print 'Training model ',conv_out_dim,'->',hidden_dim,'...'
        model = Sequential()
        model.add(Conv1D(conv_out_dim, 1,input_shape=(trX.shape[1], num_routes)))
        model.add(Dense(hidden_dim,activation='relu'))
        model.add(Dense(num_routes,activation=None))

        model.compile(loss='mse',optimizer=adam)

        
        for i in range(step):
            model.fit(trX, trY, epochs=epoch_display, verbose=0)
            predY = model.predict(valX)
            mape = my_mape(valY,predY)
            if best > mape :
                best = mape
                predY = model.predict(ttX)
                record(best,conv_out_dim,hidden_dim,epoch_display*(i+1),task,training_weeks,predY,windows,with_vacation)


