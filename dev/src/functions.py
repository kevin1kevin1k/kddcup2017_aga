from datetime import datetime,timedelta
from keras.models import Sequential
from keras.layers import Dense
import math
import pandas as pd
import numpy as np


def read_file_to_volumes(path, in_file, file_suffix):

    in_file_name = in_file + file_suffix

    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    vol_data = fr.readlines()
    fr.close()

    # Create a dictionary to caculate and store volume per time window
    volumes = {}  # key: time window value: dictionary
    for i in range(len(vol_data)):
        each_pass = vol_data[i].replace('"', '').split(',')
        tollgate_id = each_pass[1]
        direction = each_pass[2]

        pass_time = each_pass[0]
        pass_time = datetime.strptime(pass_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = int(math.floor(pass_time.minute / 20) * 20)
        start_time_window = datetime(pass_time.year, pass_time.month, pass_time.day,
                                     pass_time.hour, time_window_minute, 0)

        if start_time_window not in volumes:
            volumes[start_time_window] = {}
        if tollgate_id not in volumes[start_time_window]:
            volumes[start_time_window][tollgate_id] = {}
        if direction not in volumes[start_time_window][tollgate_id]:
            volumes[start_time_window][tollgate_id][direction] = 1
        else:
            volumes[start_time_window][tollgate_id][direction] += 1

    return volumes
    

def read_file_to_travel_times(path, in_file, file_suffix):

    in_file_name = in_file + file_suffix

    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()

    # Create a dictionary to store travel time for each route per time window
    travel_times = {}  # key: route_id. Value is also a dictionary of which key is the start time for the time window and value is a list of travel times
    for i in range(len(traj_data)):
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]

        route_id = intersection_id + '-' + tollgate_id
        if route_id not in travel_times.keys():
            travel_times[route_id] = {}

        trace_start_time = each_traj[3]
        trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = int(math.floor(trace_start_time.minute / 20) * 20)
        start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                     trace_start_time.hour, time_window_minute, 0)
        tt = float(each_traj[-1]) # travel time

        if start_time_window not in travel_times[route_id].keys():
            travel_times[route_id][start_time_window] = [tt]
        else:
            travel_times[route_id][start_time_window].append(tt)

    return travel_times


def read_weather(path, in_file, file_suffix):

    in_file_name = in_file + file_suffix

    # Load volume data
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    vol_data = fr.readlines()
    fr.close()

    # Create a dictionary to caculate and store volume per time window
    weathers = {}  # key: time window value: dictionary
    for i in range(len(vol_data)):
        each_pass = vol_data[i].replace('"', '').split(',')
        each_pass[2:] = map(float, each_pass[2:])
        date, hour, pressure, sea_pressure, wind_direction, wind_speed, temperature, rel_humidity, precipitation = each_pass
        date = datetime.strptime(date + ' ' + hour, "%Y-%m-%d %H")
        weathers[date] = (pressure, sea_pressure, wind_direction, wind_speed, temperature, rel_humidity, precipitation)

    return weathers

def read_links(path, in_file, file_suffix):

    in_file_name = path+in_file + file_suffix
    links = pd.read_csv(in_file_name,sep=',')
    links = links.drop('lane_width',1)
    links = links.as_matrix()
    l = {}
    for line in links:
        l[line[0]] = {}
        l[line[0]]['length'] = line[1]
        l[line[0]]['width'] = line[2] 
        l[line[0]]['in_top'] = [] if type(line[4])==np.float else map(int,line[4].split(','))
        l[line[0]]['out_top'] = [] if type(line[5])==np.float else map(int,line[5].split(','))
    return l

def read_routes(path, in_file, file_suffix):

    in_file_name = path+in_file + file_suffix
    routes = pd.read_csv(in_file_name,sep=',')
    routes = routes.as_matrix()
    r = {}
    for line in routes:
        r[line[0]+'-'+str(line[1])] = map(int,line[2].split(','))
    return r


def print_volumes(volumes, in_file, file_suffix):
    out_suffix = '_20min_avg_volume'
    out_file_name = in_file.split('_')[1] + out_suffix + file_suffix

    # Format output for tollgate and direction per time window
    fw = open(out_file_name, 'w')
    fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')
    time_windows = list(volumes.keys())
    time_windows.sort()
    for time_window_start in time_windows:
        time_window_end = time_window_start + timedelta(minutes=20)
        for tollgate_id in volumes[time_window_start]:
            for direction in volumes[time_window_start][tollgate_id]:
               out_line = ','.join(['"' + str(tollgate_id) + '"', 
			                     '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                                 '"' + str(direction) + '"',
                                 '"' + str(volumes[time_window_start][tollgate_id][direction]) + '"',
                               ]) + '\n'
               fw.writelines(out_line)
    fw.close()


def print_travel_times(travel_times, in_file, file_suffix):
    out_suffix = '_20min_avg_travel_time'
    out_file_name = in_file.split('_')[1] + out_suffix + file_suffix

    # Calculate average travel time for each route per time window
    fw = open(out_file_name, 'w')
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
    for route in travel_times.keys():
        route_time_windows = list(travel_times[route].keys())
        route_time_windows.sort()
        for time_window_start in route_time_windows:
            time_window_end = time_window_start + timedelta(minutes=20)
            tt_set = travel_times[route][time_window_start]
            avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)
            out_line = ','.join(['"' + route.split('-')[0] + '"', '"' + route.split('-')[1] + '"',
                                 '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                                 '"' + str(avg_tt) + '"']) + '\n'
            fw.writelines(out_line)
    fw.close()

#Example usage: gen_model(10,2, [50,25],[None,'relu'])
#will return a NN [10]->[50]-relu->[25]-linear->[2]
def gen_model(in_dim, hide_layer_num_node, hide_layer_dim, out_dim, act_func):
    model = Sequential()
    
    #input layer
    model.add(Dense(hide_layer_dim[0], input_dim=in_dim, init='normal', activation=act_func[0]))
    #hidden layer
    for i in range(1,len(hide_layer_dim)):
        model.add(Dense(hide_layer_dim[i], init='normal', activation=act_func[i]))
    #output layer
    model.add(Dense(out_dim, init='normal', activation='linear'))

    return model


def main():
    print 'This is not for executing.'
    #print read_links('../../','dataSets/training/links (table 3)','.csv')
    #print read_routes('../../','dataSets/training/routes (table 4)','.csv')
    # in_file = 'volume(table 6)_training'
    # volumes = read_file_to_volumes(in_file)
    # print_volumes(volumes, in_file)

if __name__ == '__main__':
    main()