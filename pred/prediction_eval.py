import numpy as np
import pandas as pd

def cal_mape(label,pred):
	key = 'avg_travel_time'
	y = label[key].as_matrix()
	x = pred[key].as_matrix()
	return np.sum(np.abs(y-x))/len(y)




ref = './ref/'

si_an_pred = 'test_task1_dataset_001_20170404_182345_GradientBoostingRegressor.csv'
my_pred_seq2one = 'rnn_gru_sampleweight.csv'
my_pred_seq2seq = 'rnn_seq2seq_gru.csv'

si_an = pd.read_csv(ref+si_an_pred)
my_seq2one = pd.read_csv(my_pred_seq2one)
my_seq2seq = pd.read_csv(my_pred_seq2seq)

keys = my_seq2seq.keys()
keys = [k for k in keys]
#print keys

si_an = si_an.sort_values(by=keys)
my_seq2one = my_seq2one.sort_values(by=keys)
my_seq2seq = my_seq2seq.sort_values(by=keys)


print 'si an (0.1801) vs. my seq2seq	:',cal_mape(si_an,my_seq2seq)
print 'my seq2one (0.1810) vs. my seq2seq :',cal_mape(my_seq2one,my_seq2seq)
print 'si an (0.1801) vs. my seq2one (0.1810)	:',cal_mape(si_an,my_seq2one)