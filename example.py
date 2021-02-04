from causalPartition import causalPartition
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# load and process data
data = pd.read_csv('data_ws.csv')
probabilities = np.load('probabilities_ws.npy')

new_probabilities = {}

new_probabilities['bbb_2_normalized'] =  1.0 * probabilities[4]/(probabilities[4]+probabilities[3]+probabilities[2])
new_probabilities['bbb_1_normalized'] =  1.0 * probabilities[3]/(probabilities[4]+probabilities[3]+probabilities[2])
new_probabilities['bbb_0_normalized'] =  1.0 * probabilities[2]/(probabilities[4]+probabilities[3]+probabilities[2])

new_probabilities['bbn_2_normalized'] =  1.0 * probabilities[7]/(probabilities[7]+probabilities[6]+probabilities[5])
new_probabilities['bbn_1_normalized'] =  1.0 * probabilities[6]/(probabilities[7]+probabilities[6]+probabilities[5])
new_probabilities['bbn_0_normalized'] =  1.0 * probabilities[5]/(probabilities[7]+probabilities[6]+probabilities[5])

new_probabilities['bb_1_normalized'] =  1.0 * probabilities[1]/(probabilities[1]+probabilities[0])
new_probabilities['bb_0_normalized'] =  1.0 * probabilities[0]/(probabilities[1]+probabilities[0])

new_probabilities['open_square_0_normalized'] =  1.0 * probabilities[8]/(probabilities[8]+probabilities[9]+probabilities[10]+probabilities[11])
new_probabilities['open_square_1_normalized'] =  1.0 * probabilities[9]/(probabilities[8]+probabilities[9]+probabilities[10]+probabilities[11])
new_probabilities['open_square_2_normalized'] =  1.0 * probabilities[10]/(probabilities[8]+probabilities[9]+probabilities[10]+probabilities[11])
new_probabilities['open_square_3_normalized'] =  1.0 * probabilities[11]/(probabilities[8]+probabilities[9]+probabilities[10]+probabilities[11])

new_probabilities['assignment']  = probabilities[-1]
    
# to satisfy positivity 
idx = np.logical_and(np.logical_and(data['bbb_0'] + data['bbb_1'] + data['bbb_2'] > 0,
                     data['bbn_0'] + data['bbn_1'] + data['bbn_2'] > 0),
                     data['open_square_0']+data['open_square_1']+data['open_square_2']+data['open_square_3'] > 0)
data_ = data[idx]
probabilities_ = {}
input_features = ['assignment', 
                  'bb_1_normalized', 'bb_0_normalized',
                   'bbb_0_normalized', 'bbb_1_normalized', 'bbb_2_normalized', 
                   'bbn_0_normalized', 'bbn_1_normalized', 'bbn_2_normalized',
                   'open_square_0_normalized', 'open_square_1_normalized', 'open_square_2_normalized', 
                  'open_square_3_normalized'
                 ]
    
for key in ['assignment']+input_features:
        probabilities_[key] = new_probabilities[key][idx]

# train the tree (separate=True means we treat the assignment variable as a dimension); please revise the parameters
outcome = 'y2'


partition = causalPartition(data_, probabilities_, 'assignment')
train_result_separate = partition.split_exposure_hajek(True, outcome, input_features, 
                                                       max_attempt=10, eps=0.001, 
                                                       delta=0.01, 
                                                       criteria={'non_trivial_reduction': 0,
                                                                 'min_leaf_size': 4000})
partition.plot_tree(train_result_separate)

est_result_separate = partition.estimate_exposure_hajek(train_result_separate, 
                                input_features, outcome, eps=0.001, separate=True)

partition.plot_tree(est_result_separate)


# train the tree (separate=False means we examine heterogeneous indirect effects); please revise the parameters
outcome = 'y2'
input_features = [
                   # 'assignment', 
                   'bb_1_normalized', 'bb_0_normalized',
                   'bbb_0_normalized', 'bbb_1_normalized', 'bbb_2_normalized', 
                   'bbn_0_normalized', 'bbn_1_normalized', 'bbn_2_normalized',
                   'open_square_0_normalized', 'open_square_1_normalized', 'open_square_2_normalized', 'open_square_3_normalized'
                 ]
  
partition = causalPartition(data_, probabilities_, 'assignment')
train_result_nonseparate = partition.split_exposure_hajek(False, outcome, input_features, 
                                                       max_attempt=10, eps=0.001, 
                                                       delta=0.01, 
                                                       criteria={'non_trivial_reduction': 0,
                                                                 'min_leaf_size': 4000})
 
partition.plot_tree(train_result_nonseparate)

est_result_separate = partition.estimate_exposure_hajek(train_result_nonseparate, 
                                input_features, outcome, eps=0.01, separate=False)

partition.plot_tree(est_result_separate)
