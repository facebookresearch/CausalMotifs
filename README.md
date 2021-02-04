

## CausalMotifs

This repository provides a reference implementation of  _CausalMotifs_  as described in the [paper](https://arxiv.org/abs/2010.09911):  
> Causal Network Motifs: Identifying Heterogeneous Spillover Effects in A/B Tests.  
> Yuan Yuan, Kristen M. Altenburger, and Farshad Kooti.  
> WWW'2021.


### Documentation 
This repository contains code to replicate the simulation results in the paper. We do not include data for replicating internal FB results but note that the code architecture is the same as the simulation.


### Directions

#### Environments

Please use Python 3

See _requirements.txt_ for prerequisite Python packages.

#### Generating network

- _generate_WS.py_ generates the Watts-Strogatz simulation network discussed in the paper. It outputs two files:

- _data_ws.csv_: a table file that stores the interference vector, covariates, and outcome.

- _probabilities_ws.npy_: a numpy file that stores the bootstrapping results. It is a  [num_replicates * num_observations * dimension_interference_vector] tensor.

#### Training

_causalPartition.py_ contains the class of the main algorithm
_example.py_ illustrates the usage. It has the following steps

- **Data loading and cleaning**. It loads the two files generated before, and then clean the data to satisfy positivity requirement

- **Training**. Training the data to generate the partitions (exposure conditions).

1. Create a _causalPartition_ class
``
partition = causalPartition(data_, probabilities_, 'assignment')
``
1. Train and split the space for the interference vector
``
train_result_nonseparate = partition.split_exposure_hajek(True, outcome, input_features, max_attempt=10, eps=0.01, delta=0.001, criteria={'non_trivial_reduction': 0, 'reasonable_propensity': 0.05})
``
1. Plot the training tree
``
partition.plot_tree(train_result_separate)
``
1. Sample spliting and plot the estimation tree
``
est_result_separate = partition.estimate_exposure_hajek(train_result_separate, input_features, outcome, eps=0.01, separate=True)
``
``
partition.plot_tree(est_result_separate)
``

We implemented three different splitting criteria (as the parameter _criteria_). This is a _dict_, and please use the key to specify the splitting criterion and the value for the parameter.

- 'non_trivial_reduction': WSSE should be reduced by at least the given parameter ($\gamma$).

- 'separate_reduction':  The SSE of a parent node should be greater than both children notes. No paramter needs to be input.

- 'reasonable_propensity': As discussed in the paper footnote (an empirically effective alternative). Needs to input the $\varepsilon$.

- 'min_leaf_size': Leaf size should be greater than the given parameter ($\kappa$)

### LICENSE
CausalMotifs is MIT Licensed, as found in the LICENSE file.

### Authors
* Yuan Yuan, yuan2@mit.edu
* Kristen M. Altenburger, kaltenburger@fb.com
* Farshad Kooti, farshadkt@fb.com

### Citing
If you find _CausalMotifs_ useful for your research, please consider citing the following paper


```
@inproceedings{causalmotifs-www2021,
author = {Yuan, Yuan and Altenburger, Kristen M. and Kooti, Farshad},
 title = {Causal Network Motifs: Identifying Heterogeneous Spillover Effects in A/B Tests},
 booktitle = {Proceedings of the 2021 World Wide Web Conference},
 year = {2021}
}
```
