import numpy as np
import pandas as pd
import statsmodels.api as sm
import gc
import operator
import networkx as nx
from tqdm import tqdm

G = nx.watts_strogatz_graph(2000000, 10, 0.5)
assignments = np.concatenate([[k]*10 for k in list(np.random.randint(0, 2, 2000000//10))])
sample = np.random.choice(2000000, 100000)

print('genearating the graph')
data = []
for i in tqdm(sample):
    neighbor = len(G[i])
    bb_1 = np.sum([assignments[j] for j in G[i]])
    bb_0 = neighbor - bb_1
    bbb_0 = 0
    bbb_1 = 0
    bbb_2 = 0
    bbn_0 = 0
    bbn_1 = 0
    bbn_2 = 0
    open_square_0 = 0
    open_square_1 = 0
    open_square_2 = 0
    open_square_3 = 0
    for j in G[i]:
        for k in G[i]:
            if k > j:
                if np.abs(j-k) <= 5: # this is a simplistic weight to judge if connected (to speed up )
                    if assignments[j] + assignments[k] == 0:
                        bbb_0 += 1
                    elif assignments[j] + assignments[k] == 1:
                        bbb_1 += 1
                    else:
                        bbb_2 += 1
                else:
                    if assignments[j] + assignments[k] == 0:
                        bbn_0 += 1
                    elif assignments[j] + assignments[k] == 1:
                        bbn_1 += 1
                    else:
                        bbn_2 += 1
                    for l in G[i]:
                        if l > k and np.abs(l-k) > 5 and np.abs(l-j) > 5:
                            if assignments[j] + assignments[k] + assignments[l] == 0:
                                open_square_0 += 1
                            elif assignments[j] + assignments[k] + assignments[l]== 1:
                                open_square_1 += 1
                            elif assignments[j] + assignments[k] + assignments[l]== 2:
                                open_square_2 += 1
                            else:
                                open_square_3 += 1
                            
    data.append([i, assignments[i], neighbor, bb_0, bb_1, bbb_0, bbb_1, bbb_2, bbn_0, bbn_1, bbn_2,
                open_square_0, open_square_1, open_square_2, open_square_3])

data = pd.DataFrame.from_records(data)
data.columns = ['id', 'assignment', 'neighbor', 'bb_0', 'bb_1', 'bbb_0', 'bbb_1', 'bbb_2', 'bbn_0', 'bbn_1', 'bbn_2',
                'open_square_0', 'open_square_1', 'open_square_2', 'open_square_3'
               ]

data['open_square_3_normalized'] = 1.0 * data['open_square_3']/(data['open_square_3']+data['open_square_2']+data['open_square_1']+data['open_square_0'])
data['open_square_2_normalized'] = 1.0 * data['open_square_2']/(data['open_square_3']+data['open_square_2']+data['open_square_1']+data['open_square_0'])
data['open_square_1_normalized'] = 1.0 * data['open_square_1']/(data['open_square_3']+data['open_square_2']+data['open_square_1']+data['open_square_0'])
data['open_square_0_normalized'] = 1.0 * data['open_square_0']/(data['open_square_3']+data['open_square_2']+data['open_square_1']+data['open_square_0'])

data['bbb_2_normalized'] = 1.0 * data['bbb_2']/(data['bbb_2']+data['bbb_1']+data['bbb_0'])
data['bbb_1_normalized'] = 1.0 * data['bbb_1']/(data['bbb_2']+data['bbb_1']+data['bbb_0'])
data['bbb_0_normalized'] = 1.0 * data['bbb_0']/(data['bbb_2']+data['bbb_1']+data['bbb_0'])

data['bbn_2_normalized'] = 1.0 * data['bbn_2']/(data['bbn_2']+data['bbn_1']+data['bbn_0'])
data['bbn_1_normalized'] = 1.0 * data['bbn_1']/(data['bbn_2']+data['bbn_1']+data['bbn_0'])
data['bbn_0_normalized'] = 1.0 * data['bbn_0']/(data['bbn_2']+data['bbn_1']+data['bbn_0'])

data['bbn_2_normalized'] = 1.0 * data['bbn_2']/(data['bbn_2']+data['bbn_1']+data['bbn_0'])
data['bbn_1_normalized'] = 1.0 * data['bbn_1']/(data['bbn_2']+data['bbn_1']+data['bbn_0'])
data['bbn_0_normalized'] = 1.0 * data['bbn_0']/(data['bbn_2']+data['bbn_1']+data['bbn_0'])

data['bb_0_normalized'] = 1.0 * data['bb_0']/(data['bb_0']+data['bb_1'])
data['bb_1_normalized'] = 1.0 * data['bb_1']/(data['bb_0']+data['bb_1'])

# compute structural diversity and structural diversity of the treated 
print('computing structural diversity')
structural_diversity = []

c = 0

for uid in list(data['id']):
    structural_diversity.append(
        nx.number_connected_components(nx.subgraph(G, [j for j in nx.neighbors(G, uid) if assignments[j] == 1]))
    )
    c += 1

data['structural_diversity'] = structural_diversity

structural_diversity_1 = []

c = 0

for uid in list(data['id']):
    structural_diversity_1.append(
        nx.number_connected_components(nx.subgraph(G, [j for j in nx.neighbors(G, uid)]))
    )
    c += 1

data['structural_diversity_1'] = structural_diversity_1

data['gender'] = np.random.randint(0, 2, len(data))

# pure cutoff
data['y1'] = data['neighbor'] * 0.1 + data['gender'] * 1 + \
            data['assignment'] * (data['bbb_2_normalized'] > 0.7).astype(float) * 2  + \
            np.random.normal(0, 1, len(data))

# structural diversity is causal
data['y2'] = \
    data['neighbor'] * 0.1 + data['gender'] * 1 + \
    data['structural_diversity'] + \
    data['assignment'] * data['structural_diversity'] * 1 + \
    np.random.normal(0, 1, len(data))

# structural diversity is correlational
data['y3'] = \
    data['neighbor'] * 0.1 + data['gender'] * 1 + \
    data['structural_diversity_1'] + \
    data['assignment'] * data['structural_diversity_1'] * 1 + \
    np.random.normal(0, 1, len(data))

# irrelevant covariates
data['y4'] = data['neighbor'] + np.random.normal(0, 1, len(data)) 

data.to_csv('data_ws.csv')

# bootstrapping
print('bootstrapping')
probabilities = []
for replicate in tqdm(range(100)):
    probabilities_mc = []
    assignments = np.concatenate([[k]*10 for k in list(np.random.randint(0, 2, 2000000//10))])
    r = np.random.randint(10)
    assignments = np.concatenate([assignments[r:], assignments[:r]])
    for i in sample:
        neighbor = len(G[i])
        bb_1 = np.sum([assignments[j] for j in G[i]])
        bb_0 = neighbor - bb_1
        bbb_0 = 0
        bbb_1 = 0
        bbb_2 = 0
        bbn_0 = 0
        bbn_1 = 0
        bbn_2 = 0
        open_square_0 = 0
        open_square_1 = 0
        open_square_2 = 0
        open_square_3 = 0

        for j in G[i]:
            for k in G[i]:
                if k > j:
                    if np.abs(j-k) <= 5:
                        if assignments[j] + assignments[k] == 0:
                            bbb_0 += 1
                        elif assignments[j] + assignments[k] == 1:
                            bbb_1 += 1
                        else:
                            bbb_2 += 1
                    else:
                        if assignments[j] + assignments[k] == 0:
                            bbn_0 += 1
                        elif assignments[j] + assignments[k] == 1:
                            bbn_1 += 1
                        else:
                            bbn_2 += 1
                        for l in G[i]:
                            if l > k and np.abs(l-k) > 5 and np.abs(l-j) > 5:
                                if assignments[j] + assignments[k] + assignments[l] == 0:
                                    open_square_0 += 1
                                elif assignments[j] + assignments[k] + assignments[l]== 1:
                                    open_square_1 += 1
                                elif assignments[j] + assignments[k] + assignments[l]== 2:
                                    open_square_2 += 1
                                else:
                                    open_square_3 += 1

        probabilities_mc.append([bb_0, bb_1, bbb_0, bbb_1, bbb_2, bbn_0, bbn_1, bbn_2,
                                 open_square_0, open_square_1, open_square_2, open_square_3, assignments[i]
                                ])
    probabilities.append(probabilities_mc)

probabilities = np.array(probabilities).T

np.save('probabilities_ws.npy', probabilities)
