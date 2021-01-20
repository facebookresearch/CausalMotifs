
## CausalMotifs is the Replication Repository for: "Causal Network Motifs: Identifying Heterogeneous Spillover Effects in A/B Tests"


### Documentation 
This repository contains code to replicate the simulation results in the paper. We do not include data for replicating the internal FB results but note that the code architecture is the same as the simulation.

### Directions
There are two ways to run the Watts-Strogatz simulation notebook:
1. Run all the code (including network generation and MC replicates) in notebook 'GithubWWW2021_WS_assignment_local.ipynb' --- Warning: this can take a long time.
A fast way to run it (it can be run locally; uploading probabilities_ws.csv may be too large to upload to bento):
2. Use 'data_ws.csv' and 'probabilities_ws.npy' and start in the middle of the notebook
Start from and skip generate process

```
data = df.read_csv('data_ws.csv')
probabilities = df.read_csv('probabilities_ws.csv')
```

See the CONTRIBUTING file for how to help out.

## LICENSE
CausalMotifs is MIT Licensed, as found in the LICENSE file.

### Authors
* Yuan Yuan, yuan2@mit.edu
* Kristen M. Altenburger, kaltenburger@fb.com
* Farshad Kooti, farshadkt@fb.com
