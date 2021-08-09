# Learning Based Proximity Matrix Factorization for Node Embedding

## Tested Environment

- Ubuntu 18.04
- C++ 11
- GCC 7.5
- Eigen
- Boost
- Gflags
- Inter MKL
- Python
- Sklearn

## Layout

- data
- lp_data
- embds
- alpha
- frPCA
- label
- evaluation

## Input

Place the prepared edgelist file [graphname].txt in data/. Note that the first row of data is the number of nodes.

**Example**

wiki.txt

```
4777
0 0
0 1
0 2
0 3
0 4
...
```

## Compilations

```sh
bash compile.sh
```

## Algorithm
### Learning Process

linkpred.py and classification.py are designed for small graphs, linkpred_sample.py and classification_sample.py are used for large graphs (n > 10k).

**Parameters**

- nepoch: number of training epochs, default = 50
- lr: learning rate
- nhop: maximum steps for random walks, default = 15
- sample: number of negative samples per node, default = 10
- data: dateset name
- dist: initialization distribution for alphas, p for Possion and g for geometric, default = 'p'
- param: initialization parameter, e.g., 1 for Possion and 0.5 for geometric distribution, default = 5
- beta: hyperparameter in loss function
- gamma: hyperparameter in loss function

**Examples**

Wikipedia, link prediction:

```
python linkpred.py --data wikivote --lr 0.5 --dist p --param 1 --beta 0.5 --gamma 1
```

Tweibo, classification:

```
python classification_sample.py --data tweibo --lr 0.005 --beta 1 --gamma 3
```

### Generalized Push

We provide two versions of Randomized SVD to generate embeddings, i.e., frPCA (filename: lemane_frpca_u.cpp, lemane_frpca_d.cpp) and JacobiSVD (filename: lemane_svd_u.cpp, lemane_svd_d.cpp), 'u' for undirected graphs, 'd' for directed graphs. See **example_link.sh** and **example_class.sh** for more details. 

**Parameters**

- graph: name of input graph
- graph_path: graph file path, default = 'data/'
- emb_path: embedding file path, default = 'embds/'
- alpha_path: trained alpha file path, default = 'alpha/'
- task: downstream task, link or class, default = 'class'
- d: embedding dimension, default = 128
- delta: push error, default = 1e-5
- num_thread: number of threads used for push, default = 24

**Examples**

Wikivote, link prediction:
```
./lemane_frpca_d -graph wikivote -graph_path lp_data/train_graph/ -task link -delta 0.000001
```

BlogCatalog, classification:
```
./lemane_svd_u -graph BlogCatalog
```

## Experiments

**Use the alpha files in alpha/ folder to reproduce the results.**

```
bash example_link.sh
bash example_class.sh
```

### Link Prediction

First, split the graph into training/testing set and generate negative samples. Training set will be saved into lp_data/train_graph/ folder, positive testing set will be saved into lp_data/test/positive/ folder, and negative testing set will be saved into lp_data/test/negative/ folder.

**Parameters**

- graph: name of target graph
- testing ratio: testing ratio, default = 0.3
- method: embedding method name, default = 'lemane_frpca_link'

**Examples**

Splitting graph.

```
./gendata_u -graph BlogCatalog
```

```
./gendata_d -graph wiki
```
Then get embeddings of the training set. 

```
./lemane_frpca_u -graph BlogCatalog -graph_path lp_data/ -task link -delta 0.0000001
```

The code to calculate link prediction precision:

 ```
 ./link_pre_u -graph BlogCatalog -method lemane_frpca_link
 ```

### Node Classification
Generate the embeddings of full graph.

**Parameters**

- graph: name of target graph
- method: embedding method name, default = 'lemane_frpca_class'

**Example**

```
./lemane_svd_u -graph BlogCatalog 
```

Train a classifier using the embeddings of full graph, the provided labels and the training set. The performance is evaluated in terms of average Micro-F1 of 5 runs.

```
python labelclassification.py --graph BlogCatalog --method lemane_frpca_class
```

## Evaluation on new dataset

If you have new dataset, three initialized distributions, i.e. Poisson distribution with t=1 and t=5, geometric distribution with a=0.5, are suggested for training alphas and evaluation on this dataset. 

### Link prediction:

```
python linkpred_sample.py --data [graphname] --lr 0.01 --dist p --param 1 --beta 1 --gamma 1
python linkpred_sample.py --data [graphname] --lr 0.01 --dist p --param 5 --beta 0.1 --gamma 1
python linkpred_sample.py --data [graphname] --lr 0.01 --dist g --param 0.5 --beta 0.1 --gamma 1
```

### Node classification:

```
python classification_sample.py --data [graphname] --lr 0.1 --dist p --param 1 --beta 1 --gamma 2
python classification_sample.py --data [graphname] --lr 0.05 --dist p --param 5 --beta 1 --gamma 1
python classification_sample.py --data [graphname] --lr 0.05 --dist g --param 0.5 --beta 1 --gamma 2
```

**Since the training process is complex and easily to fall into a local minimum, run each command 5 times and select the one with minimum loss for each initialized distribution,** i.e., you will get 3 groups of alphas for one task (link prediction or node classification) after the training process, one for geometric distribution and two for Poisson distribution.

### Generalized Push

We suggest using frPCA for graphs with more than 10k nodes and using JacobiSVD for small graphs with less than 10k nodes.

**Examples**

The number of nodes in [graphname] is more than 10k.

**Link prediction**, 'u' for undirected graph and 'd' for directed graph:

```
./gendata_u -graph [graphname] -test_ratio 0.3
./lemane_frpca_u -graph [graphname] -graph_path lp_data/train_graph/ -task link
./linkpred_u -graph [graphname] -method lemane_frpca_link
```

**Node classification**, 'u' for undirected graph and 'd' for directed graph:

```
./lemane_frpca_u -graph [graphname]
python labelclassification.py --graph [graphname] --method lemane_frpca_class
```

**Run each part 3 times for 3 different alphas (one for each distribution) and report the best result.**

## Citation

```
@inproceedings{lemane,
author = {Xingyi, Zhang and Kun, Xie and Sibo Wang and Zengfeng, Huang},
title = {Learning Based Proximity Matrix Factorization for Node Embedding},
year = {2021},
booktitle = {KDD},
}
```

