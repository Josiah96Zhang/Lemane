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

Place the prepared edgelist file [graphname].txt in data/. Note that the first row of data is the node number.

**Example**

Wiki.txt

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
**Move the files in the frPCA and evaluation folder to the root directory before compiling.**


## Algorithm
### Learning Process

**Parameters**

- nepoch: number of training epochs 
- lr: learning rate
- nhop: maximum steps for random walks
- sample: number of negative samples per node
- data: Dateset name
- dist: initialization distribution for alphas, p for Possion and g for geometric
- param: initialization parameter, e.g., 1 for Possion and 0.5 for geometric distribution
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

**Parameters**

- graph: name of target graph
- graph_path: graph file path, default = 'data/'
- emb_path: embedding file path, default = 'embds/'
- alpha_path: trained alpha file path, default = 'alpha/'
- task: downstream task, link or class
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
bash example.sh
```

### Link Prediction

First, split the graph into training/testing set and generate negative samples. Training set will be saved into lp_data/train_graph/ folder, positive testing set will be saved into lp_data/test/positive/ folder, and negative testing set will be saved into lp_data/test/negative/ folder.

**Parameters**

- graph: name of target graph
- testing ratio: testing ratio, default = 0.3

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

The code to calculate link prediction precision is provided:

 ```
 ./link_pre_u -graph BlogCatalog -method lemane_frpca_link_u
 ```

### Node Classification
Generate the embeddings of full graph.

**Example**

```
./lemane_svd_u -graph BlogCatalog 
```

Train a classifier using the embeddings of full graph, the provided labels and the training set. The performance is evaluated in terms of average Micro-F1 of 5 runs.

```
python labelclassification.py --graph BlogCatalog --method lemane_frpca_class
```


## Citation
```
@inproceedings{lemane,
author = {Xingyi, Zhang and Kun, Xie and Sibo Wang and Zengfeng, Huang},
title = {Learning Based Proximity Matrix Factorization for Node Embedding},
year = {2021},
booktitle = {KDD},
}
```

