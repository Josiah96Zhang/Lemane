# Learning Based Proximity Matrix Factorization for Node Embedding

## Update

- There were some implementation issues in our evaluation code for node classification and we have fixed them. 
- We observe some significant improvement on Orkut dataset when normalization is applied. Thus, we take normalization for the embedding vectors on all datasets.

## Tested Environment

- Ubuntu 18.04
- C++ 11
- GCC 7.5
- [Eigen](https://eigen.tuxfamily.org/dox/)
- [Boost](https://www.boost.org/)
- [Gflags](https://github.com/gflags/gflags)
- [Inter MKL](https://software.intel.com/content/www/cn/zh/develop/tools/oneapi/base-toolkit/download.html?operatingsystem=linux&distributions=webdownload&options=offline)
- [Python 3.8.3](https://www.anaconda.com/products/individual#Downloads)
- [Sklearn 0.23.1](https://scikit-learn.org/stable/install.html)
- [Numpy 1.18.5](https://numpy.org/install/)
- [Pytorch 1.7.1](https://pytorch.org/get-started/locally/#linux-installation)
- [Networkx 2.4](https://networkx.org/)

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

**The trained alpha files are saved in alpha/ folder.** linkpred.py and classification.py are designed for small graphs, linkpred_sample.py and classification_sample.py are used for large graphs (n > 10k).  We also provide some suggestions to train alphas on a new graph in **Evaluation on new dataset** section.

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

  | task                | dataset     | dist | param | beta | gamma |  lr   | delta for push | svd for push |
  | :------------------ | ----------- | :--: | :---: | :--: | :---: | :---: | :------------: | :----------: |
  | link prediction     | Wikipedia   |  p   |   5   | 0.01 |   1   | 0.001 |      1e-5      |  JacobiSVD   |
  | link prediction     | Wikivote    |  p   |   1   | 0.5  |   1   |  0.5  |      1e-6      |    frPCA     |
  | link prediction     | BlogCatalog |  g   |  0.5  | 0.01 |   1   |  0.1  |      1e-7      |    frPCA     |
  | link prediction     | Slashdot    |  p   |   5   | 0.1  |   1   | 0.001 |      1e-5      |    frPCA     |
  | link prediction     | Tweibo      |  g   |  0.5  | 0.1  |   1   | 0.01  |      1e-5      |    frPCA     |
  | link prediction     | Orkut       |  p   |   1   |  1   |   1   | 0.01  |      1e-4      |    frPCA     |
  | node classification | BlogCatalog |  p   |   5   |  3   |   1   |  0.5  |      1e-5      |    frPCA     |
  | node classification | Wikipedia   |  p   |   5   | 0.5  |   1   |  0.5  |      1e-5      |  JacobiSVD   |
  | node classification | Tweibo      |  p   |   1   |  1   |   2   | 0.01  |      1e-5      |    frPCA     |
  | node classification | Orkut       |  p   |   1   |  1   |   2   |  0.5  |      1e-4      |    frPCA     |

**Examples**

Wikivote, link prediction, note that we need to split the data before training:

```
./gendata_d -graph wikivote -test_ratio 0.3
python linkpred.py --data wikivote --lr 0.5 --dist p --param 1 --beta 0.5 --gamma 1
```

BlogCatalog, classification:

```
python classification.py --data BlogCatalog --dist p --param 5 --beta 3 --gamma 1 --lr 0.5
```

### Generalized Push

We provide two versions of Randomized SVD to generate embeddings, i.e., [frPCA](https://github.com/XuFengthucs/frPCA_sparse) (filename: lemane_frpca_u.cpp, lemane_frpca_d.cpp) and [JacobiSVD](https://github.com/yinyuan1227/STRAP-git) (filename: lemane_svd_u.cpp, lemane_svd_d.cpp), 'u' for undirected graphs, 'd' for directed graphs. See **example_link.sh** and **example_class.sh** for more details. 

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

Wikipedia, link prediction:
```
./lemane_svd_d -graph wikipedia -graph_path lp_data/train_graph/ -task link -delta 0.00001
```

BlogCatalog, classification:
```
./lemane_frpca_u -graph BlogCatalog
```

## Experiments

**Use the following commands to reproduce the results.** example_link.sh for link prediction and example_class for node classification.

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

Splitting graph. For link prediction task, you should split the graph before the training process.

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
 ./linkpre_u -graph BlogCatalog -method lemane_frpca_link
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

Train a classifier(one vs all logistic regression) using the embeddings of full graph and the provided labels. The performance is evaluated in terms of Micro-F1 metric.

```
python labelclassification.py --graph BlogCatalog --method lemane_frpca_class
```

## Training and evaluation on new dataset

**Since the training process is complex and easily to fall into a local minimum,** if you have a new dataset, three initialized distributions, i.e. Poisson distribution with t = 1 and t = 5, geometric distribution with a = 0.5, are suggested for training alphas and evaluation. 

**If you have any questions on hyperparameter settings and training process, feel free to contact us <zxy96cuhk@gmail.com> and we are willing to provide suggestions or help to train our model on new datasets.**

### Training Process

We suggest using **grid search** to set hyperparameters beta and gamma from {0.01, 0.1, 0.5, 1, 2, 3}, learning rate from {0.001, 0.005, 0.01, 0.05, 0.1, 0.5} for each initialized distribution. **Run at least 5 times and select the one with the minimum loss** under each hyperparameter setting, i.e., you will get several groups of alphas for one task (link prediction or node classification) after the training process, each group is corresponding to a certain hyperparameter setting.

**Examples**

Suppose the number of nodes in [graphname] is more than 10k, and [graphname] is undirected.

Suppose we use Possion distribution with t = 1 as initialized distribution, set learning rate = 0.001, beta = 1, and gamma = 0.01 for link prediction, first split the data, then train alphas on the training graph:

```
./gendata_u -graph [graphname] -test_ratio 0.3
python linkpred_sample.py --data [graphname] --lr 0.001 --dist p --param 1 --beta 1 --gamma 0.01
```

Suppose we use geometric distribution with a = 0.5 as initialized distribution, set learning rate = 0.5, beta = 3 and gamma = 1 for node classification:

```
python classification_sample.py --data [graphname] --lr 0.5 --dist g --param 0.5 --beta 3 --gamma 1
```

### Generalized Push

We suggest using frPCA for social networks. For other types, using frPCA for graphs with more than 10k nodes and using JacobiSVD for small graphs with less than 10k nodes.

**Examples**

Suppose the number of nodes in a social network graph [graphname] is more than 10k, and [graphname] is undirected.

Generate embedding, and evaluate *lemane* for link prediction:

```
./lemane_frpca_u -graph [graphname] -graph_path lp_data/train_graph/ -task link
./linkpred_u -graph [graphname] -method lemane_frpca_link
```

Generate embedding, and evaluate lemane for node classification:

```
./lemane_frpca_u -graph [graphname]
python labelclassification.py --graph [graphname] --method lemane_frpca_class
```

**For each task, generate embeddings with different alphas trained under different settings and report the best result.**

## Citation

```
@inproceedings{lemane,
author = {Xingyi, Zhang and Kun, Xie and Sibo Wang and Zengfeng, Huang},
title = {Learning Based Proximity Matrix Factorization for Node Embedding},
year = {2021},
booktitle = {KDD},
}
```

