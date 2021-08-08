#############################################
# File Name: graph_util.py                  #
# Created Time: 7 Jan 2021 :: PM +08        #
#############################################

import random
import svd
import numpy as np
import scipy.sparse as sp
import networkx as nx
import time
import os.path
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer

# load graph information
def load_info(dataset):     
    attr_path = "data/" + dataset + "/attr.txt"
    if not attr_path or not os.path.exists(attr_path):
        raise Exception("graph attr file does not exist!")
    with open(attr_path) as fin:
        n = int(fin.readline().split("=")[1])
        m = int(fin.readline().split("=")[1])
        directed = (fin.readline().strip()=="directed")
    fin.close()
    print("graph name: {}".format(dataset))
    return n, m, directed

# load graph edge
def load_edge(dataset, n, directed, task):
    if task == "lp":
        edgelist_path = "data/" + dataset + "/edgelist_link.txt"
    else:    
        edgelist_path = "data/" + dataset + "/edgelist.txt"
    if not edgelist_path or not os.path.exists(edgelist_path):
        raise Exception("edgelist file does not exist!")
    t1 = time.time()
    with open(edgelist_path, 'r') as f:
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for line in f:
            edge = line.strip().split()
            u, v = int(edge[0]), int(edge[1])
            G.add_edge(u, v)
        for i in range(n):
            if i not in G.nodes():
                G.add_edge(i,i)
    f.close()
    t2 = time.time()
    print('%fs taken for loading graph' % (t2 - t1))
    m = show_info(G)
    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    print('%fs taken for generating adj matrix' % (time.time() - t2))
    return G, adj, m

# show graph information
def show_info(G):
    print('Num of nodes: %d, num of edges: %d, Avg degree: %f, Directed:%s' % (G.number_of_nodes(), G.number_of_edges(), G.number_of_edges()*2./G.number_of_nodes(), str(nx.is_directed(G))))
    return G.number_of_edges()
    
# sample 5% labeled nodes as the supervised information for classification
def sup_label_set(dataset,n):
    label_path = "label/" + dataset + ".txt"
    sup_label_path = "label/" + dataset + "_sup.txt"
    labeled_list = []
    if dataset in ["tweibo", "orkut"]:
        if not label_path or not os.path.exists(label_path):
            raise Exception("node label file does not exist!")
        else:
            with open(label_path) as f:
                for line in f:
                    vec = line.strip().split()
                    i = int(vec[0])
                    labeled_list.append(i)
            f.close()
        index_list = random.sample(list(range(len(labeled_list))), int(0.05 * len(labeled_list)))
        sup_list = []
        for i in index_list:
            sup_list.append(labeled_list[i])
        sup_list.sort()
    else:    
        sup_list = random.sample(list(range(n)), int(0.05 * n))
        sup_list.sort()
    fout = open(sup_label_path, "w")
    for i in sup_list:
        fout.write(str(i) + "\n")
    fout.close()

# load node label
def load_label(dataset, n):
    sup_label_path = "label/" + dataset + "_sup.txt"
    Sup_List = []
    if not sup_label_path or not os.path.exists(sup_label_path):
        raise Exception("supervised node index file does not exist!")
    with open(sup_label_path) as f:
        for line in f:
            vec = line.strip().split()
            i = int(vec[0])
            Sup_List.append(i)
    f.close()

    label_path = "label/" + dataset + ".txt"
    Node_Label = [[]for i in range(n)]
    if not label_path or not os.path.exists(label_path):
        raise Exception("node label file does not exist!")
    if dataset in ["tweibo", "orkut"]:
        with open(label_path) as f:
            for line in f:
                vec = line.strip().split()
                i = int(vec[0])
                if i in Sup_List:
                    Node_Label[i] = vec[1:]
                    for j in range(len(Node_Label[i])):
                        Node_Label[i][j] = int(Node_Label[i][j]) + 1
    else: 
        with open(label_path) as f:
            for line in f:
                vec = line.strip().split()
                i = int(vec[0])
                if i in Sup_List:
                    Node_Label[i] = vec[1:]
                    for j in range(len(Node_Label[i])):
                        Node_Label[i][j] = int(Node_Label[i][j])
    f.close()   
    return Node_Label

# get label list of the sampled subgraph
def get_sample_label_list(Node_label, sample_list,  num_label):
    if len(sample_list) == 0:
        node_label = Node_label
    else:
        node_label = [Node_label[i] for i in sample_list]

    label_list = [ [] for i in range(num_label+1)]
    for i in range(len(node_label)):
        for j in node_label[i]:
            label_list[j].append(i)
    label_list = [k for k in label_list if len(k)>0]
    binarizer = MultiLabelBinarizer(sparse_output=False, classes=list(range(1,num_label+1)))
    node_label = torch.tensor(binarizer.fit_transform(node_label))
    return label_list, node_label

# get transition matrix of the sampled subgraph
def get_trans_prob_mat(adj, sample_list, sample_size, task):
    assert len(sample_list) == sample_size
    if sample_size == 0:
        sample_adj = adj
    else:
        sample_adj = adj[sample_list]
        sample_adj = sp.csc_matrix(sample_adj)[:,sample_list]
        sample_adj = sp.coo_matrix(sample_adj)
    row_sum = np.array(sample_adj.sum(1))
    row_sum[row_sum < 1] = 1
    degree = np.array(sample_adj.sum(1)).flatten()
    prob = sp.coo_matrix(sample_adj / row_sum).astype(np.float32) 
    indices = torch.from_numpy(np.vstack((prob.row, prob.col)).astype(np.int64))
    values = torch.from_numpy(prob.data)
    shape = torch.Size(prob.shape)
    if task == 'lp':
        adj = sample_adj
        return torch.sparse.FloatTensor(indices, values, shape), adj, torch.FloatTensor(degree)
    if task == 'cl':
        return torch.sparse.FloatTensor(indices, values, shape)

# get label-based graph laplacian matrix
def get_label_laplacian(L):
    Laplabel = []
    for i in range(len(L)):
        n = len(L[i])
        Laplabel_i = sp.coo_matrix(np.eye(n) * n - 1).tocoo().astype(np.float32)
        indices_i = torch.from_numpy(np.vstack((Laplabel_i.row, Laplabel_i.col)).astype(np.int64))
        values_i = torch.from_numpy(Laplabel_i.data)
        shape_i = torch.Size(Laplabel_i.shape)
        Laplabel_i = torch.sparse.FloatTensor(indices_i, values_i, shape_i)
        Laplabel.append(Laplabel_i)
    return Laplabel

# get graph laplacian matrix
def get_graph_laplacian(adj):
    degree = sp.diags(np.array(adj.sum(1)).flatten())
    Lapgraph = (degree - adj).tocoo().astype(np.float32)
    indices_g = torch.from_numpy(np.vstack((Lapgraph.row, Lapgraph.col)).astype(np.int64))
    values_g = torch.from_numpy(Lapgraph.data)
    shape_g = torch.Size(Lapgraph.shape)    
    return torch.sparse.FloatTensor(indices_g, values_g, shape_g)

# get sampling laplacian matrix and negative-sampling laplacian matrix
def get_sample_laplacian(adj, sample, n):
    adj_neg = sp.random(n, n, density=sample/n, data_rvs=np.ones)
    deg_neg = sp.diags(np.array(adj_neg.sum(1)).flatten())
    Lapneg = (deg_neg - adj_neg).tocoo().astype(np.float32)
    indices_neg = torch.from_numpy(np.vstack((Lapneg.row, Lapneg.col)).astype(np.int64))
    values_neg = torch.from_numpy(Lapneg.data)
    shape_neg = torch.Size(Lapneg.shape)
    
    adj = sp.coo_matrix(adj)
    sample_row = sorted(np.random.choice(len(adj.row), int(np.floor(sample * n)), replace=False))
    adj.row = adj.row[sample_row]
    adj.col = adj.col[sample_row]
    adj.data = adj.data[sample_row]
    deg_pos = sp.diags(np.array(adj.sum(1)).flatten())
    Lappos = (deg_pos - adj).tocoo().astype(np.float32)
    indices_pos = torch.from_numpy(np.vstack((Lappos.row, Lappos.col)).astype(np.int64))
    values_pos = torch.from_numpy(Lappos.data)
    shape_pos = torch.Size(Lappos.shape)
    
    Lapneg = torch.sparse.FloatTensor(indices_neg, values_neg, shape_neg)
    Lappos = torch.sparse.FloatTensor(indices_pos, values_pos, shape_pos)
    return Lappos, Lapneg

# get the laplacian matrix of a negative-sampled graph
def get_sample_neg_laplacian(sample, num_sample):
    adj_neg = sp.random(num_sample, num_sample, density=sample/num_sample, data_rvs=np.ones)
    deg_neg = sp.diags(np.array(adj_neg.sum(1)).flatten())
    Lapneg = (deg_neg - adj_neg).tocoo().astype(np.float32)
    
    indices_neg = torch.from_numpy(np.vstack((Lapneg.row, Lapneg.col)).astype(np.int64))
    values_neg = torch.from_numpy(Lapneg.data)
    shape_neg = torch.Size(Lapneg.shape)
    
    return torch.sparse.FloatTensor(indices_neg, values_neg, shape_neg)

# sample a subgraph with "max_sample" nodes via bfs
def bfs_sampling(G, max_sample):
    sample_list = set()
    while len(sample_list) < 5000:
        seed = random.randint(0,G.number_of_nodes()-1)
        #print("sample seed node: " + str(seed))
        queue = [seed]
        while queue:
            temp_node = queue.pop()
            if temp_node not in sample_list:
                sample_list.add(temp_node)
                queue.extend(list(set(G.adj[temp_node]) - sample_list))
            if len(sample_list) >= max_sample:
                return list(sample_list)

# compute node pair proximity and decompose the proximity matrix using SVD
class ComputeProximity4SVD(nn.Module):
    def __init__(self, ngraph, niter, dist, param, nclass=1):
        super(ComputeProximity4SVD, self).__init__()
        self.ngraph = ngraph
        self.niter = niter
        self.nclass = nclass
        self.params1 = Parameter(torch.FloatTensor(self.niter + 1))
        self.init_params(dist, param)
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(256, nclass))
        torch.nn.init.xavier_uniform_(self.fcs[0].weight)
    
    # compute the teleport probabilities of the random walks generated from Poisson distribution
    def poisson_dist(self, t):
        K = 100 
        poisson = [0]*(K+1)
        poissonsum = [0]*(K+2)
        stay = torch.FloatTensor(self.niter+1)
        poisson[0] = 1.0/math.exp(t)
        poissonsum[0] = 0.
        poissonsum[1] =  poissonsum[0] + poisson[0]
        for i in range(1, K+1):
            poisson[i] = poisson[i-1] * t * 1.0 / i
            poissonsum[i+1] = poissonsum[i] + poisson[i]
        
        for i in range(self.niter+1):
            stay[i] = poisson[i]/(1.0 - poissonsum[i]) 
        return stay
    
    # teleport probabilities initialization
    def init_params(self, dist, param):
        if dist == 'p':
            self.params1.data = self.poisson_dist(param)
        if dist == 'g':
            self.params1.data = torch.ones(self.niter + 1) * param
        #self.params1.data.uniform_(0,1)
        #self.params1.data = self.poisson_dist(1)
        #self.params1.data = torch.ones(self.niter + 1) * 0.2

    # forward propagation process   
    def forward(self, prob, identity, threshold, task):
        hi = identity
        prx_mat = hi * self.params1[0]
        for i in range(self.niter):
            hi = (prob @ hi) * (1 - self.params1[i])
            prx_mat = prx_mat + hi * self.params1[i+1]     
        prx_mat = prx_mat / threshold
        prx_mat[prx_mat < 1] = 1.
        prx_mat_log = prx_mat.log()
        if task == 'lp':
            U, V = svd.simple_randomized_torch_svd(prx_mat_log, 128, task) 
            return U, V
        if task == 'cl':
            embds_svd = svd.simple_randomized_torch_svd(prx_mat_log, 128, task)
            embds = self.fcs[0](embds_svd)    
            return embds
