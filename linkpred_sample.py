import argparse
import torch.optim as optim
import numpy as np
import random
import torch
import graph_util as gutil
import time
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate.')  
parser.add_argument('--wdecay', type=float, default=0.05, help='weight decay (L2 loss on parameters).')
parser.add_argument('--nhop', type=int, default=15, help='Number of hops.')
parser.add_argument('--sample', type=int, default=10, help='Number of negative samples per node.')
parser.add_argument('--sample_size', type=int, default=5000, help='Number of samples.')
parser.add_argument('--patience', type=int, default=10, help='Patience.')
parser.add_argument('--data', default='BlogCatalog', help='Dateset.')
parser.add_argument('--dist', default='p', help='Initialization distribution for stopping probabilities, p for Possion and g for geometric.')
parser.add_argument('--param', type=float, default=5, help='Initialization parameter, 1 or 5 for Possion distribution and 0.5 for geometric distribution.')
parser.add_argument('--beta', type=float, default=1, help='Beta.')
parser.add_argument('--gamma', type=float, default=0.2, help='Gamma.')
parser.add_argument('--dev', type=int, default=0, help='Device id')
args = parser.parse_args()

seed = int(time.time())
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

task = 'lp'  
n, _, directed = gutil.load_info(args.data)
alphafile = "alpha/" + args.data + "_link.txt"
print("----------------------------------------------------------------")

G, adj, m = gutil.load_edge(args.data, n, directed, task)
cudaid = "cuda:" + str(args.dev)
device = torch.device(cudaid)
identity = torch.eye(args.sample_size).to(device)

threshold = 1e-5
model = gutil.ComputeProximity4SVD(ngraph=args.sample_size, niter=args.nhop, dist=args.dist, param=args.param).to(device)
optimizer = torch.optim.SGD([{'params':model.params1,'weight_decay':args.wdecay}],lr=args.lr)

def train(adj):
    length_flag = True
    model.train()
    optimizer.zero_grad()
    U, V  = model(prob, identity, threshold, task)
    pro_mat = U @ V.transpose(0, 1)
    link_pre_loss_fn1 = F.mse_loss((pro_mat - torch.triu(pro_mat) + torch.triu(pro_mat, diagonal=1)).sum(0), degree, reduction = 'mean') / args.sample_size / args.sample_size
    output_prob = F.logsigmoid(U @ V.transpose(0, 1))
    link_pre_loss_fn2 =  - torch.mul(adj, output_prob).sum() / args.sample_size
    
    loss_fn = link_pre_loss_fn1 * args.beta + args.gamma * link_pre_loss_fn2
    loss_fn.backward()
    
    temp_dist = model.params1.clone().cpu().detach().numpy()
    optimizer.step()
    params_zero = model.params1.data[model.params1.data>=0]
    satlen = len(params_zero[params_zero<=1])
    if satlen < (args.nhop+1):
        length_flag = False
        print("Some teleport probabilities are out of range!")
    return loss_fn.item(), temp_dist, length_flag

best_dist = []
best_dist.append(model.params1.clone().cpu().detach().numpy())
min_loss = 999999999
best_epoch = 0
bad_count = 0
length_flag = True
train_begin = time.time()
print("----------------------------------------------------------------")

for epoch in range(args.nepoch):
    begin_time = time.time()
    
    bfs_sample_node_list = gutil.bfs_sampling(G, args.sample_size)
    prob, sample_adj, degree = gutil.get_trans_prob_mat(adj, bfs_sample_node_list, args.sample_size, task)   
    prob = prob.to(device)
    sample_adj = torch.tensor(sample_adj.todense()).to(device)
    degree = degree.to(device)
    
    loss_train, temp_dist, length_flag = train(sample_adj)
    if length_flag == False:
        break
    if loss_train < min_loss:
        min_loss = loss_train
        best_epoch = epoch+1
        best_dist.append(temp_dist)
        bad_count = 0
    else:
        bad_count += 1
    
    if(epoch+1)%10 == 0:
        print('Epoch:{:03d}'.format(epoch+1),'train_loss:{:.5f}'.format(loss_train),'time_spent:{:.5f}s'.format(time.time() - begin_time))
    if bad_count == args.patience:
        break
    np.random.seed(int(time.time()))

print("----------------------------------------------------------------")
print("Training time cost: {:.4f}s".format(time.time() - train_begin))
print("Best epoch:{}th".format(best_epoch))
print("Best distribution: {}".format(best_dist[-1]))
print("----------------------------------------------------------------")

np.savetxt(alphafile, best_dist[-1],fmt="%.4f",delimiter="\n")