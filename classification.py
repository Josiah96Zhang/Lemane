import argparse
import torch.optim as optim
import numpy as np
import random
import torch
import os
import sys
import graph_util as gutil
import time
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wdecay1', type=float, default=0.01, help='weight decay (L2 loss on parameters 1).')
parser.add_argument('--nhop', type=int, default=15, help='Number of hops.')
parser.add_argument('--sample', type=int, default=10, help='Number of negative samples per node.')
parser.add_argument('--sample_sup_label_set', type=bool, default=False, help='Flag for sampling supervised node set.')
parser.add_argument('--patience', type=int, default=10, help='Patience.')
parser.add_argument('--data', default='BlogCatalog', help='Dateset.')
parser.add_argument('--dist', default='p', help='Initialization distribution for stopping probabilities, p for Possion and g for geometric.')
parser.add_argument('--param', type=float, default=5, help='Initialization parameter, e.g. 1 or 5 for Possion distribution and 0.5 for geometric distribution.')
parser.add_argument('--beta', type=float, default=1.0, help='Beta.')
parser.add_argument('--gamma', type=float, default=1.0, help='Gamma.')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--seed', type=int, default=1629459462, help='seed')
args = parser.parse_args()

seed = gutil.set_seed(args.seed)

task = "cl"
if args.data == 'BlogCatalog':
    num_label = 39
elif args.data == 'wiki':
    num_label = 40
else:
    raise Exception("A wrong graph name!")

n, _, directed = gutil.load_info(args.data)

if args.sample_sup_label_set == True:
   gutil.sup_label_set(args.data,n)
alphafile = "alpha/" + args.data + "_class.txt"
print("----------------------------------------------------------------")

G, adj, m = gutil.load_edge(args.data, n, directed, task)
cudaid = "cuda:" + str(args.dev) if torch.cuda.is_available() else "cpu"
device = torch.device(cudaid)
identity = torch.eye(n,dtype=torch.float64).to(device)

Lap_neg = gutil.get_sample_neg_laplacian(args.sample, n)
Lap_neg = Lap_neg.to(device)
prob = gutil.get_trans_prob_mat(adj,[],0,task)
prob = prob.to(device)
Node_Label = gutil.load_label(args.data, n)
Label_list, node_Label = gutil.get_sample_label_list(Node_Label, [],  num_label)
Lap_label = gutil.get_label_laplacian(Label_list)
for i in range(len(Lap_label)):
    Lap_label[i] = Lap_label[i].to(device)

threshold = 1e-5
model = gutil.ComputeProximity4SVD(ngraph=n, niter=args.nhop, dist=args.dist, param=args.param, nclass=num_label).to(device)
optimizer = torch.optim.SGD([{'params':model.params1,'weight_decay':args.wdecay1}],lr=args.lr)

def train():
    length_flag = True
    model.train()
    optimizer.zero_grad()
    output = model(prob, identity, threshold, task)
    output_prob = F.log_softmax(output, dim=1)
    output_0 = output[Label_list[0]]
    neg_loss = torch.trace(torch.transpose(output, 0, 1) @ (Lap_neg @ output))

    class_loss_fn1 = torch.trace(torch.transpose(output_0, 0, 1) @ (Lap_label[0] @ output_0))
    for i in range(1, len(Label_list)):
        output_i = output[Label_list[i]]
        class_loss_fn1 = class_loss_fn1 + torch.trace(torch.transpose(output_i, 0, 1) @ (Lap_label[i] @ output_i))
    class_loss_fn1 = class_loss_fn1  / (neg_loss * len(Label_list))
    class_loss_fn2 = - torch.mul(output_prob, node_Label.to(device)).sum() / n

    loss_fn = class_loss_fn1 * args.beta + class_loss_fn2 * args.gamma
    loss_fn.backward()

    print("params: ", model.params1.data)
    print("grad: ", model.params1.grad)

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
    loss_train, temp_dist, length_flag = train()
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
        print('Epoch:{:03d}'.format(epoch+1),'train_loss:{:.5f}'.format(loss_train)) #'time_spent:{:.5f}s'.format(time.time() - begin_time)
    if bad_count == args.patience:
        break

print("----------------------------------------------------------------")
print("Training time cost: {:.4f}s".format(time.time() - train_begin))
print("Best epoch:{}th".format(best_epoch))
print("Best distribution: {}".format(best_dist[-1]))
print("----------------------------------------------------------------")

np.savetxt(alphafile, best_dist[-1],fmt="%.4f",delimiter="\n")
