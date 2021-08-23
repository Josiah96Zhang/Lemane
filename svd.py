import numpy as np
import timeit
import pandas as pd
from sklearn import decomposition
import torch
from scipy import linalg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def simple_randomized_torch_svd(B, k, task):
    _, n = B.size()
    rand_matrix = torch.rand((n,k), dtype=torch.float64).to(device)   
    Q, _ = torch.qr(B @ rand_matrix)                                # qr decomposition
    Q.to(device)
    smaller_matrix = (Q.transpose(0, 1) @ B).to(device)
    U_hat, s, V = torch.svd(smaller_matrix, True)                   # matrix decompostion
    U_hat.to(device)
    U = (Q @ U_hat)

    if task == 'lp':
        return U @ (s.pow(0.5).diag()), V @ (s.pow(0.5).diag())     # for link prediction
    if task == 'cl':
        return torch.cat((U @ (s.pow(0.5).diag()),V @ (s.pow(0.5).diag())), 1) # for node classification
