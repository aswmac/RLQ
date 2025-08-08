#!/usr/bin/python
# just getting familiar with numpy a bit

import numpy as np
from rlq import *

x = [314159, 271828, 69314]
rows = len(x)
cols = rows + 1

ml = rlq(rows, cols)
ml.setnull(x)
ml.digallinc()

p = np.array(ml.pid)
(U, S, Vh) = np.linalg.svd(p)
ppt = p@np.transpose(p)

# build the actual diagonal for the singular values
ss = np.diag(S)
#ss.reshape(rows, cols)
S = np.concatenate((ss, np.zeros((rows,1))), axis=1)

#check the SVD is good
if np.allclose(p, U@S@Vh):
  print("The SVD checks out.")
else:
  print("The SVD went wrong!!!!")

E = np.eye(rows)
e = []
for i in range(rows):
  e.append(np.outer(E[i], E[i]))

d = []
for i in range(rows):
  dd = np.transpose(U)@e[i]@U
  ddd = np.diag(dd)
  d.append(ddd.tolist())

D = np.array(d)
beta = np.diag(p@np.transpose(p))
deltas = beta@np.linalg.inv(D)

xxt = np.diag(deltas) - p@np.transpose(p)

import scipy.optimize
import torch

def objective_function(d_numpy):
  d_torch = torch.tensor(d_numpy, requires_grad=True)
  loss = torch.sum(torch.diag(d_torch)**2)
  loss.backward()
  return loss.item(), d_torch.grad.numpy()

initial_guess = np.array([1, 2])
result = scipy.optimize.minimize(objective_function, initial_guess, jac=True, method='L-BFGS-B')
print(f"Optimal parameters: {result.x}")
print(f"Minimum loss: {result.fun}")


## see the state and interact...
import code
code.interact(local=locals())
