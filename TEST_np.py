#!/usr/bin/python

import numpy as np
from rlq import *

x = [314, 272, 69]
rows = len(x)
cols = rows + 1

ml = rlq(rows, cols)
ml.setnull(x)
ml.digallinc()

p = np.array(ml.pid)
(U, S, Vh) = np.linalg.svd(p)

ss = np.diag(S)
#ss.reshape(rows, cols)
S = np.concatenate((ss, np.zeros((rows,1))), axis=1)

## see the state and interact...
import code
code.interact(local=locals())
