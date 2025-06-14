
# File: ./TEST_rlq.py with 91 lines copied on Sat Jun 20 14:53:14 2015
from __future__ import print_function
from rlq import *
import random
import time
from prime import primesize
import copy

seed = time.time()
###seed = 1368991650.032
#seed = 1373060350.62
#seed= 1379445114.32
random.seed(seed)

ns=29
p=primesize(ns)
q=primesize(ns+1)
n=p*q
x=[n]
rows=30
cols=rows +1
quality = 1.8
disp = True


for i in range(1,rows): # say the others can be factored down to less digits
  x.append(random.randint(1,n-1))
  #bvec.append(random.randint(-1,1))

ml = rlq(rows,cols)

print("File: TEST_rlq.py")
print(time.ctime())
print("rows=", rows)
print("cols=", cols)
print("quality=", quality)
print("disp=", disp)
print("ns=", ns)
print("p=",p)
print("q=", q)
print("\nseed=",seed)


##ml.setnull(x)
##d = time.time()
##
##for i in range(2, ml.rows):
##  ml.reddim = i
##  ml.xrows()
##  ml.reset()
##  while ml.zrow(i): continue
##  if disp:print(i%10,end="")
##
##ml.reddim = ml.rows
##ml.xrows()
##ml.rowsort(ml.rownorm())
##for i in ml.rowrange: ml.house_row(i,i)
##while ml.zrow(ml.rows - 1):
##  while ml.zrow(ml.rows - 1): continue
##  ml.rowsort(ml.rownorm())
##  for i in ml.rowrange: ml.house_row(i,i)
##  if disp:print("|",end="")

##dtime = time.time() - d
##
##m1 = copy.deepcopy(ml)
##
##print("\nReduction round (xrows & zrows) completed for m1...")
##print("time = ", dtime)
##print("with norm = ", m1.norm())
##print("and smallest vector norm = ", dot(m1[0], m1[0]))

ml.setnull(x)

stime = time.time()

count = ml.digallinc(ml.rows, disp, quality)
  
dtime = time.time() - stime

m2 = copy.deepcopy(ml)

print("\nReduction round (digallinc) completed for m2...")
print("time = ", dtime)
print("with norm = ", m2.norm())
ml.reset()
print("and smallest vector norm = ", dot(m2[0], m2[0]))

print("\n\nFinished at ",end="")
print(time.ctime())
