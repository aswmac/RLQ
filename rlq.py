'''
2022.10.03.181337---need general matrix to start, creating function setRowData(rowdata) 
                 --- (hopefully that is a new function because I think I always did x vector
                 --- which defined only the first column and made the rest the identity)
                 ml.setRowData([1,0,0,1,34,195,-270,208,-20,-110,0,1,0,0,5,27,-37,27,-3,-14,0,0,1,-1,-4,-50,73,-32,4,-17,0,0,0,2,-53,-288,392,-317,31,163,0,0,0,0,121,677,-932,737,-70,-380])
'''

# File: ./rlq.py with 1701 lines copied on Sat Jun 20 14:54:47 2015
# Integer lattice (row) reduction. Stores the integer exact values and the
# factored equivalent lattice when unitary operations applied on the column,
# and stores also that unitary matrix.
# This is a lattice in RLQ form, Reduced-diagonal Lower Unitary:
# or Row times Lower times Unitary
# The R is unimodular; integer row operations with unity determinant
# The L is lower triangular with the diagonal values (striving for lowest possible) from the top down
# The Q is unitary column operations.
# All row operations are integer, and a copy of the matrix with unitary column operations
# allowed is kept so that possible row-mix reductions become apparent.
# The main data are integers, stored in the list called self.pid
# The column (unitarily) altered list is called self.row and the alterations are
# stored in the list self.corow. The corow is kept so that as reductions are made
# the reduced row values may be recalculated in LQ form for precision without
# the need to recalculate the full LQ for all rows.

# CALL DEPENDENCY
# ----------------------------------------------------------------------
# -----------main functions---------------------------------------------
# ----------------------------------------------------------------------
# __init__()      --> NONE
# house_row()     --> NONE
# align()         --> house_row()
# setnull()       --> align()
# digallinc()     --> digall(), zrow()
# digall()        --> digest(), reset(), lq() (lq implied: input form)
# zrow()          --> house_row()
# digest()        --> givens()
# reset()         --> NONE
# lq()            --> house_row()
# givens()        --> NONE
# ----------------------------------------------------------------------
# -----------finishing graph enumeration functions----------------------
# ----------------------------------------------------------------------
# ered()          --> enum(), rowswap(), rownorm()
# enum()          --> house_row(), row_sub_place(), rowswap(), colswap()
# row_sub_place() --> NONE
# rowswap()       --> NONE
# colswap()       --> NONE
# ----------------------------------------------------------------------
# ------------other useful functions------------------------------------
# ----------------------------------------------------------------------
# rownorm()       --> NONE
# crestmax()      --> house_row(), rowswap(), row_sub_place(), givens()
# xrows()         --> house_row(), xcol()
# xcol()          --> NONE
from __future__ import print_function
from num_funs import dot, gcd
#import os # for read and write bitmaps, from_bmp() and to_bmp()
import time # to print run times in qsvd()
import math # used in the duo_mat(), orthotridiag()
import random # used to shuffle the row order in shuffle()
#from matrix_base import * # the use of minbox in minit() for now

class rlq(object):
  def __init__(self,rows,cols):
    ''' matrix with co-matrix rows. co-matrix is the scaled inverse with determinant (self.det) as that scale '''
    #if rows > cols:
    #  print("class rlq requires rows <= cols!")
    #  self.cols = rows
    #else: self.cols=cols  # columns are always same as or greater than rows since this is a row operation class type matrix
    self.cols=cols  # columns are always same as or greater than rows since this is a row operation class type matrix
    self.rows=rows
    self.rowrange=range(self.rows)
    self.colrange=range(self.cols) #
    self.pid  =[[0 for i in self.colrange] for j in self.rowrange] # table of rows
    self.row  =[[0.0 for i in self.colrange] for j in self.rowrange] # full span in dimension
    self.corow=[[0.0 for i in self.colrange] for j in self.colrange] # full span and square to keep Q of the LQ
    self.disp = 1 # flag to denote the display of call matrix or not
    self.temprow = [0.0 for i in self.colrange] # a row of scratch memory to use for calculations
    self.tempcol = [0.0 for i in self.rowrange] # a col of scratch memory to use for calculations
    self.reddim = 0
    #self.det=1.0 # the volume of the parallepiped of the full span of rows (determinant if matrix is square)
    #self.epsilon = 1e-11 # the machine smallest value
    # row_sub_place, rowneg, rowswap, colswap, colneg each function which alters the self, functions like rowsort and rowmix use rowswap and row_sub_place
  def setRowData(self, rowdata):
  	''' set the pid row data and reset all else '''
  	assert len(rowdata) == self.rows*self.cols, "Data length mismatch!"
  	for i in self.colrange: # set the unitary corow square matrix as identity
  	  for j in self.colrange:
  	    if i==j: self.corow[i][j] = 1.0
  	    else: self.corow[i][j] = 0.0
  	for j in self.colrange: # set the pid integer data as the input data
  	  for i in self.rowrange:
  	    self.pid[i][j] = rowdata[i*self.cols + j]
  	    self.row[i][j] = float(self.pid[i][j])
  def setnull(self, x):
    ''' set the self as rlq to the vector values x'''
    for i in self.colrange:
      for j in self.colrange:
        if i==j: self.corow[i][j] = 1.0
        else: self.corow[i][j] = 0.0
    for i in range(self.rows):
      self.pid[i][0]  = x[i]
      self.row[i][0]  = float(x[i])
      for j in range(1, self.cols):
        if j==i+1:
          self.pid[i][j] = 1
          self.row[i][j] = 1.0
        else:
          self.pid[i][j] = 0
          self.row[i][j] = 0.0
    return self.align(x)
    for i in self.rowrange: self.house_row(i,i) # do not reorder the rows just yet as would lq()
    self.reddim = 2
  def align(self, x, reset = False):
    ''' set the self as rlq to the vector values x, allowing
        for large values, altering the bulk of the pid
        only if reset is True'''
    if reset: # If the pid values are to be reset to identity
      for i in self.rowrange:
        self.pid[i][0]  = 0
        self.row[i][0]  = 0.0
        for j in range(1, self.cols):
          if j==i+1:
            self.pid[i][j] = 1
            self.row[i][j] = 1.0
          else:
            self.pid[i][j] = 0
            self.row[i][j] = 0.0
    full = True
    z = self*([-1] + x) # TODO: dimension checking...
    zs = [self.pid[i][0] + z[i] for i in self.rowrange]
    mz = abs(max(zs, key=lambda x: abs(x)))
    scale = 1
    while mz//scale >= 10**127: scale *= 2 # scale by a factor of two so that results are less than 127 binary digits
    if scale > 1:
      full = False
      for i in self.rowrange:
        zs[i]//=scale
    for i in self.rowrange: # reset the row values
      self.pid[i][0]  = zs[i]
      self.row[i][0]  = float(zs[i])
      for j in range(1, self.cols):
        self.row[i][j] = float(self.pid[i][j])
    for i in self.colrange: # reset the unitary rotation matrix to identity, to prepare for LQ form
      for j in self.colrange:
        if i==j: self.corow[i][j] = 1.0
        else: self.corow[i][j] = 0.0
    for i in self.rowrange: self.house_row(i,i) # do not reorder the rows as would lq()
    return full
  def reset(self, reorder = True):
    ''' reset the rows as copy of pid, and corows as the comatrix pseudo-inverse'''
    for i in self.rowrange:
      for j in self.colrange:
        self.row[i][j]   = float(self.pid[i][j])
    for i in self.colrange:
      for j in self.colrange:
        if i==j: self.corow[i][j]   = 1.0
        else: self.corow[i][j]   = 0.0
    #self.reddim = self.rows
    if reorder: self.lq()
  def test(self):
    ''' test that self.row (the L matrix) is in lower triangular form '''
    t = 0.0
    for i in self.rowrange:
      for j in range(i+1, self.cols):
        t += self.row[i][j]**2
    return t
  def setrow(self, rm):
    ''' use the corow matrix to recalculate the row '''
    for j in self.colrange: # use the comatrix to recalculate the row to keep precision
      self.row[rm][j] = 0.0
      for g in self.colrange:
        self.row[rm][j] += float(self.pid[rm][g])*self.corow[g][j]
  def norm(self, rm=0):
    ''' return the sum of squares for rows rm and below'''
    s=0
    for i in range(rm, self.rows):
      self.tempcol[i] = 0
      for j in self.colrange:
        self.tempcol[i] += self.pid[i][j]*self.pid[i][j]
      s += self.tempcol[i]
    return s
  def house_row(self, ir, ic): #
    ''' zero across the row of the row matrix'''
    if ir >= self.rows or ic >= self.cols: return
    nn=0 # the norm squared
    for j in range(ic+1,self.cols):
      nn+=self.row[ir][j]*self.row[ir][j]
      self.temprow[j]=self.row[ir][j]               # set the u vector (self.temp) as the column ic
    if nn <= 1e-6: return
    uu=nn  # the norm squared of the u vector ( the column ic plus one element at index ir)
    nn+=self.row[ir][ic]*self.row[ir][ic]
    n=math.sqrt(nn) # n is the norm of row ir
    a1 = abs(self.row[ir][ic])
    if a1 >= 1e-6:
      mu = self.row[ir][ic]/a1 # complex phase of the first element (just one for real value)
      self.temprow[ic] = self.row[ir][ic] + mu*n # first element of the u vector plus the norm rotated by the phase of the first element
    else:
      self.temprow[ic] = n # first element of the u vector plus the norm rotated by the phase of the first element
    uu+=self.temprow[ic]*self.temprow[ic]
    k=-2/uu
    for i in self.rowrange: # now do the right multiplication (col-mix)
      urow_i = 0
      for j in range(ic,self.cols):
        urow_i  += self.temprow[j]*self.row[i][j]
      kr = k*urow_i
      for j in range(ic, self.cols):
        self.row[i][j]  += kr*self.temprow[j]
    for i in self.colrange: # all of the corows are kept
      ucorow_i = 0
      for j in range(ic,self.cols):
        ucorow_i+= self.temprow[j]*self.corow[i][j]
      kc = k*ucorow_i
      for j in range(ic, self.cols):
        self.corow[i][j]+= kc*self.temprow[j]
  def house_col(self, ir, ic, target=None):
    ''' rotate the rows of the row and corow matrix only (not the pid) to zero the column '''
    nn=0 # the norm squared of column ic from row ir down
    if target == None: target = [self.row[i][ic] for i in self.rowrange]
    for i in range(ir+1,self.rows):
      nn += target[i]*target[i]
      self.tempcol[i]=target[i]               # set the u vector (self.temp) as the column ic
    if nn <= 1e-6: return
    uu=nn  # the norm squared of the u vector ( the column ic plus one element at index ir)
    nn+=target[ir]*target[ir]
    n=math.sqrt(nn) # n is the norm of column ic
    a1 = abs(target[ir])
    if a1 >= 1e-6:
      mu = target[ir]/a1 # complex phase of the first element (just one for real value)
      self.tempcol[ir] = target[ir] + mu*n # first element of the u vector plus the norm rotated by the phase of the first element
    else:
      self.tempcol[ir] = n # first element of the u vector plus the norm rotated by the phase of the first element
    uu+=self.tempcol[ir]*self.tempcol[ir]
    k=-2/uu
    for j in self.colrange: # now do the left multiplication (row-mix)
      urow_j = 0
      for i in range(ir,self.rows):
        urow_j+=self.tempcol[i]*self.row[i][j]
      kc = k*urow_j
      for i in range(ir, self.rows):
        self.row[i][j]+=kc*self.tempcol[i]
    for j in self.colrange: # keep the corows
      uq_j = 0
      for i in range(ir,self.rows):
        uq_j   +=self.tempcol[i]*self.corow[i][j]
      kq = k*uq_j
      for i in range(ir, self.rows):
        self.corow[i][j] +=kq*self.tempcol[i]
  def givens(self, r, c0, c1):
    ''' do givens on the column between columns c0 and c1 so that the value in row r column c1 is zero '''
    c= self.row[r][c0]
    s = self.row[r][c1]
    n = c*c + s*s
    if n < 1e-4: dis = math.hypot(c,s) # accuracy for small values
    else: dis = math.sqrt(n)
    if dis==0: return
    c,s = c/dis, s/dis        #left.row=[[c,-s],[s,c]] # the left unitary rotation
    for g in self.rowrange: # now do the similarity #####inline of unitary similarity #####
      i = c*self.row[g][c0] + s*self.row[g][c1]
      j = c*self.row[g][c1] - s*self.row[g][c0]
      self.row[g][c0] = i
      self.row[g][c1] = j
    for g in self.colrange: # now do the similarity #####inline of unitary similarity #####
      i = c*self.corow[g][c0] + s*self.corow[g][c1]
      j = c*self.corow[g][c1] - s*self.corow[g][c0]
      self.corow[g][c0] = i
      self.corow[g][c1] = j
  def pidcol(self, ic):
    ''' return the column of the pid matrix'''
    return [self.pid[i][ic] for i in self.rowrange]
  def cocol(self, ic):
    ''' return the column of the corow matrix'''
    return [self.corow[i][ic] for i in self.rowrange]
  def col(self, ic):
    ''' return the column of the row matrix'''
    return [self.row[i][ic] for i in self.rowrange]
  ############################################################################################################
  ###  non-basic methods (may want to remove or put in a separate class for simplicity of the rlq class)  ####
  ############################################################################################################
  def target(self, v=None, full=False):
    ''' least squares rowmix to target vector v, or to self.temprow if not v given'''
    self.reset(False)
    if v != None:
      for i in self.colrange:
        self.corow[0][i] = float(v[i]) #<-------- copy input v into a corow so that it gets the same rotations
    else:
      for j in self.colrange: #<-------------------------------------------- or use the temprow as the default
        self.corow[0][j] = self.temprow[j] #<--
    for i in range(self.reddim-1, -1, -1):
      self.house_row(i,self.reddim - 1 - i) #<---------------------- perform (rowflipped) LQ up to self.reddim
    for i in range(self.reddim): #<--------------------------------- do least squares reduction on the temprow
      for j in range(i): #<--------------------- when j = i-1 want givens(j, self.reddim - 2, self.reddim - 1)
        self.givens(j, self.reddim + j - i - 1, self.reddim + j - i) #
      self.tempcol[i] = self.corow[0][self.reddim - 1]/self.row[i][self.reddim - 1]
      if not full: self.tempcol[i] = int((self.tempcol[i] + 0.5)//1.0) #<----------------- the nearest integer
    return self.tempcol[:]
  def target2(self, t):
    ''' use house_col() to target a vector'''
    b = [0.0 for i in self.rowrange] # the scalar vector-- the only non unitary element
    d = [0.0 for i in self.rowrange] #
    te = [0.0 for i in self.rowrange] # the target error at each step
    tr = t[:] # the residual if keep adding dimension to the targetting
    tt = t[:] # the residual if end the targetting
    self.reset(False)
    y = [dot(self.row[i], tt) for i in self.rowrange]
    for i in self.rowrange:
      self.house_col(i,0, y)
      b[i] = dot(tt,tt)/dot(self.row[i], tt)
      d[i] = dot(self.row[i], tt)/dot(self.row[i], self.row[i])
      for j in self.colrange:
        tr[j] = tt[j] - d[i]*self.row[i][j]
        tt[j] = tt[j] - b[i]*self.row[i][j]
      te[i] = dot(tr,tr)
      y = [dot(self.row[i], tt) for i in self.rowrange]
    return b, te
  def colnorm(self, c):
    ''' return the norm squared of the c column'''
    s=0
    for r in self.rowrange: s+=self.pid[r][c]*self.pid[r][c]
    return s
  ############################################################################################################
  ####  non-basic row reduction methods (may want to remove  for simplicity of the rlq class)
  ############################################################################################################
  def plq(self, ir=0, ic=0):
    ''' LQ except for all positive values, requires some rowmixing.
        Run once and all vectors are oriented such that dot products are all positive'''
    while ir < self.reddim and ic < self.reddim: # The regular lq()
      self.house_row(ir,ic)
      ir+=1
      ic+=1
    for i in range(self.reddim-1, -1, -1):
      if self.row[i][i] < 0: self.rowneg(i)
      for r in range(i+1, self.reddim):
        k = int(self.row[i][r]//self.row[i][i])
        if k!=0: self.row_sub_place(r,i,k)
  def qr(self):
    ''' do lq except upper triangular instead of lower'''
    ir,ic = self.rows-1,0
    while ir >=0 and ic < self.cols:
      self.house_row(ir,ic)
      ir-=1
      ic+=1
    self.colflip()
  def xrows(self, pivot=0, disp=False):
    ''' do row reductions. From xcol() for pivot >=1 the assumption is lower triangular form of the row matirix '''
    redd = True
    while redd:
      if disp: print(".", end="")
      redd = False
      for i in range(pivot, self.reddim): # give each row a chance at reducing
        self.house_row(i,pivot)
        redd = self.xcol(i,pivot, pivot) or redd
  def xcol(self, ir, ic, first = 0):
    ''' use the row matrix to find row mixes which reduce other pid-rows.
        For first>0 the assumption is that the row matrix is in lq form up to the first row'''
    rrd = False
    for i in range(first, self.reddim): #<- use row ir to reduce other rows
      if i==ir or abs(self.row[ir][ic]) < 1e-6: continue
      t = self.row[i][ic]/self.row[ir][ic]
      if abs(t) - 0.5 < 1e-6: continue # if the possible change is not of any significant magnitude
      k = int((t + 0.5)//1.0)
      if k!=0: #<-- a reduction may take place
        for j in self.colrange:
          self.pid[i][j] -= k*self.pid[ir][j] # row_sub_place() on just the pid[], row[] is recalculated later
        for j in range(first-1, -1, -1): #<-find the other row value reductions
          self.row[i][j] = 0.0
          for g in self.colrange: # recalculate the row[i][j] value for precision
            self.row[i][j] += float(self.pid[i][g])*self.corow[g][j]
          if abs(self.row[j][ic]) < 1e-6: continue
          kj = int((self.row[i][ic]/self.row[j][ic] + 0.5)//1.0)
          if kj != 0: #<- a previous row j now gives a reduction
            for g in self.colrange:
              self.pid[i][g] -= kj*self.pid[j][g]
        rrd = True # TODO: an infinite loop is possible here, need to calculate the norm change
        for j in self.colrange: # now use corow[] to recalculate all of row[] to keep precision
          self.row[i][j] = 0.0
          for g in self.colrange:
            self.row[i][j] += float(self.pid[i][g])*self.corow[g][j]
    return rrd
  def prows(self, disp=False):
    ''' do row reductions, similar to xrows but with positive cross-dot results '''
    redd = True
    while redd:
      for i in range(self.reddim): #self.reset() # re-align the rows to the integers
        for j in self.colrange:
          self.row[i][j] = float(self.pid[i][j])
      if disp: print(".", end="")
      redd = False
      rn = self.rownorm()
      mn = min(rn)
      mni = rn.index(mn)
      self.house_row(mni,0)
      redd = self.pcol(mni,0) or redd # TODO: choose all positive or all negative as is better
  def pcol(self, ir, ic): # TODO: if all elements are positive, does the svd to integer box become easier?
    ''' xcol() except for positive only results'''
    t = [self.row[i][ic]/self.row[ir][ic] for i in range(self.reddim)] #<-- get all mix numbers
    tpf = [t[i] - t[i]//1.0 for i in range(self.reddim)] #<-- get all the fractional error for positive
    mx = max(tpf)
    tpf[ir] = 1.0 # the reducing row has exact and not fractional...
    mn = min(tpf)
    if mn < 1.0 - mn: # do rowmix that matches the sign
      rrd = False
      for i in range(self.reddim):
        if i==ir: continue
        k = int(t[i]//1.0) # nearest int of the ratio
        if k!=0:
          rrd = True
          for j in self.colrange: # row_sub_place() without keeping corows
            self.pid[i][j] -= k*self.pid[ir][j]
            self.row[i][j] -= k*self.row[ir][j]
      return rrd
    else: # do rowmix that differs from the sign
      rrd = False
      for i in range(self.reddim):
        if i==ir: continue
        k = int(t[i]//1.0) + 1 # nearest int of the ratio
        if k!=0:
          rrd = True
          for j in self.colrange: # row_sub_place() without keeping corows
            self.pid[i][j] -= k*self.pid[ir][j]
            self.row[i][j] -= k*self.row[ir][j]
      self.rowneg(ir)
      return rrd
  def tspec(self, rm, thresh):
    ''' find the hypercube outside which a rowmix to rm necessarily has norm greater than thresh'''
    if rm >= self.rows: return
    if rm != self.reddim-1: self.row[rm], self.row[self.reddim-1] = self.row[self.reddim-1], self.row[rm]
    point = [0.0 for i in range(self.reddim)] #<- rowmix of floating point targets exactly
    point[self.reddim-1] = 1.0 # the target row, to be swapped back to index rm later
    shift = [0.0 for i in range(self.reddim)] #<-- the plus or minus shift difference from point defining the hypercube
    for i in range(self.reddim-2, -1, -1): #<- target row is now at self.reddim-1
      self.house_row(i,self.reddim - 2 - i) #<-- perform (rowflipped) LQ up to that point
    self.house_row(self.reddim - 1, self.reddim - 1) #<-- the rest of target's norm in lower right,
    ospan = self.row[self.reddim-1][self.reddim-1] # reveals the orthogonal portion of the row which is not reduceable
    play_2 = thresh**2 - ospan**2 #<- the remaining norm
    if play_2 < 0.0: return point, shift #<-- all rowmixes are larger than thresh, return a zero shift
    play = math.sqrt(play_2)
    for i in range(self.reddim-1):
      for j in range(i): #<- when j=i-1, want givens(j, reddim-3, reddim-2)
        self.givens(j,self.reddim + j - i - 2, self.reddim + j - i - 1) #<---------------- see page 158 of 2013.5.19.1048
      point[i] = -self.row[self.reddim - 1][self.reddim - 2]/self.row[i][self.reddim - 2]
      shift[i] = abs(play/self.row[i][self.reddim - 2]) #<-- the scale at which row i alone assures norm is larger than thresh
    if rm != self.reddim-1: #<-------- if the row was swapped, swap it back along with the calculation data
      self.row[rm], self.row[self.reddim-1] = self.row[self.reddim-1], self.row[rm]
      point[rm], point[self.reddim-1] = point[self.reddim-1], point[rm]
      shift[rm], shift[self.reddim-1] = shift[self.reddim-1], shift[rm]
    return point, shift
  def trow(self, rm):
    ''' use only unitary column mixing to calculate the least squares targetting of row rm'''
    if rm >= self.rows: return
    if rm != self.reddim-1: self.row[rm], self.row[self.reddim-1] = self.row[self.reddim-1], self.row[rm]
    point = [0.0 for i in range(self.reddim)] #<--------------------- rowmix of floating point targets exactly
    point[self.reddim-1] = 1.0 #<------------------------ the target row, to be swapped back to index rm later
    berror = [0.0 for i in range(self.reddim)] #<-------- the error to the nearest integer (Babai point error)
    for i in range(self.reddim-2, -1, -1): #<------------------------------ target row is now at self.reddim-1
      self.house_row(i,self.reddim - 2 - i) #<----------------------- perform (rowflipped) LQ up to that point
    self.house_row(self.reddim - 1, self.reddim - 1) #<------------- the rest of target's norm in lower right,
    for i in range(self.reddim-1):
      for j in range(i): #<------------------------------------ when j=i-1, want givens(j, reddim-3, reddim-2)
        self.givens(j,self.reddim + j - i - 2, self.reddim + j - i - 1) #<----- see page 158 of 2013.5.19.1048
      point[i] = -self.row[self.reddim - 1][self.reddim - 2]/self.row[i][self.reddim - 2]
      babai_diff = point[i] - (point[i] + 0.5)//1.0
      berror[i] = abs(babai_diff*self.row[i][self.reddim - 2]) #<---- the mandatory error with nearest integer
    if rm != self.reddim-1: #<----------- if the row was swapped, swap it back along with the calculation data
      self.row[rm], self.row[self.reddim-1] = self.row[self.reddim-1], self.row[rm]
      point[rm], point[self.reddim-1] = point[self.reddim-1], point[rm]
      berror[rm], berror[self.reddim-1] = berror[self.reddim-1], berror[rm]
    return point, berror
  def zup(self, rm = None):
    ''' do zrow, and shift the row if basis is improved.
        Improvement is any upward rowshift that reduces a diagonal (basis) value.
        Assumption of LQ form '''
    if rm==None: rm = self.reddim - 1 # always looking at the last row
    newrow = rm # the index to which the row should be moved for a better basis/diag-value
    t0 = self.row[rm][rm]**2 # the running norm
    for rp in range(rm-1, -1, -1):
      if abs(self.row[rp][rp]) < 1e-6: continue # should not be a problem, but want to avoid zero
      self.row[rm][rp] = 0.0 # reset and recalculate the pertinent row value
      for g in self.colrange:
        self.row[rm][rp] += float(self.pid[rm][g])*self.corow[g][rp]
      t = self.row[rm][rp]/self.row[rp][rp] # find the next rowmix value
      if abs(abs(t) - 0.5) < 1e-6: k = 0
      else: k = int((t + 0.5)//1.0) # the nearest int
      if k != 0: # perform the rowmix
        self.row[rm][rp] -= k*self.row[rp][rp]
        for g in self.colrange:
          self.pid[rm][g] -= k*self.pid[rp][g]
      t0 += self.row[rm][rp]**2 # add to the running norm total
      if t0 - self.row[rp][rp]**2 < 1e-6: # if a reducing mix has been found (better than precision)
        newrow = rp # TODO: perform rowswap, and reorder the basis instead of calling lq() or reset()
    for i in range(rm, newrow, -1): # slide the row to the new index
      self.givens(i, i-1, i)          # while preserving the lq form
      self.rowswap(i, i-1)
    return newrow
  def zrow(self, rm = None, start = None):
    ''' target reduce the row, default of the last row. Use start to limit which rows to use for reducing
        Assumption: row[] is in lq form.'''
    if rm==None: rm = self.rows-1
    if start==None: start = 0 # the least index of rows to use for reducing
    ir,ic = 0,0
    d0 = 0
    kvec = [0 for i in range(rm+1)]
    kvec[rm] = 1
    for j in self.colrange: d0 += self.pid[rm][j]**2 # get the rownorm of row rm
    for i in range(rm-1, start-1, -1):
      if abs(self.row[i][i]) < 1e-6: continue # ignore small components (while also avoiding zero division)
      self.row[rm][i] = 0.0 # setup to reset row[rm][i] for precision/accuracy
      for g in self.colrange:
        self.row[rm][i] += float(self.pid[rm][g])*self.corow[g][i]
      t = self.row[rm][i]/self.row[i][i]
      if abs(abs(t) - 0.5) < 1e-6: continue # ignore small reduction amounts
      k = int((t + 0.5)//1.0) # nearest int of the ratio TODO: consider if too large, ie NaN values
      if k == 0: continue # if there is no reduction
      kvec[i] = k
      for j in self.colrange: # inline of row_sub_place(rm, i, k)
        self.pid[rm][j] -= k*self.pid[i][j]
      self.row[rm][j] = 0.0 # reset the new row[rm][i] value for precision (reductions lose accuracy)
      for g in self.colrange: #TODO: after reducing, other values become less "zero like" in relative comparison
        self.row[rm][i] += float(self.pid[rm][g])*self.corow[g][i]
    #print(kvec)
    d1 = 0
    for j in self.colrange: d1 += self.pid[rm][j]**2 # get the new rownorm of rm after possible mixing
    return d1 < d0
  def spinered(self):
    ''' use the spine function to find reducing rowmixes'''
    norms = [0 for i in range(self.reddim)]
    for i in range(self.reddim):
      norms[i] = 0.0
      for j in self.colrange:
        self.row[i][j] = float(self.pid[i][j]) # reset the rows
        norms[i] += self.row[i][j]**2 #<-- store the rownorm for later comparison
    mn = None #<-- prepare to find the smallest result of spine()
    for i in range(self.reddim):
      z = self.spine(i,2)
      y = z*self
      test= dot(y,y)
      if mn == None or test < mn:
        mn, minz= test, z[:]
    mxdiff = None
    for i in range(self.reddim): #<-- find a unitary rowmix to test placement
      if minz[i] != 1 and minz[i] != -1: continue
      testdiff = norms[i] - mn #<-- find the difference in norm if use this row for rowmix
      if testdiff > 0: #<-- if the result is a reduction
        if mxdiff == None or testdiff > mxdiff: #<-- make a note of it
          mxdiff, diff_i = testdiff, i
    if mxdiff == None: return False #<- no improving rowmix is possible, at least any that are unimodular
    if minz[diff_i] == -1: self.rowneg(diff_i)
    self.rowmix(diff_i, minz)
    return True
  def spine(self, rm, k):
    ''' find the reduction rowmix of a scaled row'''
    for i in self.rowrange:
      if i==rm: continue
      if i> rm: j = i-1
      else: j=i
      self.house_row(i, j)
    for j in self.colrange: self.temprow[j] = k*self.row[rm][j] #<------- get the scaled version of the row rm
    self.tempcol[rm] = -k #<-----------------------------------------------------------store it for the rowmix
    for i in range(self.rows-1, -1, -1): #<------------------------- do least squares reduction on the temprow
      if i==rm: continue #<---------------------------------------------------------- without using the rm row
      if i>rm: j = i-1
      else: j=i
      m = int((self.temprow[j]/self.row[i][j] + 0.5)//1.0) #<--------------------------- the reduction integer
      #m = self.temprow[j]/self.row[i][j] #<------------------------- use this to test the exact least squares
      self.tempcol[i] = m
      if m==0: continue #<----------------------------- if it already is reduced up to an interger combination
      for g in self.colrange:
        self.temprow[g] -= m*self.row[i][g]
    return self.tempcol[:]
  def rowsort(self,y=None,dim=None, key = lambda x: x):
    '''row sort the cubint, based on elements of y'''
    if dim==None: dim=self.reddim
    if y==None: #y is rownorms
      t=self.norm() # self.tempcol now has the rownorms
      y = []
      for i in range(dim):
        y.append(self.tempcol[i])
    order=list(range(dim))
    reverse=list(range(dim))
    order.sort(key=lambda x: key(y[x])) #get indexes of sorted order
    reverse.sort(key=lambda x:order[x])
    for i in range(dim):# perform the permutation,
      t=reverse.index(i)      # order is indexes after the sort,
      self.rowswap(i,t)       # so reverse gives the permutation
      reverse[i],reverse[t]=reverse[t],reverse[i]
      #self.tempcol[i], self.tempcol[t] = self.tempcol[t], self.tempcol[i] # sort tempcol also to use if want
  def colzero_pass(self,cm=0,rm=None):
    ''' reduce using no divides per-se down the column from rm index if given, from index 0 if rm not given'''
    if rm==None: rm=0
    mn, mni=0,None
    for r in range(rm,self.rows):# find the minimum non-zero element and index mn,mni
      if self.pid[r][cm]==0: continue
      if mni==None or abs(self.pid[r][cm])<abs(mn): mn,mni=self.pid[r][cm],r
    if mni==None: return False
    if mni!=rm:self.rowswap(rm,mni)
    change=False
    for r in range(rm+1,self.rows):
      k=(self.pid[r][cm]*2+self.pid[rm][cm])//(self.pid[rm][cm]*2)
      if k!=0:
        self.row_sub_place(r,rm,k)
        change=True
    return change
  def colzero(self, cm=0, rm=0):
    while self.colzero_pass(rm,cm): continue
    if self.pid[rm][cm] < 0: self.rowneg(rm)
    return
  def dotzero_pass(self,rm=None):
    ''' reduce dot-product with row rm using no divides per-se down from rm index (from index 0 if rm not given)'''
    if rm==None: rm=0
    for r in range(rm+1,self.rows):
      self.tempcol[r] = dot(self.pid[rm], self.pid[r])
    mn, mni=0,None
    for r in range(rm+1,self.rows):# find the minimum non-zero element and index mn,mni
      if self.tempcol[r]==0: continue
      if mni==None or abs(self.tempcol[r])<abs(mn): mn,mni=self.tempcol[r],r
    if mni==None: return False
    if mni!=rm+1:  # put the smallest non-zero in the index just after row rm
      self.rowswap(rm+1,mni)
      self.tempcol[rm+1], self.tempcol[mni] = self.tempcol[mni], self.tempcol[rm+1]
    change=False
    for r in range(rm+2,self.rows): # use the smallest (at rm+1) to reduce the others from rm+2 down
      k=(self.tempcol[r]*2+self.tempcol[rm+1])//(self.tempcol[rm+1]*2)
      if k!=0:
        self.row_sub_place(r,rm+1,k)
        self.tempcol[r] = self.tempcol[r] - k*self.tempcol[rm+1]
        change=True
    return change
  def dotzero(self, rm=0):
    while self.dotzero_pass(rm): continue
    #if self.pid[rm][cm] < 0: self.rowneg(rm)
    return
  def pivot_up(self,rm=0,cm=0):
    ''' reduce using no divides per-se up the column cm from row rm '''
    for r in range(0,rm):
      k=(self.pid[r][cm]*2 + self.pid[rm][cm])//(self.pid[rm][cm]*2)
      if k!=0:
        self.row_sub_place(r,rm,k)
        change=True
    return change
  def smith(self): # previously def diag()
    ''' diagonalize with pivoting, use only integer rowmixing'''
    for i in self.rowrange:
      self.colzero(i,i)
      for r in range(i):
        k=(2*self.pid[r][i] + self.pid[i][i])//(2*self.pid[i][i])
        if k!=0: #self.row_sub_place(r,i,k)
          for g in self.colrange:
            self.pid[r][g] -= k*self.pid[i][g]
    kden=dot(self.pid[self.rows-1],self.pid[self.rows-1]) # for the last row,
    for j in range(0,i): # do a norm reducing mix with the rows above
      knum=dot(self.pid[self.rows-1],self.pid[j])
      k=(2*knum+kden)//(2*kden)
      if k!=0: # self.row_sub_place(j,self.rows-1,k)
        for g in self.colrange:
          self.pid[j][g] -= k*self.pid[self.rows-1][g]
  def diag(self):
    ''' return the elements along the diagonal '''
    y=[]
    for i in self.rowrange: y.append(self.row[i][i])
    return y
  def codiag(self):
    ''' return the elements along the diagonal '''
    y=[]
    for i in self.rowrange: y.append(self.corow[i][i])
    return y
  def det(self):
    ''' multiple of the diagonal values'''
    d=1
    for i in self.rowrange: # TODO: consider if do anything with cols < rows
      d *= self.row[i][i]
    return d
  def idet(self):
    ''' determinant for integer matrix'''
    self.reset()
    self.lq()
    d=1
    for i in self.rowrange: # TODO: consider if do anything with cols < rows
      d *= self.row[i][i]
    return int((d + 0.5)//1.0)
  def codet(self):
    ''' multiple of the diagonal values'''
    d=1
    for i in self.rowrange: # TODO: consider if do anything with cols < rows
      d *= self.corow[i][i]
    return d
  ############################################################################################################
  ####  BEGIN matrix/co-matrix methods
  ############################################################################################################
  def xcocol(self, ir, ic):
    ''' look at the corow matrix to see what integer mixes reduce other corows'''
    rrd = False
    for i in self.rowrange:
      if i==ir or self.corow[ir][ic]==0: continue
      k = int((self.corow[i][ic]/self.corow[ir][ic] + 0.5)//1.0) # nearest int of the ratio
      if k!=0:
        rrd = True
        for j in self.colrange: # row_sub_place()
          self.pid[ir][j] += k*self.pid[i][j]
          self.row[ir][j] += k*self.row[i][j]
          self.corow[i][j] -= k*self.corow[ir][j]
    return rrd
  def xcorows(self, disp=False):
    ''' do corow reductions '''
    redd = True
    while redd:
      redd = False
      for i in self.rowrange:
        self.house_corow(i,0)
        if self.xcocol(i,0):
          if disp: print("*", end="")
          redd = True
  ############################################################################################################
  ####  END matrix/co-matrix methods
  ############################################################################################################
  def rowflip(self, dim = None):
    '''reverse row order so that increasing sort can become decreasing'''
    if dim==None: dim=self.rows
    h=dim//2
    for i in range(h):
      self.rowswap(i, dim - i - 1)
  def colflip(self, dim = None):
    '''reverse col order'''
    if dim==None: dim=self.rows # flip the square portion by default
    h=dim//2
    for i in range(h):
      self.colswap(i, dim - i - 1)
  def crest(self):
    ''' store into tempcol the crest independence values for each of the first reddim rows. '''
    if self.reddim > self.cols: self.reddim = self.cols #<- cannot column operate for an overdetermined matrix
    for i in range(self.reddim - 1, -1, -1): #<----------------------------------------------- : * * 0 : -----
      self.house_row(i, self.reddim - 1 - i) #<- perform rowflipped LQ of the first reddim rows: * 0 0 : -----
    for i in range(self.reddim):
      for j in range(i): #<------------------------------------ when j=i-1, want givens(j, reddim-2, reddim-1)
        self.givens(j,self.reddim + j - i - 1, self.reddim + j - i) #<--------- see page 158 of 2013.5.19.1048
      self.tempcol[i] = self.row[i][self.reddim - 1] #<----------- the size of the independent part of the row
  def crest_det(self):
    ''' change diagonal pid values for least determinant, keeping values in {-1, 0, -1} '''
    for i in range(self.rows - 1, -1, -1): #<----------------------------------------------- : * * 0 : -----
      self.house_row(i, self.rows - 1 - i) #<- perform rowflipped LQ of the first reddim rows: * 0 0 : -----
    for i in self.rowrange:
      for j in range(i): #<------------------------------------ when j=i-1, want givens(j, reddim-2, reddim-1)
        self.givens(j,self.rows + j - i - 1, self.rows + j - i) #<--------- see page 158 of 2013.5.19.1048
      t = self.row[i][self.rows - 1] #<----------- the size of the independent part of the row
      u = self.corow[i][self.rows - 1]
      z = t*u
      if self.pid[i][i] != 1:
        diff = 1 - self.pid[i][i]
        test = abs(t + diff*u)
        #if test > 1e-6 and test < abs(t):
        if test < abs(t):
          self.pid[i][i] = 1
          for j in self.colrange:
            self.row[i][j] += diff*self.corow[i][j]
      if self.pid[i][i] != -1:
        diff = -1 - self.pid[i][i]
        test = abs(t + diff*u)
        #if test > 1e-6 and test < abs(t):
        if test < abs(t):
          self.pid[i][i] = -1
          for j in self.colrange:
            self.row[i][j] += diff*self.corow[i][j]
  def gzrow(self, trow=None):
    ''' gradual least squares, looking for least residue at each step.
        zrow() but with looking at the options a-la the crest() at each step.
        Assumption of LQ form '''
    if self.reddim > self.cols: self.reddim = self.cols #<- cannot column operate for an overdetermined matrix
    if trow==None: trow = self.reddim-1
    ddq = [i for i in self.rowrange] # the LQdiagonal, see page 68 of 2013.A.12 TODO: consider self.ddq as the list
    rm = trow-1
    while rm>=0:
      mn = None
      for i in range(rm, -1, -1): # Iterate through each diagonal ddq row index in reverse
        for j in range(i, rm): # swap the diagonal at row i to be at the end to test it
          self.givens(ddq[j + 1], j, j + 1)
          ddq[j], ddq[j+1] = ddq[j+1], ddq[j] #<--------- see page 68 of 2013.A.12
        test = (self.row[trow][rm]%self.row[ddq[rm]][rm])/self.row[ddq[rm]][rm]
        if mn==None or test < mn: mn, mni = test, ddq[rm]
        if mn==None or (1.0-test) < mn: mn, mni = (1.0-test), ddq[rm]
      i = ddq.index(mni) # recall the minimum fractional and its place in the ddq order
      for j in range(i,rm): #<--------------------------------- move to the positon at rm (the end)
        self.givens(ddq[j+1], j, j+1) #<----------------------- so that it may be the next reducer
        ddq[j], ddq[j+1] = ddq[j+1], ddq[j]
      k = int((self.row[trow][rm]/self.row[ddq[rm]][rm] + 0.5)//1.0)
      if k!= 0: self.row_sub_place(trow, ddq[rm], k)
      print("(%d,%d)"%(ddq[rm],k), end=" ")
      rm -= 1 #<------------------------------------------------- then decrease the counter
    return ddq
  def pinch(self, rm=None, block = 0): # TODO: consider extra indexing to avoid rowswapping/sorting, ie LQ in random row order...
    ''' use the "crest wave" to find the smallest row values using previous rows to reduce.
        Basically, it is zrow() but each reduction is chosen to give the smallest possible component'''
    if rm==None:
      rn = self.rownorm()
      i = rn.index(max(rn))
      self.rowswap(self.rows-1, i)
      rm = self.rows - 1
    for dim in range(rm - 1, block-1, -1): # look for the minimum independent component for row rm reduced by row dim
      mn = None # setup to find the minimum
      for i in range(dim, -1, -1): #<- --------------------------------------- ----- : * * 0 : -----
        self.house_row(i, dim - i) #<- perform rowflipped LQ of the first reddim rows: * 0 0 : -----
      for i in range(dim + 1):
        for j in range(i): #<------------------------------------ when j=i-1, want givens(j, reddim-2, reddim-1)
          self.givens(j, dim + j - i, dim + j - i + 1) #<--------- see page 158 of 2013.5.19.1048
        t = self.row[rm][dim]%abs(self.row[i][dim]) #<-- self.row[i][dim] is the independent component of row i
        ta = min(abs(t), abs(t - self.row[i][dim])) # the reduced component value, modded to range (-0.5, 0.5)
        if mn == None or ta < mn:
          mn, mni = ta, i
      self.rowswap(dim, mni)
      for i in range(dim):
        self.house_row(i,i)
      k = int((self.row[rm][dim]/self.row[dim][dim] + 0.5)//1.0)
      self.row_sub_place(rm, dim, k)
  def pinch_sort(self, rm, rt = 0): # TODO: consider extra indexing to avoid rowswapping/sorting, ie LQ in random row order...
    ''' use the "crest wave" to find the smallest row values without reducing.
        Insert the row from rm to block to give the smallest possible components'''
    for dim in range(rm - 1, rt-1, -1): # look for the minimum independent component for row rm reduced by row dim
      mn = None # setup to find the minimum
      for i in range(dim, -1, -1): #<- --------------------------------------- ----- : * * 0 : -----
        self.house_row(i, dim - i) #<- perform rowflipped LQ of the first reddim rows: * 0 0 : -----
      for i in range(dim + 1):
        for j in range(i): #<------------------------------------ when j=i-1, want givens(j, reddim-2, reddim-1)
          self.givens(j, dim + j - i, dim + j - i + 1) #<--------- see page 158 of 2013.5.19.1048
        t = self.row[rm][dim] #<-- self.row[i][dim] is the independent component of row i
        ta = abs(t) # the reduced component value, modded to range (-0.5, 0.5)
        if mn == None or ta < mn:
          mn, mni = ta, i
      self.rowswap(dim, mni)
    for i in range(rm, rt-1, -1): #self.rowslide(rm, rt)
      self.rowswap(i, i-1)
    for i in self.rowrange: self.house_row(i,i)
  def entropy(self):
    ''' return the norm magnitude of the maximum possible targeted value'''
    det = 1.0
    for i in range(self.reddim):
      det += (self.row[i][i]/2.0)**2
    return math.sqrt(det)
  def lq(self, dim = None):
    ''' form all rows into LQ, sort the first dim (default reddim) rows for minimum diagonal'''
    if dim == None: dim = self.reddim
    for i in range(dim): # the entire range so that all stays in LQ form
      mn = None # find the minimum next possible in range for the diagonal
      for j in range(i, dim):
        t = 0.0
        for g in range(i, self.cols): t += self.row[j][g]**2
        if mn==None or t < mn: mn, mni = t, j
      if mni != i: #<-- inline of rowswap(i, mni)
        self.pid[i], self.pid[mni] = self.pid[mni], self.pid[i]
        self.row[i], self.row[mni] = self.row[mni], self.row[i]
      self.house_row(i,i)
    for i in range(dim, self.rows): self.house_row(i,i) # do the rest of the rows as well.
  def append(self, newrow):
    ''' add a new row, adjusting various dimensions as necessary, keeping the comatrix and row matrix '''
    lenrow =len(newrow)
    if lenrow < self.cols:
      xrow = newrow[:] + [0 for i in range(self.cols - lenrow)]
    elif lenrow == self.cols:
      xrow = newrow[:]
    else: # now here the entire dimensions of the self need be expanded to fit the new row, **ASSUMING** extra dimension values are non-zero
      print("rlq.append(): dimension extension required, not yet emplemented!")
      return
      #coldiff = lenrow - self.cols
    self.rows += 1
    self.rowrange = range(self.rows)
    self.pid.append(xrow)
    self.row.append(xrow[:]) # will be zeroed then reset in the following loop
    r1 = self.rows-1 # index for the new last row
    for j in self.colrange: # now use the comatrix to calculate the row
      self.row[r1][j] = 0.0
      for i in self.colrange:
        self.row[r1][j] += self.pid[r1][i]*self.corow[i][j]
    self.house_row(r1,r1) # condense the remaining rownorm (LQ form)
    self.temprow = [0.0 for i in self.colrange] # a row of scratch memory to use for calculations
    self.tempcol = [0.0 for i in self.rowrange] # a col of scratch memory to use for calculations
    return self.row[r1][r1]
  def lq_max(self, dim = None):
    ''' form all rows into LQ, sort the first dim (default reddim) rows for maximum diagonal'''
    if dim == None: dim = self.reddim
    for i in range(dim): # the entire range so that all stays in LQ form
      mx = None # find the minimum next possible in range for the diagonal
      for j in range(i, dim):
        t = 0.0
        for g in range(i, self.cols): t += self.row[j][g]**2
        if mx == None or t > mx: mx, mxi = t, j
      if mxi != i: #<-- inline of rowswap(i, mni)
        self.pid[i], self.pid[mxi] = self.pid[mxi], self.pid[i]
        self.row[i], self.row[mxi] = self.row[mxi], self.row[i]
      self.house_row(i,i)
    for i in range(dim, self.rows): self.house_row(i,i) # do the rest of the rows as well.
  def lqred(self, disp = False):
    ''' form the lq but with reductions also '''
    change = False
    i = 0
    while i < self.rows:
      mn = None # find the minimum next possible in the diagonal
      for j in range(i, self.rows):
        self.tempcol[i] = 0.0 # the row norm squared
        for g in range(i, self.cols): self.tempcol[i] += self.row[j][g] * self.row[j][g]
        if mn == None or self.tempcol[i] < mn: mn, mni = self.tempcol[i], j
      if mni != i: #<-- inline of rowswap(i, mni)
        self.pid[i], self.pid[mni] = self.pid[mni], self.pid[i]
        self.row[i], self.row[mni] = self.row[mni], self.row[i]
        change = True
      self.house_row(i,i)
      if self.xcol(i,i):
        i = 0
        if disp: print(".", end="")
        continue
      else:
        i += 1
        if disp: print("*", end="")
      for j in range(i-1, -1, -1): self.xcol(j,j)
      self.reset() #TODO: inline this, and use return values of xcol() to avoid unneeded reset...
    return change
  def dig(self, quality = 1.6):
    ''' reduce the worst 2-dimensional sub-lattice.
        Assumption of LQ form'''
    drat = [abs(self.row[i][i]/self.row[i+1][i+1]) for i in range(self.rows-1)]
    mx = max(drat)
    while mx > quality:
      mi = drat.index(mx)
      #print(int(mx), end = " ")
      #print(mi, end = " ")
      t = self.row[mi + 1][mi]/self.row[mi][mi]
      if abs(abs(t) - 0.5 ) < 1e-6: k = 0# if the possible magnitude change is negligible
      else: k = int((t + 0.5)//1.0)
      while k != 0:
        if k > 765390:
          print("--", k)
          return
        print(k, end=" ")
        for g in self.colrange:# self.row_sub_place(mi+1, mi, k) without the row update
          self.pid[mi+1][g] -= k*self.pid[mi][g]
        self.row[mi+1][mi] = 0.0 # prepare to recalculate
        for g in self.colrange:
          self.row[mi+1][mi] += float(self.pid[mi + 1][g])*self.corow[g][mi]
        for j in range(mi-1, -1, -1):    #<-- then also reduce the now changed values
          self.row[mi + 1][j] = 0.0
          for g in self.colrange:         #<-- make precise the pertinent row value
            self.row[mi+1][j] += float(self.pid[mi+1][g])*self.corow[g][j]
          t = self.row[mi+1][j]/self.row[j][j]
          if abs(abs(t) - 0.5) < 1e-6: continue # if the reduction is negligible
          k = int((t + 0.5)//1.0)
          if k == 0: continue
          for g in self.colrange:
            self.pid[mi+1][g] -= k*self.pid[j][g]
          self.row[mi + 1][j] = 0.0
          for g in self.colrange:         #<-- re-calculate the row value
            self.row[mi+1][j] += float(self.pid[mi+1][g])*self.corow[g][j]
        if self.row[mi][mi]**2 > self.row[mi + 1][mi]**2 + self.row[mi + 1][mi + 1]**2:
          self.givens(mi + 1, mi, mi+1)
          self.pid[mi], self.pid[mi+1] = self.pid[mi+1], self.pid[mi] # TODO: implement indexed LQ
          self.row[mi], self.row[mi+1] = self.row[mi+1], self.row[mi]
          t = self.row[mi + 1][mi]/self.row[mi][mi]
          if abs(abs(t) - 0.5 ) < 1e-6: k = 0# if the possible magnitude change is negligible
          else: k = int((t + 0.5)//1.0)
        else: k=0
      if mi > 0: drat[mi - 1] = abs(self.row[mi-1][mi-1]/self.row[mi][mi])
      drat[mi] = abs(self.row[mi][mi]/self.row[mi+1][mi+1])
      if mi < self.rows - 2:drat[mi + 1] = abs(self.row[mi + 1][mi + 1]/self.row[mi + 2][mi + 2])
      mx = max(drat)
  def digest(self, start=0):
    ''' look at adjacent diagonal values of the LQ form and mix for larger values at the lower end.
        Assumption: row[] is in lq form'''
    change = False
    i = start + 1
    while i < self.reddim: #<- look at each 2-dimensional sub-lattice on the diagonal
      a = self.row[i-1][i-1]    #<---  | a 0 |
      e = self.row[i][i-1]      #<---  | e f |
      f = self.row[i][i]        #<--- variables renamed only for reading and typing convenience.
      if abs(a) < 1e-6: # ignore values smaller than epsilon
        i += 1
        continue
      t = e/a
      if abs(abs(t) - 0.5 ) < 1e-6: # if the possible magnitude change is negligible
        i += 1
        continue
      k = int((t + 0.5)//1.0)
      if k!=0: # if there is a reduction
        for g in self.colrange: # self.row_sub_place(i, i - 1, k) without row[] update
          self.pid[i][g] -= k*self.pid[i-1][g]
        self.row[i][i-1] = 0.0            # prepare to recalculate row[i][i-1]
        for g in self.colrange:         #<-- update the pertinent row[] value for precision
          self.row[i][i-1] += float(self.pid[i][g])*self.corow[g][i-1]
        for j in range(i-2, -1, -1):    #<-- then also reduce the now changed values
          self.row[i][j] = 0.0
          for g in self.colrange:         #<-- make precise the pertinent row value
            self.row[i][j] += float(self.pid[i][g])*self.corow[g][j]
          t = self.row[i][j]/self.row[j][j]
          if abs(abs(t) - 0.5) < 1e-6: continue # if the reduction is negligible
          k = int((t + 0.5)//1.0)
          if k == 0: continue # if there is no reduction at all
          for g in self.colrange: #self.row_sub_place(i, j, k)
            self.pid[i][g] -= k*self.pid[j][g]
          self.row[i][j] = 0.0            #<---------------------------------------------===
          for g in self.colrange:         #<-- update the changed row value for precision ==
            self.row[i][j] += float(self.pid[i][g])*self.corow[g][j] #<------------------===
        e = self.row[i][i-1]
      if a*a - (e*e + f*f) > 1e-6: #<- if the reductions found a significantly smaller "upper-low" value
        change = True
        self.givens(i, i - 1, i)#inline of rowslide(i, i-1) which preserves the LQ form
        self.pid[i], self.pid[i - 1] = self.pid[i - 1], self.pid[i]
        self.row[i], self.row[i - 1] = self.row[i - 1], self.row[i]
        #i = max(1, i-1)
      else: i += 1
    return change
  def digall(self, quality = 1.65):
    ''' digest() in a loop. Quality limits to (greater than) 2/math.sqrt(3) '''
    count = 0
    rred = True
    while rred:
      rred = False
      self.reset(True) # reset(True) includes lq()
      dratio = []
      for i in range(1, self.reddim):
        if abs(self.row[i][i]) < 1e-6: # there is no discernable independence to the row
          dratio.append(0.0)
        else:
          dratio.append(abs(self.row[i-1][i-1]/self.row[i][i]))
      while max(dratio) > quality:
        while self.digest(): continue
        self.reset(True)
        for i in range(1, self.reddim):
          if abs(self.row[i][i]) < 1e-6: # no discernable independence
            dratio[i-1] = 0.0
          else:
            dratio[i-1] = abs(self.row[i-1][i-1]/self.row[i][i])
        count += 1
    return count
  def digallinc(self, dim = None, disp = False, quality = 1.65):
    ''' digall() from reddim = 2 up to rowrange'''
    if dim==None: dim = self.rows
    if dim < 2: return 0
    self.reddim = 2
    count = self.digall(quality)
    self.reddim = 3
    while self.reddim <= dim:
      while self.zrow(self.reddim-1): continue
      count += self.digall(quality)
      self.reddim +=1
      if disp: print(self.reddim%10, end="")
    self.reddim = self.rows
    return count
  def enum(self, trow, dim = None):
    ''' enumerate all reduction possibilities for rowmix to row trow, using first dim rows.
        A reduction is relative to the target row, any smaller possibility than the target'''
    if trow >= self.rows: return False #<- target row range checking
    if dim==None: dim = trow
    d0 = 0
    for j in self.colrange: d0 += self.pid[trow][j]**2 #<- get the rownorm to test reduction later
    if dim != trow: # if rows beyond trow are being used, then swap trow to the end of the dim rows
      swapped = True
      self.rowswap(dim - 1, trow)
      original_trow = trow
      trow = dim - 1
    else: swapped = False #TODO: a rowslide may be better in case a certain row order is desired.
    for i in range(trow+1):
      for j in self.colrange:
        self.row[i][j] = float(self.pid[i][j]) # reset up to and including trow
    for i in range(trow - 1, -1, -1): #<----------------------------------------------------- : 0 * * 0 0 0 0 0
      self.house_row(i, trow - 1 - i) #<------- perform rowflipped LQ of the first reddim rows: 0 0 * 0 0 0 0 0
    #< do colflip() of pertinent columns so that the reddim-1 row may be targeted in row order: * * * * 0 0 0 0
    h=trow//2
    for i in range(h): self.colswap(i,trow-i-1) # TODO: consider house_row_left() to shape directly to the desired form
    self.house_row(trow, trow)
    rnorm = 0.0 # the norm of the in span portion
    for j in range(trow ): rnorm += float(self.row[trow][j]**2) # get the current norm of the in-span portion of the target row
    poss = [] #<-- the list of possible rowmixes that, up to the point of definition, are less than rnorm
    k = math.floor(self.row[trow][0]/self.row[0][0]) # the floor
    k0 = k
    t0 = (self.row[trow][0] - k0*self.row[0][0])**2
    while t0 < rnorm:
      poss.append([t0,[k0],1]) # put the norm value, rowmix, and mix-length in the possibles list
      k0-=1
      t0 = (self.row[trow][0] - k0*self.row[0][0])**2
    k1 = k + 1
    t1 = (self.row[trow][0] - k1*self.row[0][0])**2
    while t1 <  rnorm:
      poss.append([t1,[k1],1]) # put the norm value, rowmix, and mix-length in the possibles list
      k1+=1
      t1 = (self.row[trow][0] - k1*self.row[0][0])**2
    while poss != []:
      #[tp,kvec,rp] = min(poss, key=lambda x: x[0]) # breadth first search
      [tp,kvec,rp] = max(poss, key=lambda x: x[2]) # depth first search
      mni = poss.index([tp,kvec,rp]) # and the index
      temp = poss.pop(mni) # remove it
      self.temprow[rp] = self.row[trow][rp] # copy the pertinent portion of the target again
      for i in range(rp): # and perform the kvec rowmix on it
        self.temprow[rp] -= kvec[i]*self.row[i][rp] # perform the rowmix at the pertinent index
      k = math.floor(self.temprow[rp]/self.row[rp][rp])
      k0 = k
      t0 = tp + (self.temprow[rp] - k0*self.row[rp][rp])**2
      if t0<rnorm and rp+1==trow: # if a reducing mix has been found
        for i in range(trow-1): self.row_sub_place(trow,i,kvec[i])
        self.row_sub_place(trow, trow-1, k0)
        if dim != trow: self.rowswap(dim,trow)
        if swapped: self.rowswap(original_trow, dim - 1) # prepare for the return next
        if dot(self.pid[trow], self.pid[trow]) < d0: return True
        else: return False
      while t0 < rnorm:
        poss.append([t0, kvec[:] + [k0], rp + 1])
        k0 -= 1
        t0 = tp + (self.temprow[rp] - k0*self.row[rp][rp])**2
      k1 = k + 1
      t1 = tp + (self.temprow[rp] - k1*self.row[rp][rp])**2
      if t1<rnorm and rp+1==trow: # if a reducing mix has been found
        for i in range(trow-1): self.row_sub_place(trow,i,kvec[i])
        self.row_sub_place(trow, trow-1, k1)
        if dim != trow: self.rowswap(dim,trow)
        if swapped: self.rowswap(original_trow, dim - 1) # prepare for the return next
        if dot(self.pid[trow], self.pid[trow]) < d0: return True
        else: return False
      while t1 < rnorm:
        poss.append([t1, kvec[:] + [k1], rp + 1])
        k1 += 1
        t1 = tp + (self.temprow[rp] - k1*self.row[rp][rp])**2
    if swapped: self.rowswap(original_trow, dim - 1) # prepare for the return next
    return False
  def enum_full(self, rm = None, dim = 30, quality = 1.3, maxtries = 10000):
    ''' enumerate to find a smaller basis vector (smaller diagonal in the LQ) by ratio of quality.
        LQ form is assumed, with row 0 (or rm - dim) the smallest vector.
        dim is the largest sub-lattice to try'''
    if rm==None: rm = self.reddim-1
    mindim = max(0, rm - dim) # the first row of the block
    poss = [] # the list of possibilities to try for rowmixing TODO: keep poss sorted for (maybe?) quicker search
    count = 0 # the size of the poss list
    limited = False # if the search had to be pruned to keep under maxtries
    dn = [ self.row[i][i]**2 for i in range(rm + 1)] # the norms of the diagonals
    box = [ 0.0 for i in range(rm + 1)] # The boxed max possible, to enable the sorting possibilities
    box[rm] = dn[rm]
    for i in range(rm-1, -1, -1): box[i] = dn[i]/4.0 + box[i+1]
    rm_current = -1.0 # find if the row already is smaller, in which case that is the improvement goal
    for i in range(mindim, rm+1): rm_current += self.row[rm][i]**2
    thresh = min(dn[mindim], rm_current)/quality
    k = self.row[rm][rm]
    while k**2 < thresh:
      count += 1
      poss.append([(k**2)/box[rm], [float(count)], rm]) # append the norm relative to the box for sorted search
      k += self.row[rm][rm]
    maxcount = count
    while poss != []:
      [tp, kvec, rp] = min(poss, key = lambda x: x[0])
      mni = poss.index([tp, kvec, rp])
      temp = poss.pop(mni)
      count -= 1
      t0 = tp*box[rp] # the norm squared up to that point
      bb = 0.0 # TODO: need check rp - 1 >= mindim ? Or just not push it to poss[]...
      rnext = rp-1
      for i in range(rp, rm + 1): bb += kvec[i - rp]*self.row[i][rnext] # the rowmixed row[rm][rp] value
      t = -bb/self.row[rnext][rnext]
      k_first = (t + 0.5)//1.0
      if rnext == mindim: #if it is the last rowmix to check
        if limited: print("enum_full(): Limited is", limited) #TODO: here pop the largest in poss to push the new one
        k = k_first
        if t0 + (bb + k*self.row[rnext][rnext])**2 < thresh:
          mix = [0 for i in range(mindim)] + [int(k)] + [int(kvec[i]) for i in range(len(kvec))]
          if mix[rm] == 1: # -1 is not tried, it is redundant
            for i in range(mindim, rm): self.row_sub_place(rm, i, -mix[i])
            for i in range(mindim-1, -1, -1): # do a zrow() on the rest of the row rm to reduce
              k = int((self.row[rm][i]/self.row[i][i] + 0.5)//1.0)
              self.row_sub_place(rm, i, k)
            return True
          else:
            print("enum_full(): Found vec =", mix)
            return False
      else: # if it is not the last rowmix to check
        k_floor = t//1.0
        sig = 2.0*(k_floor - k_first) + 1.0 # 1.0 if floor is nearest, -1.0 if it is not
        i = 0 # iterate k_first, k_first + sig, k_first - sig, k_first + 2*sig, k_first - 2*sig...
        k = k_first
        while t0 + (bb + k*self.row[rnext][rnext])**2 < thresh:
          if count >= maxtries:
            if limited==False:
              print("enum_full(): Setting limited!")
              limited = True
            break
          poss.append([(t0 + (bb + k*self.row[rnext][rnext])**2)/box[rnext], [k] + kvec[:], rnext])
          count += 1
          if i>0: i = -i
          else: i = -i + 1
          k = k_first + i*sig
      if count > maxcount: maxcount = count
    if rm > mindim: return self.enum_full(rm-1, dim-1, maxtries)
    return maxcount
  def enum_unit(self, trow, dim = None):
    ''' enumerate all reduction possibilities for rowmix to row trow,
        using first dim rows and mix values from in {-1,0,1} only'''
    if trow >= self.rows: return False #<- target row range checking
    if dim==None: dim = trow
    d0 = 0
    for j in self.colrange: d0 += self.pid[trow][j]**2 #<- get the rownorm to test reduction later
    if dim != trow: # if rows beyond trow are being used, then swap trow to the end of the dim rows
      swapped = True
      self.rowswap(dim - 1, trow)
      original_trow = trow
      trow = dim - 1
    else: swapped = False #TODO: a rowslide may be better in case a certain row order is desired.
    for i in range(trow+1):
      for j in self.colrange:
        self.row[i][j] = float(self.pid[i][j]) # reset up to and including trow
    for i in range(trow - 1, -1, -1): #<----------------------------------------------------- : 0 * * 0 0 0 0 0
      self.house_row(i, trow - 1 - i) #<------- perform rowflipped LQ of the first reddim rows: 0 0 * 0 0 0 0 0
    #< do colflip() of pertinent columns so that the reddim-1 row may be targeted in row order: * * * * 0 0 0 0
    h=trow//2
    for i in range(h): self.colswap(i,trow-i-1) # TODO: consider house_row_left() to shape directly to the desired form
    self.house_row(trow, trow)
    rnorm = 0.0 # the norm of the in span portion
    for j in range(trow ): rnorm += float(self.row[trow][j]**2) # get the current norm of the in-span portion of the target row
    poss = [] #<-- the list of possible rowmixes that, up to the point of definition, are less than rnorm
    k = math.floor(self.row[trow][0]/self.row[0][0]) # the floor
    k0 = k
    t0 = (self.row[trow][0] - k0*self.row[0][0])**2
    if t0< rnorm: poss.append([t0,[k0],1]) # put the norm value, rowmix, and mix-length in the possibles list
    k1 = k + 1
    t1 = (self.row[trow][0] - k1*self.row[0][0])**2
    if t1 < rnorm: poss.append([t1,[k1],1]) # put the norm value, rowmix, and mix-length in the possibles list
    while poss != []:
      #[tp,kvec,rp] = min(poss, key=lambda x: x[0]) # breadth first search
      [tp,kvec,rp] = max(poss, key=lambda x: x[2]) # depth first search
      mni = poss.index([tp,kvec,rp]) # and the index
      temp = poss.pop(mni) # remove it
      self.temprow[rp] = self.row[trow][rp] # copy the pertinent portion of the target again
      for i in range(rp): # and perform the kvec rowmix on it
        self.temprow[rp] -= kvec[i]*self.row[i][rp] # perform the rowmix at the pertinent index
      k = math.floor(self.temprow[rp]/self.row[rp][rp])
      k0 = k
      t0 = tp + (self.temprow[rp] - k0*self.row[rp][rp])**2
      if t0<rnorm and rp+1==trow: # if a reducing mix has been found
        for i in range(trow-1): self.row_sub_place(trow,i,kvec[i])
        self.row_sub_place(trow, trow-1, k0)
        if dim != trow: self.rowswap(dim,trow)
        if swapped: self.rowswap(original_trow, dim - 1) # prepare for the return next
        if dot(self.pid[trow], self.pid[trow]) < d0: return True
        else: return False
      if t0 < rnorm: poss.append([t0, kvec[:] + [k0], rp + 1])
      k1 = k + 1
      t1 = tp + (self.temprow[rp] - k1*self.row[rp][rp])**2
      if t1<rnorm and rp+1==trow: # if a reducing mix has been found
        for i in range(trow-1): self.row_sub_place(trow,i,kvec[i])
        self.row_sub_place(trow, trow-1, k1)
        if dim != trow: self.rowswap(dim,trow)
        if swapped: self.rowswap(original_trow, dim - 1) # prepare for the return next
        if dot(self.pid[trow], self.pid[trow]) < d0: return True
        else: return False
      if t1 < rnorm: poss.append([t1, kvec[:] + [k1], rp + 1])
    if swapped: self.rowswap(original_trow, dim - 1) # prepare for the return next
    return False
  def lq_target(self, trow):
    ''' - form the previous rows to target trow in order -: * * * 0 -----
        --------------------------------------------------: 0 * * 0 -----
        --------------------------------------------------: 0 0 * 0 -----
        --------------------------------------------------: * * * * -----
        '''
    if trow >= self.rows or trow<0: trow = self.rows-1 #<- target row range checking
    preddim = self.reddim # preserve the reddim
    self.reddim = trow + 1 # set so that reset() is for all pertinent rows
    self.reset(False)
    self.reddim = trow # set so that the last row stays there even with the re-ordering in lq()
    self.lq() # and form entropic LQ (smallest possible diagonal at each step)
    self.reddim = trows + 1 # put back to all pertinent rows
    self.rowflip(dim = trow) #<----------------------------------------------- : * * 0 0
    self.colflip(dim = trow) #<- perform rowflipped LQ of the first reddim rows: 0 * 0 0
    self.reddim = preddim # restore the reddim
  def enum_unit_block(self, trow, block = 30):
    ''' enumerate all reduction possibilities for rowmix to row trow,
        using first dim rows and mix values from in {-1,0,1} only.
        A reduction is relative to the diagonal, any smaller diagonal.
        Test norm against diagonal/spine instead of against self.
        Stop branching the search at block number of rows'''
    #self.lq_max(trow) # shape row[] to LQ with maximum sorted order, preserving trow's position
    if trow <= 0: return 0 # no higher up than that is possible
    pnorm = 0.0 # prepare to calculate the norm of the out-of-span portion of trow
    for j in range(trow, self.cols): pnorm += float(self.row[trow][j]**2) # get the norm of out-of-span portion
    rp = trow - 1
    k = math.floor(self.row[trow][rp]/self.row[rp][rp]) # the floor, so that k+1 is the ceil
    k0 = k
    k1 = k + 1
    t0 = (self.row[trow][rp] - k0*self.row[rp][rp])**2
    t1 = (self.row[trow][rp] - k1*self.row[rp][rp])**2
    if t0 + pnorm < self.row[rp][rp]**2:
      self.row_sub_place(trow, rp, k0)
      self.rowswap(trow, trow - 1)
      return rp
    elif t1 + pnorm < self.row[rp][rp]**2:
      self.row_sub_place(trow, rp, k1)
      self.rowswap(trow, rp)
      return rp
    else:
      poss = []
      poss.append([t0,[k0],1]) # stack the in-span norm value, rowmix, and mix-length on the list
      poss.append([t1,[k1],1]) # stack the in-span norm value, rowmix, and mix-length on the list
    while poss != []:
      #[tp,kvec,rp] = min(poss, key=lambda x: x[0]) # breadth first search
      [tis, kvec, length] = max(poss, key=lambda x: x[2]) # depth first search
      g = rp - length
      mni = poss.index([tis,kvec, g]) # and the index
      temp = poss.pop(mni) # remove it
      self.temprow[g] = self.row[trow][g] # copy the pertinent portion of the target again
      for i in range(length): # and perform the kvec rowmix on it
        self.temprow[g] -= kvec[i]*self.row[i][g] # perform the rowmix at the pertinent index
      k = math.floor(self.temprow[g]/self.row[g][g])
      k0 = k
      k1 = k + 1
      t0 = tis + (self.temprow[g] - k0*self.row[g][g])**2
      if t0 + pnorm < self.row[g][g]**2: # if a new reduced diagonal has been found
        for i in range(length): self.row_sub_place(trow, i, kvec[i])
        self.row_sub_place(trow, g, k0)
        if dim != trow: self.rowswap(dim,trow)
        if dot(self.pid[rm], self.pid[rm]) < d0: return True
        else: return False
      if t0 < rnorm: poss.append([t0, kvec[:] + [k0], rp + 1])
      t1 = tis + (self.temprow[rp] - k1*self.row[rp][rp])**2
      if t1 < rnorm and rp+1==rm: # if a reducing mix has been found
        for i in range(rm-1): self.row_sub_place(rm,i,kvec[i])
        self.row_sub_place(rm, rm-1, k1)
        if dim != rm: self.rowswap(dim,rm)
        if dot(self.pid[trow], self.pid[trow]) < d0: return True
        else: return False
      if t1 < rnorm: poss.append([t1, kvec[:] + [k1], rp + 1])
    return False
  def ered(self, disp = True): # TODO: use and maintain the lq_min2() form
    ''' use enum and sorting of rows' rownorms to reduce,
        incrementing dimension on enum until reductions cease'''
    rn = self.rownorm()
    self.rowsort(rn) #<-- sort the rows by rownorm
    rn = self.rownorm() #<- store the rownorms to track them
    redd = True #<- intitialize for the while loop
    i = 1 # start the reductions at the first two rows
    while redd:
      while i < self.reddim and not self.enum(i): i += 1
      if i == self.reddim: redd = False
      else: #<- if a reduction occured, reorder the rows to keep sorted
        if disp:print(i, end=" ")
        rn[i] = dot(self.pid[i], self.pid[i])
        while i > 0 and rn[i] < rn[i-1]:
          self.rowswap(i,i-1)
          rn[i], rn[i-1] = rn[i-1], rn[i]
          i-=1
        if i == 0: i=1 # if the new row became the smallest, put i back to 1 not 0
  def thresh(self, t=1.0e-6):
    ''' all elements less than value in magnitude are zeroed'''
    for i in self.rowrange:
      for j in self.colrange:
        if abs(self.row[i][j])<t: self.row[i][j]=0
  def rowamax(self):
    ''' return the abs max elements of each row'''
    ar=[]
    for i in self.rowrange: ar.append(abs(max(self.pid[i],key=lambda x: abs(x))))
    return ar
  def rownorm(self): # TODO: use tempcol and return a copy, so that values may be recalled.
    ''' return the norms of each row'''
    ar=[]
    for i in range(self.reddim): ar.append(dot(self.pid[i], self.pid[i]))
    return ar
  def qrownorm(self, c=0):
    ''' the rownorm for rows starting at column c of the row (not the pid) matrix'''
    ar = []
    for i in self.rowrange:
      t = 0.0
      for j in range(c, self.colrange):
        t += self.row[i][j]*self.row[i][j]
      ar.append(t)
    return ar
  def corownorm(self):
    ''' return the norms of each corow--basically a test to make sure all are 1.0'''
    ar=[]
    for i in self.rowrange: ar.append(dot(self.corow[i], self.corow[i]))
    return ar
  def cocolnorm(self):
    ''' return the norms of each cocol--basically a test to make sure all are 1.0'''
    for j in self.colrange:
      self.temprow[j] = 0.0
      for i in self.colrange:
        self.temprow[j] += self.corow[i][j]*self.corow[i][j]
    return self.temprow[:]
  ############################################################################################################
  ####  rowmix/co-rowmix vector targetting functions
  ############################################################################################################
  def rowmix(self,rm,a,dim=None): # TODO: consider could do some sort of *= notation if could denote the row to change
    ''' use the constant list to combine with other rows'''  # TODO: such as m[0]*= vector rowmix
    if dim==None:dim=min(self.rows,len(a)) # so can input shorter row vector if omission is taken as zero
    rm=rm%self.cols
    for i in range(dim):
      if i==rm or a[i]==0:continue
      self.row_sub_place(rm,i,-a[i])
  def write_pid(self, filename):
    ''' write the integers to a file '''
    pfile = open(filename, "w")
    pfile.write( self.rows.__repr__())
    pfile.write("\n")
    pfile.write( self.cols.__repr__())
    pfile.write("\n")
    for i in self.rowrange:
      for j in self.colrange:
        pfile.write( self.pid[i][j].__repr__())
        pfile.write(" ")
      pfile.write("\n")
    pfile.close()
  def read_pid(self, filename):
    ''' read the integers from a file '''
    pfile = open(filename, "r")
    r = int(pfile.read(1))
    newline = pfile.read(1)
    c = int(pfile.read(1))
    newline = pfile.read(1)
    for i in self.rowrange:
      for j in self.colrange:
        self.pid[i][j] = pfile.read(1)
        space = pfile.read(1)
      pfile.read("\n")
    pfile.close()
  ############################################################################################################
  ####  built in functions
  ############################################################################################################
  def __call__(self,r1,r2,k=1): # shortcut for row_sub_place
    if k==0 or r1==r2: return
    for i in self.colrange:
      self.pid[r1][i]=self.pid[r1][i]-k*self.pid[r2][i]
      self.row[r1][i]=self.row[r1][i]-k*self.row[r2][i]
  def __len__(self):
    return len(self.pid)
  def htmlSection(self):
  	'''  '''
  	webstring = '''
  		<section>
  		<div>
  		<div style="color:rgba(5,148,10,1);">
  		<math xmlns='http://www.w3.org/1998/Math/MathML'>
  			'''
  	webstring += self.htmlString()
  	webstring += '''
  		</math>
  		</div>
  		</div>
  		</section>
  		'''
  	return webstring
  def htmlString(self):
  	pstr = [] # 
  	cwidth=[0 for i in self.colrange]
  	hs = "<mrow>"
  	hs += "<mfenced open='[' close=']' separators=''>"
  	hs += "<mtable>"
  	for i in self.rowrange: # do column first in order to mark if there are negative symbols to align
  		hasNeg = False
  		pstr.append([])
  		for j in self.colrange:
  			x=self.pid[i][j].__repr__()
  			lxl = len(x)
  			if lxl > cwidth[j]:
  				cwidth[j] = lxl
  			pstr[i].append(x)
  			#if not hasNeg and self.pid[i][j] < 0
  	#str=""
  	for i in range(self.rows):
  		for j in prange:
  			diff=cwidth[j]-len(pstr[i][j])
  			for d in range(diff): str+=" "
  			str+=pstr[i][j]
  			str+=sep
  		str+=end
  	return str
  	hs += "</mtable>"
  	hs += "</mfenced>"
  	hs += "</mrow>"
  	return hs
  def __repr__(self,sep=" ",end="\n"): # 1 for PID   2 for ROW   3 for PID, ROW    4 for COROW
    if self.disp==0: return ""
    if self.disp == 1: k=1# TODO: number of one bits determines the multiple for the range, they are flags
    if self.disp == 2: k=1# TODO: number of one bits determines the multiple for the range, they are flags
    if self.disp == 3: k=2# TODO: number of one bits determines the multiple for the range, they are flags
    if self.disp == 4: k=1# TODO: number of one bits determines the multiple for the range, they are flags
    cwidth=[0 for i in range(self.cols*k)]
    prange = range(self.cols*k)
    pstr=[]
    if self.disp == 4: # the corow is square, do it separately
    	for i in self.colrange:
    		pstr.append([])
    		for j in self.colrange:
    			x = self.corow[i][j].__repr__()
    			lxl = len(x)
    			if lxl > cwidth[j]: cwidth[j] = lxl
    			pstr[i].append(x)
    	str=""
    	for i in self.colrange:
    		for j in prange:
    			diff=cwidth[j]-len(pstr[i][j])
    			for d in range(diff): str+=" "
    			str+=pstr[i][j]
    			str+=sep
    		str+=end
    	return str
    for i in range(self.rows):
      pstr.append([])
      if self.disp & 1 == 1: # TODO: rather than doing self.thresh() a lot, threshhold on the printed numbers...
        for j in range(self.cols):
          x=self.pid[i][j].__repr__()
          lxl = len(x)
          if lxl > cwidth[j]: cwidth[j] = lxl
          pstr[i].append(x)
      if self.disp == 2: # do not do the call matrix if the flag does not denote such
        for j in range(self.cols):
          x=self.row[i][j].__repr__()
          lxl = len(x)
          if lxl > cwidth[j]: cwidth[j] = lxl
          pstr[i].append(x)
      if self.disp == 3: # do not do the call matrix if the flag does not denote such
        for j in range(self.cols):
          x=self.row[i][j].__repr__()
          lxl = len(x)
          if lxl > cwidth[j + self.cols]: cwidth[j + self.cols] = lxl
          pstr[i].append(x)
      if self.disp == 4: # do not do the call matrix if the flag does not denote such
        for j in range(self.cols):
          x=self.corow[i][j].__repr__()
          lxl = len(x)
          if lxl > cwidth[j]: cwidth[j] = lxl
          pstr[i].append(x)
    str=""
    for i in range(self.rows):
      for j in prange:
        diff=cwidth[j]-len(pstr[i][j])
        for d in range(diff): str+=" "
        str+=pstr[i][j]
        str+=sep
      str+=end
    return str
  def math_string(self,digits=None):
    if digits==None:
      digits=12
    str="\\pmatrix{"
    for i in range(self.rows):
      for j in range(self.cols):
        str+=(int(self.pid[i][j]  *10**digits)/(10**digits)).__repr__()
        str+=" & "
      str+=" \\\\"
    return str + "}"
  def math_latex(self):
    str="\\pmatrix{"
    for i in range(self.rows):
      for j in range(self.cols):
        str+=(self.pid[i][j]).__repr__()
        str+=" & "
      str+=" \\\\"
    return str + "}"
  def math_star(self,thresh = 1.0):
    str="\\pmatrix{"
    for i in range(self.rows):
      for j in range(self.cols):
        if abs(self.pid[i][j]) < thresh: str+= "0"
        else: str+="*"
        str+=" & "
      str+=" \\\\"
    return str + "}"
  def print_red(self):
    ''' print the row matrix showing where component reductions may happen'''
    str = ""
    for i in self.rowrange:
      for j in self.colrange:
        if j == i: str += "."
        elif j > i: str += " "
        elif abs(self.row[i][j]*2) > abs(self.row[j][j]): str += "*"
        else: str += "-"
      str += "\n"
    print(str)
  def __mod__(self, n):
    ''' x.__mod__(y)  <==>  x%y  return the element by element mod n'''
    y=rlq(self.rows,self.cols)
    for i in self.rowrange:
      for j in self.colrange:
        y.pid[i][j]=self.pid[i][j]%n
    return y
  def __invert__(self):
    ''' x.__invert__() <==> ~x  returns the transpose of the matrix'''
    y=rlq(self.crank,self.rows) # do only the "base" matrix of rows and crank cols
    for i in y.rowrange:
      for j in y.colrange:
        y.pid[i][j]=self.pid[j][i]
    return y
  def __add__(self,other):
    '''add the values of the two, but return matrix has invalid co-matrix'''
    c=rlq(self.rows,self.cols)
    if not isinstance(other, rlq): # use scaled identity if other is not a rlq
      for i in self.rowrange:
        for j in self.colrange:
          if i==j: c.pid[i][j]=self.pid[i][j]+other
          else: c.pid[i][j]=self.pid[i][j]
      return c
    for i in self.rowrange:
      for j in self.colrange:
        c.row[i][j] = self.pid[i][j]+other.pid[i][j]
    return c
  def __radd__(self,other):
    '''add the values of the two, but return matrix has no valid co-matrix'''
    c=rlq(self.rows,self.cols)
    if not isinstance(other, rlq): # use scaled identity if other is not a rlq
      for i in self.rowrange:
        for j in self.colrange:
          if i==j: c.row[i][j]=self.pid[i][j]+other
          else: c.row[i][j]=self.pid[i][j]
      return c
    for i in self.rowrange:
      for j in self.colrange:
        c.pid[i][j] = self.pid[i][j]+other.pid[i][j]
    return c
  def __sub__(self,other):
    '''subtrac the values of the two, but return matrix has no valid co-matrix'''
    c=rlq(self.rows,self.cols)
    for i in self.rowrange:
      for j in self.colrange:
        c.pid[i][j] = self.pid[i][j]-other.pid[i][j]
    return c
  def __isub__(self, n):
    ''' subtract the identity in place'''
    for i in self.rowrange: # TODO: consider non-square here
      self.pid[i][i]-=n
    return self
  """
  def __imul__(self,other):
    ''' in-place multiply by scalar'''
    for i in self.rowrange:
      for j in self.colrange:
        self.pid[i][j]*=other
    return self
  """
  def __imul__(self,other): # TODO: consider how to do left in place if possible (maybe an "in-place" transpose could help pull that off)
    ''' self = self * other the right multiplication''' # TODO: add type checking...
    for r in self.rowrange:
      for c in self.colrange:
        self.temprow[c]=0 # use the scratch memory until can change the j column all at once
        for j in self.colrange:
          self.temprow[c]+=self.pid[r][j]*other.row[j][c]
      self.pid[r], self.temprow = self.temprow, self.pid[r]
    return self
  def __mul__(self,other):
    '''self*other the single column mix'''
    if isinstance(other,rlq):
      if other.rows!=self.cols:print("__mul__(): dimension mismatch!") # TODO: make this throw error
      y=rlq(self.rows,other.cols)
      for i in range(y.rows):
        for j in range(y.cols):
          s=0
          for mrc in self.colrange:s+=self.pid[i][mrc]*other.pid[mrc][j]
          y.row[i][j]=s
      return y
    y=[]
    dim=min(self.cols,len(other)) # so can input shorter row vector if omission is taken as zero
    for i in self.rowrange:
      s=0
      for j in range(dim):
        s += other[j]*self.pid[i][j]
      y.append(s)
    return y
  def __rmul__(self,other):
    ''' other*self, the single rowmix''' # TODO: implement full matrix multiply
    if not isinstance(other,list): # TODO: if use vector or 1 X 1 matrix...?
      y=rlq(self.rows,self.cols)
      for i in self.rowrange:
        for j in self.colrange:
          y.row[i][j]=other*self.pid[i][j]
      return y
    y=[]
    dim=min(self.rows,len(other)) # so can input shorter row vector if omission is taken as zero
    for j in self.colrange:
      s=0
      for i in range(dim):
        s += other[i]*self.pid[i][j]
      y.append(s)
    return y
  def __floordiv__(self,other):
    '''nearest multiple'''
    if isinstance(other, int): ## if had matlab syntax dot, element by element
      y=rlq(self.rows,self.cols)
      for i in self.rowrange:
        for j in self.colrange:
          y.row[i][j] //= other
      return y
  # has the self.det in it which is not used right now
  """
  def __rtruediv__(self,other): # other/self ==> other times self inverse
    y=[]
    dim=min(len(other),self.cols) # using other is a row list vector right now
    for i in self.colrange:
      s=0
      for j in range(dim):
        s=s+other[j]*(self.row[i][j]*self.det+self.y[i]*self.x[j])
      y.append(s)
    #if self.det!=1:
    #  for i in self.colrange: y[i]=frac(y[i],self.det)
    return y
  def __truediv__(self,other): # self/other ==> self inverse times other # TODO: consider if this the best way to do left vs. right inverse
    y=[]
    dim=min(len(other),self.rows) # using other is a col list vector right now
    for i in self.rowrange:
      s=0
      for j in range(dim):
        s=s+other[j]*(self.row[j][i]*self.det+self.y[j]*self.x[i])
      y.append(s)
    #if self.det!=1:
    #  for i in self.colrange: y[i]=frac(y[i],self.det)
    return y
  """
  def __neg__(self):
    c=rlq(self.rows,self.cols)
    for i in self.rowrange:
      for j in self.colrange:
        c.row[i][j]=-self.pid[i][j]
        c.corow[i][j]=-self.row[i][j]
    return c
  def __getitem__(self,i):
    return self.pid[i]
  def __setitem__(self,a):
    print(a)
  def __abs__(self):
    y=rlq(self.rows,self.cols)
    for i in range(y.rows):
      for j in range(y.cols):
        y.row[i][j]=abs(self.pid[i][j]) # non-linear modification
    return y
  ############################################################################################################
  ############################################################################################################
  ##  Column functions
  ############################################################################################################
  def colswap(self,c1,c2):
    c1,c2=c1%self.cols,c2%self.cols
    if c1==c2: return
    for i in self.rowrange:
      self.row[i][c1], self.row[i][c2] = self.row[i][c2], self.row[i][c1]
    for i in self.colrange:
      self.corow[i][c1],self.corow[i][c2]=self.corow[i][c2],self.corow[i][c1]
  def colneg(self,cm):
    cm=cm%self.cols
    for i in self.rowrange:
      self.row[i][cm]=-self.row[i][cm]
    for i in self.colrange:
      self.corow[i][cm]=-self.corow[i][cm]
  ############################################################################################################
  ##  Row functions
  ############################################################################################################
  def rowswap(self,r1,r2):
    ''' swap the rows while retaining the LQ form '''
    #r1,r2=r1%self.rows,r2%self.rows
    if r1==r2:return
    if r1 < r2: first, last = r1, r2
    else: first, last = r2, r1
    self.pid[first], self.pid[last] = self.pid[last], self.pid[first]
    self.row[first], self.row[last] = self.row[last], self.row[first]
    #for i in range(first, last): # now re-form the LQ
    #  self.givens(first, first, i + 1)
  def rowneg(self,row):
    row=row%self.rows
    for i in self.colrange:
      self.pid[row][i] = -self.pid[row][i]
      self.row[row][i] = -self.row[row][i]
  def row_sub_place(self,r1,r2,k=1): #TODO: check if r1 < r2 since LQ form would be invalidated in such case
    ''' row r1 gets minus k times r2, or
        row r1 becomes  r1 - k*r2, corow r2 becomes  r2 + k*r1'''
    if k==0 or r1==r2: return
    for i in self.colrange:
      self.pid[r1][i] = self.pid[r1][i] - k*self.pid[r2][i]
    for j in self.colrange: # now use the comatrix to recalculate the row to keep precision
      self.row[r1][j] = 0.0
      for i in self.colrange:
        self.row[r1][j] += self.pid[r1][i]*self.corow[i][j]
  def rowslide(self, r1, r2, preserveLQ = True):
    ''' "slide" row at r1 to position r2 while preserving the LQ form '''
    if r1 < r2: #<- if the direction is positive or down to the lower row
      for i in range(r1, r2):
        self.pid[i], self.pid[i + 1] = self.pid[i + 1], self.pid[i]
        self.row[i], self.row[i + 1] = self.row[i + 1], self.row[i]
        if preserveLQ: self.givens(i, i, i + 1)
    elif r2 < r1: #<- if the direction is negative or up to the higher row
      for i in range(r2, r1, -1):
        if preserveLQ: self.givens(i, i - 1, i)
        self.pid[i], self.pid[i - 1] = self.pid[i - 1], self.pid[i]
        self.row[i], self.row[i - 1] = self.row[i - 1], self.row[i]
    else:
      return
    return

############################################################################################################
##  Helper functions
############################################################################################################
