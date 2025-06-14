##_____________________________________________
##                PRIME NUMBER FUNCTIONS
##---------------------------------------------
import math
#from num_funs import int_root ## int_root is used
import random

def int_root(n,yfun=lambda x:x**2,init=0):
  """
  returns the integer root (inverse) of function,
  obtained by iterating for the solution (thus must
  be an appropriate function). Default function is the
  square, hence returns square root by default
  """
  if isinstance(yfun,int):
    exp=yfun
    yfun=lambda x:x**exp
  [root,p]=[init,1]
  while n-yfun(root)>=0 and p!=0:
    [r,p]=[1,0]
    while n-yfun(root+r*2)>=0:
      r*=2
      p+=1
    root+=r//2 # got next binary digit here
    r=1
  return root+1

def seive(n):
  """
  returns list of primes <= n, by seiving
  """
  primes_list=[]
  rt=[]
  for i in range(n):
    rt.append(i)
  rt[1]=0
  for i in range(2,int(int_root(n)+1)):
    for j in range(2*i, n, i):
      rt[j]=0
  j=0
  for i in range(2,n):
    if (rt[i]!=0):
      primes_list.append(rt[i])
      j=j+1
  return primes_list

def pptest(n):
  """
  Simple implementation of Miller-Rabin test for
  determining probable primehood.
  """
  # if any of the primes is a factor, we're done
  if (n<=50000): ## added this line for to work on small primes
    if n in prime:
      return True
    return False
  if (n<=1):
    return False
  return primetest(n)

def primetest(n):
  """
  Simple implementation of Miller-Rabin test for
  determining probable primehood.
  """
  bases  = [random.randrange(2,50000) for x in range(90)] # random numbers to be used as base tests
  for b in bases:
    if n%b==0: return False        
    tests,s  = 0,0
    m        = n-1
    # turning (n-1) into (2**s) * m
    while not m&1:  # while m is even
      m >>= 1
      s += 1
    for b in bases:
      tests += 1
      isprob = algP(m,s,b,n)
      if not isprob: break            
    #if isprob: return (1-(1./(4**tests)))
    if isprob: return True
    else:      return False

def algP(m,s,b,n):
    """
    based on Algorithm P in Donald Knuth's 'Art of
    Computer Programming' v.2 pg. 395 
    """
    result = 0
    y = pow(b,m,n)    
    for j in range(s):
       if (y==1 and j==0) or (y==n-1):
          result = 1
          break
       y = pow(y,2,n)       
    return result

def find_next_prime(n,f=lambda x:pptest(x),step=2):
  if n%2==0:
    n+=1
  while f(n) == False:
    n+=step
  return n

def find_large_prime(n,f=lambda x:primetest(x),step=2):
  ''' find the next prime of large n'''
  if n%2==0:
    n+=1
  while f(n) == False:
    n+=step
  return n

def primesize(d):
  """ random prime with d decimal digits"""
  return find_large_prime(random.randint(10**(d-1),10**d))

prime=seive(500000)

"""
## define the first few phi(x) (divided by 2)
phi=[0,0,1,1,1,2,1,3,2,3,2,5,2,6,3,4,4,8,3,9,4,\
6,5,11,4,10,6,9,6,14,4,15,8,10,8,12,6,18,9,12,8,\
20,6,0,10,12,11,0,8,21,10,16,12,26,9,20,12,18,0,\
0,0,0,0,18,16,24,10,0,16,22,12]

##00111213232526344839465B4A  696E4F8A8C6I9C8

prime=seive(800000)
"""

