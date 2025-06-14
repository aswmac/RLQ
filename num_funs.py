def gcd(n1, n2=None):
  ''' gcd(n1 [,n2]) if not n2 then n1 should be a list '''
  if n2==None:
    if isinstance(n1,list): # take a vector gcd if only one input of a list
      g=0
      for i in range(len(n1)):
        g=gcd(g,n1[i])
        if g==1: return 1 # cut the calculation short if no more can be calculated
      return g
    elif n1<0: return -n1
    else: return n1
  if 'gcd' in dir(n1): return n1.gcd(n2)
  if 'gcd' in dir(n2): return n2.gcd(n1)
  a,b=n1,n2
  if a<0: a=-a
  if b<0:b=-b
  elif b==0: return a
  while a!=0:
    c=b%a
    if c<a-c:a,b=c,a ## do min. path speedup
    else:a,b=a-c,a  ##could do a,b=a-c,c but it gains nothing surprisingly
  if b>=0:return b
  else: return -b

def dot(list1, list2):
	n1 = len(list1)
	n2 = len(list2)
	if n1 < n2:
		n = n1
	else:
		n = n2
	return sum(x*y for x, y in zip(list1[:n], list2[:n]))
