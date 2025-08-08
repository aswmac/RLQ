## takes a lot of digits and treats them as the decimal value of \alpha \in (0, 1)
## multiplies it by numbers 1...n and keeps only the fractional part up to num_digits
## Then checks if the pairs of those can be found thier ratio


from __future__ import print_function
from rlq import * # doing some side tests of things, not used for final answer!
import random
import time
import copy
import prime


# the digits after the decimal of e = 2.7182818284...
estring ='''71828182845904523536028747135266249775724709369995957496696\
76277240766303535475945713821785251664274274663919320030599218174135966\
290435729003342952605956307381323286279434907632338298807531952510190115\
7383418793070215408914993488416750924476146066808226480016847741185374234\
54424371075390777449920695517027618386062613313845830007520449338265602976\
0673711320070932870912744374704723069697720931014169283681902551510865746\
3772111252389784425056953696770785449969967946864454905987931636889230098\
7931277361782154249992295763514822082698951936680331825288693984964651058\
2093923982948879332036250944311730123819706841614039701983767932068328237\
6464804295311802328782509819455815301756717361332069811250996181881593041\
6903515988885193458072738667385894228792284998920868058257492796104841984\
'''
tstring = '''8920149245755439349113919376993323776170088510979008073273376\
60443062611162053029217546340958485'''

edigits = int(estring)
all_digits = len(estring)

''' '''
def fractionals(alpha_string, n, num_digits):
	'''
	take the digits of a number, interpreted as those after the decimal
	scale and take only the fractional of the scale, for number 1 to n
	'''
	num = int(alpha_string)
	num_precision = len(alpha_string)
	trunc_digits = num_precision - num_digits
	x = [10**num_digits]
	for i in range(1, n + 1):
		ni = num*i
		xi = ni//10**trunc_digits
		yi = (xi - (xi//10**num_digits)*10**num_digits)
		x.append(yi)
	return x

def closest(x, y, diff):
	'''
	'''
	good_diffs = []
	for i in range(1,len(y)):
		d2 = abs(x - y[i])%y[0]
		if d2 < diff:
			good_diffs.append([d2, i])
	return good_diffs

def closest_search(y):
	'''
	'''
	best_index = {}
	n = len(y)
	for i in range(1, len(y)): # for each value
		for j in range(2,n): # for multiples of that value
			good_diffs = closest(j*y[i]%y[0], y, 2*n) # find the closest match
			for pair in good_diffs:
				(d2, ki) = pair
				if d2 < 2*n:
					if i in best_index.keys():
						best_index[i].append([d2,j,ki])
					else:
						best_index[i] = [[d2,j,ki]]
	max_num = 0
	max_i = 0
	for k in best_index.keys():
		num_this = len(best_index[k])
		#print("key", k, "has", num_this, "elements:")
		if num_this > max_num:
			max_num = num_this
			max_i = k
		#for v in best_index[k]:
			#print(v[1], " times list[", k, "] == list[", v[2], "]", sep="", end="")
			#print("\t\t\t", y[v[2]] - v[1]*y[k]%y[0] )
	print("LOOKS LIKE your base element is at index", max_i, "!!!")
	print("......................................................")
	print("... is this your number:")
	print("")
	print("")
	print("\n\n\n")
	print(y[max_i])
	print("")
	print("")
	print("\n\n\n")
	print("... it is ... isn't it.")
	print("\n\n\n")
	#return best_index

def frac_ratio(den, f1, f2):
	mm = rlq(3,4)
	mm.setnull([den, f1, f2])
	mm.digallinc()
	return mm.pid[0][2], mm.pid[0][3], mm.pid[0][1] # a, b, c where a(f2) + b(f1) + c = 0

class primitives_set(object):
	'''
	primes in the range (n/2, n]
	
	'''
	def __init__(self, num):
		self.extradictionary = {}
		half_num = num//2
		pi = prime.find_next_prime(half_num)
		self.dictionary = {pi: 0}
		pi = prime.find_next_prime(pi + 1)
		while pi <= num:
			self.dictionary[pi] = 0
			pi = prime.find_next_prime(pi + 1)

	def __getitem__(self,k):
		return self.dictionary[k]

	def append(self, element, index):
		if element in self.dictionary.keys():
			if self.dictionary[element] == 0:
				self.dictionary[element] = index
			if self.dictionary[element] == index:
				return True
			else:
				print("NEW ONE FOUND...? element=", element, "index =", index)
				return False
		else:
			#print("                                      APPENDING A NEW ELEMENT!!! NO GOOD!!! element=", element)
			self.extradictionary[element] = 1 # add more later if want to keep track of this
			return False

	def primes(self):
		return self.dictionary.keys()

def run_pairs(xlist):
	'''
	xlist[0] is the denominator for all the fractions
	so that f[1] = xlist[1]/xlist[0]
	and ies f[2] = xlist[2]/xlist[0]
	etc...
	'''
	n = len(xlist) - 1
	p_indices = primitives_set(n)
	for i in range(1, n):
		for j in range(i + 1, n + 1):
			a, b, c = frac_ratio(x[0], xlist[i], xlist[j]) # do the LLL reduction of [[1,1,0,0],[f1,0,1,0],[f2,0,0,1]], smallest in lattice
			if p_indices.append(abs(a), j): pass #print(i,j,a,b, " with units", c)
			if p_indices.append(abs(b), i): pass #print(i,j,a,b, " with units", c)
			
	#print(" indices:", p_indices.dictionary)
	return p_indices
	
	

def main():
	global ml # to have it available on the IDLE after file executes, has to be prior to def
	global edigits # want to test from a number, e = 2 + edigits/10**2076
	global x
	global mm
	global y
	global yy
	global p_indices
	
	num_digits = 25
	num_scales = 150

	x = fractionals(estring, num_scales, num_digits)

	test_good = True
	for i in range(1, len(x) - 1):
		for j in range(i + 1, len(x)):
			mm = rlq(3,4)
			mm.setnull([x[0], x[i], x[j]])
			mm.digallinc()
			##print(i,j,mm.pid[0])
			ztest = i*mm.pid[0][2] + j*mm.pid[0][3]
			if ztest != 0:
				print("********************************************************Little boo at", i, j)
				test_good = False
				z2 = i*mm.pid[1][2] + j*mm.pid[1][3]
				z3 = i*mm.pid[2][2] + j*mm.pid[2][3]
				if z2 != 0 and z3 != 0:
					print("********************************************************BIG BOO at", i, j)
	print("test_good is", test_good)
	if test_good: print("-------------------------YAY!-------------------------")
	else: print("-------------------------BOO!-------------------------")
	print("")
	print("")
	y = fractionals(tstring, num_scales, num_digits)
	yy = y[:]
	yy.sort()
	yy = [yy[-1]] + yy[:-1] # put the denominator back to the front
	p_indices = run_pairs(yy)
	for p in p_indices.primes():
		ztest = yy[p_indices[p]] - y[p] ## CHECKING HERE,
		if ztest != 0: print("OH NO!!! FAIL.....................!")
		else: print(".", end="") # OTHERWISE we have that (p1 - p0)*frac == yy[p_i_0] - yy[p_i_1] (mod den = yy[0])
	closest_search(yy)
	


import code
if __name__ == "__main__":
	main()
	
	print("\t\t\t\t\t\t\tINTERACTIVE MODE---type exit() to quit.")
	code.interact(local=locals())
