## takes a lot of digits and treats them as the decimal value of \alpha \in (0, 1)
## multiplies it by numbers 1...n and keeps only the fractional part up to num_digits
## Then checks if the pairs of those can be found thier ratio


from __future__ import print_function
from rlq import *
import random
import time
import copy
import prime


# the digits after the decimal of e = 2.7182818284...
estring ='''718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427427466391932003059921817413596629043572900334295260595630738132328627943490763233829880753195251019011573834187930702154089149934884167509244761460668082264800168477411853742345442437107539077744992069551702761838606261331384583000752044933826560297606737113200709328709127443747047230696977209310141692836819025515108657463772111252389784425056953696770785449969967946864454905987931636889230098793127736178215424999229576351482208269895193668033182528869398496465105820939239829488793320362509443117301238197068416140397019837679320683282376464804295311802328782509819455815301756717361332069811250996181881593041690351598888519345807273866738589422879228499892086805825749279610484198444363463244968487560233624827041978623209002160990235304369941849146314093431738143640546253152096183690888707016768396424378140592714563549061303107208510383750510115747704171898610687396965521267154688957035035402123407849819334321068170121005627880235193033224745015853904730419957777093503660416997329725088687696640355570716226844716256079882651787134195124665201030592123667719432527867539855894489697096409754591856956380236370162112047742722836489613422516445078182442352948636372141740238893441247963574370263755294448337998016125492278509257782562092622648326277933386566481627725164019105900491644998289315056604725802778631864155195653244258698294695930801915298721172556347546396447910145904090586298496791287406870504895858671747985466775757320568128845920541334053922000113786300945560688166740016984205580403363795376452030402432256613527836951177883863874439662532249850654995886234281899707733276171783928034946501434558897071942586398772754710962953741521115136835062752602326484728703920764310059584116612054529703023647254929666938115137322753645098889031360205724817658511806303644281231496550704751025446501172721155519486685080036853228183152196003735625279449515828418829478761085263981395599006737648292244375287184624578036192981971399147564488262603903381441823'''
tstring = '''892014924575543934911391937699332377617008851097900807327337660443062611162053029217546340958485'''

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
			if p_indices.append(abs(a), j): print(i,j,a,b, " with units", c)
			if p_indices.append(abs(b), i): print(i,j,a,b, " with units", c)
			
	print(" indices:", p_indices.dictionary)
	return p_indices
	
	

def main():
	global ml # to have it available on the IDLE after file executes, has to be prior to def
	global edigits # want to test from a number, e = 2 + edigits/10**2076
	global x
	global mm
	global y
	global yy
	global p_indices
	
	num_digits = 12
	num_scales = 77

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
	for k in p_indices.primes():
		ztest = yy[p_indices[k]] - y[k]
		if ztest != 0: print("OH NO!!! FAIL.....................!")
		else: print(".", end="")
	


import code
if __name__ == "__main__":
	main()
	
	print("INTERACTIVE MODE---type exit() to quit.")
	code.interact(local=locals())
