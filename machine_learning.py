import numpy as np
import math 
import random
import time 


def converToNumpyArray(python_list):
	return np.array(python_list)

def convertToPythonArray(numpy_array):
	return list(numpy_array)


def pointMagnitude(point):
	# get the sum of each squared 
	_sum = sum([i**2 for i in point])
	# return the square root of the final
	return math.sqrt(_sum)


def DistanceMagnitude(point_A,point_B):
	# points must be of the same dimension 
	# get the sum of each diff squared
	_sum=0
	for i in range(len(point_A)):
		_sum+=(point_A[i]-point_B[i])**2
	return math.sqrt(_sum)



def Magnitude():
	pass

def randn(n,a=-100,b=100):
	return [random.gauss(a,b) for i in range(n)]
def convertToUnitVector(point):
	#[1,2]
	#[1,2,3]
	# return each point divided by its magnitude 
	magnitude = pointMagnitude(point)
	if (magnitude==0):
		return 0
	A=[]
	for i in point:
		A.append(float(i)/magnitude)

	return A 

def gen_point(point,radius):
	# P = O + tv
	# get unit vector of point 
	O=point
	length = len(point)
	if pointMagnitude(point)==0:
		unit_vector=[]
		# get random index
		index = random.randrange(len(point))
		#gen all zeros
		for i in range(0,length):
			unit_vector.append(0)
		# insert 1 at random index
		unit_vector[index]=1		
	else:
		#unit_vector = convertToUnitVector(point) 
		unit_vector = convertToUnitVector([(random.triangular(-1,1)) for i in range(0,length)]) 
	#return(DistanceMagnitude(point,list(O+np.array(unit_vector)*radius)))
	return(list(O+np.array(unit_vector)*radius))



def solve():
	pass





def kNN(data,test_point,K=3):
	#[1,2,3,a],	
	#[0.7,1.3] test point 
	# get each point and its class	
	# get test point
	# find distance of test point to each point and store (class,distance) bcos point not needed after this step	
	# ('a',2) ('b',1) ('a',1.5) ('b',1.2)   //    {'a':2,'b',3,'a':4}    //      {'2':'a','3':'a','4':a}

	distances=[]
	for i in data:
		current_data_point=i[:len(i)-1]		
		current_class=i[len(i)-1]
		current_distance=DistanceMagnitude(current_data_point,test_point)
		print("##############################################################")
		print("Current data point is {}".format(current_data_point))
		print("Current distance between {} and {} is {}".format(test_point,current_data_point,current_distance))
		print("##############################################################")
		distances.append([current_distance,current_class])
		print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
		print(distances)
		print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
	# sort data by distances
	sorted_array=sorted(distances)
	print("---------------------------------------------------")
	print(sorted_array)
	print("---------------------------------------------------")
	# [('b',1),('b',1.2),('a',1.5)]
	# find first 2 elements
	sorted_array=sorted_array[:K]
	print("---------------------------------------------------")
	print(sorted_array)
	print("---------------------------------------------------")
	#select most common class
	_dict={}
	for j in sorted_array:
		if j[1] not in _dict:			
			_dict[j[1]]=1
		else:
			_dict[j[1]]+=1
	print("dict is {}".format(_dict))
	_max=max(_dict.values())
	print("max is "+str(_max))
	for k in _dict:
		if _dict[k]==_max:
			print(k)


#---------------------------------------------------------------------
def randomvector(n):
	return [(random.triangular(-1,1)) for i in range(n)]


def evaluate(X):
	#collect x and y 	
	x,y=X[0],X[1]	
	#solve a1	
	a1=x+y-6 
	a2=x-y+3
	#a3=(x**3)*(y**7)+87*(y**4)-1927*((x**5)*7*y)-49
	# a1=(x**3)-4*(y**8)-89
	# a2=(x*y)-27*(x**3)-24
	#square both
	a1=a1**2 
	a2=a2**2 
	return a1+a2

CD=[[-94,44,24],[31,42,44],[-73,-51,39],]
CD=[[-94,44,24],[31,42,5],[-73,-51,39],]

def evalll(X):
	x,y,r = X[0],X[1],X[2] 
	Ca,Cb,Cc=CD
	xa,ya,ra=Ca
	xb,yb,rb=Cb 
	xc,yc,rc=Cc
	s=0
	s+=((r+ra)**2-(xa-x)**2-(ya-y)**2)**2
	s+=((r+rb)**2-(xb-x)**2-(yb-y)**2)**2
	s+=((r+rc)**2-(xc-x)**2-(yc-y)**2)**2 
	return s 

def evall(X):
	x,y,r=X 
	s=0 
	s+=((x-255)**2+(y-129)**2-(r**2))**2
	s+=((x-258)**2+(y-34)**2-(r**2))**2
	s+=((x-171)**2+(y-55)**2-(r**2))**2
	return s


#sol=[random.triangular(-1,1),random.triangular(-1,1),random.triangular(-1,1)]

'''
def gen_initial(X):
	return ([0]* )
'''

# y = 2 x +1 ==>m=2,c=1
data=[[0,1],[1,3],[2,5],[3,7]]
data=[[0, -3.67], [1, -1.27], [2, 1.13], [3, 3.5299999999999994], [4, 5.93], [5, 8.33], [6, 10.729999999999999], [7, 13.13], [8, 15.53], [9, 17.93]]
best=9e99
sol=[0,0]

def calc(m,c,x):
	return (m*x)+c

def solvee():
	global best
	global data
	global sol 
	for i in range(6000000):
		
		e=0
		xx=gen_point(sol,random.triangular(-1,1))
		for i in range(len(data)):
			# get current point
			current_point = data[i]
			current_x,current_y=current_point
			current_guess = calc(xx[0],xx[1],current_x)
			current_error = (current_y-current_guess)
			e+=abs(current_error)**2
		print("---------------------------------------------")
		print("best is ",best)
		print("current value is ",sol)
		print("---------------------------------------------")
		
		if e<best:
			best=e 
			sol = xx 
		if e<=0:
			print(e)
			print("stop")
			break
	print (str(i)+"done")

def solve():	
	global best
	global sol	
	for i in range(0,1000000):		
		ll=pointMagnitude(sol)		
		x=gen_point(sol,random.triangular(-1,1))				
		#x=gen_point(sol,random.triangular((-1*best),best))				
		z=evall(x)		
		print("best is ",best)
		print("---------------------------------------------")
		print("current value is ",sol)
		print("---------------------------------------------")
		if z<best: 									
			best = z
			sol  = x
		if z<=0.000000000000001:
			break
	print (sol)
	# evaluate P
#---------------------------------------------------------------------



# def log_best_to_file(value,file_name):
# 	# get the file name 
# 	file_name=file_name 
# 	# get current time 
# 	current_time = time.asctime 
# 	# get best  value 
# 	best_value = value  
# 	file = open("{}.txt".format(file_name),"a")
# 	file.write("{} \t\t {}".format(best_value,current time)) 



# x+y=3  --> x+y-3=0
# x-y=23 --> x-y-23=0

#---------------------------------------------------------------------------------------------------------------------

def calcShannonEnt(dataSet):
	#[[1,1'a'],[1,0,'b']]
	#[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
	# number of examples 
	numEntries=len(dataSet)
	# create dictionary of number of each class
	# eg. {"yes":7,"no":4}
	labelCounts={} 
	# for each instance
	for featureVect in dataSet:
		# if last index not in dictionary
		if featureVect[-1] not in labelCounts:
			# dictionary[last_index] = 1
			labelCounts[featureVect[-1]]=1
		else:
			labelCounts[featureVect[-1]]+=1
	#labelcount={'a':21,'b':13,'c':11}	
	# entropy= sum(p(x)log(p(x),2))
	entropy=0
	for i in labelCounts:
		# p(x)=i/total(total=sum of all values in k,v pairs)
		total=sum(labelCounts.values())
		probability=labelCounts[i]/total 
		current_entropy=probability*math.log(probability,2)
		entropy+=current_entropy
	return entropy*-1


def splitdata(dataset,axis,value):
	A=[]
	for i in dataset:
		if i[axis]==value:
			r=i[:axis]
			r.extend(i[axis+1:])
			A.append(r)
	return A 
		




def chooseBestFeatureToSplit(dataSet):
	# length of the first example -1
	numFeatures=len(dataSet[0])-1
	baseEntropy=CalcShannon(dataSet)



#---------------------------------------------------------------------------------------------------------------------
class NeuralNetwork:
	iter_ = 1000

	def __init__(self,data):
		pass

	def predict(self,test_point):
		pass

	def sigmoid(self):
		pass

	def sum_weights(self):
		pass

	def train(self,):
		pass


#---------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------



def main():
	#test_points=[[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]]
	#test_point=[18,90]
	#print(pointMagnitude([1,0]))
	#for i in test_points:
	#	print(DistanceMagnitude(i,test_point))
	#print(pointMagnitude([3,4]))
	#print(DistanceMagnitude([1,2,3],[1,2,3]))





	#print(convertToUnitVector([1,1]))
	#print(pointMagnitude(convertToUnitVector([1,4])))
	#print(pointMagnitude(convertToUnitVector([random.triangular(-1,1),random.triangular(-1,1)])))
	#print(gen_point([5,5],1))
	
	#b=[1,2,3]
	#a=gen_point(b,13)
	#print(a)
	#print(DistanceMagnitude(b,a))

	#arr = [[3,104,'r'],[2,100,'r'],[1,81,'r'],[101,10,'a'],[99,5,'a']]
	#kNN([[0,0,'0'],[0,1,'1'],[1,0,'1']],[1,1])
	#kNN(arr,[18,90])

	
	#solve()
	#a=calcShannonEnt([[1, 1, 'a'],[1, 1, 'b'],[1, 1, 'a'],[1, 1, 'a'],[1, 1, 'a'],])
	#print(a)

	solvee()

	# a=[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
	# print(splitdata(a,0,1))
	# a=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	# b=gen_point(a,1)
	# print(b)



if __name__=="__main__":
	main()



