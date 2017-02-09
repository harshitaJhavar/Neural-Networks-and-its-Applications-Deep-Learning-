#Solution for part b and c of question 10.1
import numpy as np
from math import *
import matplotlib.pyplot as plt
from numpy import linalg as LA
#Defining matrix M
M = np.matrix('-2 -2 3; -10 -1 6; 10 -2 -9')

#Defining the random vector u
u = np.random.rand(3,1)

#Defining the vector v computed from part a
#Since norm of v is 1, so, we have divided it by sqroot of six to keep the norm one of vector v
v = (1/6**(.5))*np.matrix('1; 2; 1')
#print(v)

#Defining product as an empty list so that we can plot it later
product = []
product.append(LA.norm(np.dot(u,np.transpose(v))))
#Defining the iterations and applying the power method
while(LA.norm(np.dot(u,np.transpose(v)))!=1):
	tempA = M * u
	#print(tempA)
	tempB = tempA
	u = tempB / min(tempA.min(), tempA.max(), key=abs)
	#Since, norm of u is also 1, so, resizing u
	u = u/LA.norm(u)
	product.append(LA.norm(np.dot(u,np.transpose(v))))
	#print(u)
#Part c for plotting
plt.plot(product)
plt.xlabel("No of iterations")
plt.show()
	
