##Question 7_3 

from math import pow
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


## Defining the given function
def f( x,y ):
	return (0.001*pow(x,2) - 0.001*pow(y,2))

## Defining the first order partial derivative with respect to x
def firstPartialDerivative_fx(x,y):
	return (0.002*x)

## Defining the first order partial derivative with respect to y
def firstPartialDerivative_fy(x,y):
	return (-1*0.002*y)
	
# Starting point as given
x = 3
y = -1
x_AGD = 3
y_AGD = -1

#Learning rate as given 
epsilon=0.1

#Given for AdaGrad, value of delta - small constant for numerical stability
delta = pow(10,-9)
#Initialising for AdaGrad, small gradient descent accumulation variable 'r_x_ADG' and 'r_y_ADG' as given in slide 21 of Lecture
r_x_AGD = 0
r_y_AGD = 0

#For GD algorithm
#Initialising two arrays to store the values of x and y which are visited to find the minimum value during the following iterations
Values_of_x_visted=[]
Values_of_y_visted=[]

# Number of iterations given = 300
for i in range(0, 300):
	print (x,y,f(x,y))
	xNew = x - epsilon * firstPartialDerivative_fx(x,y)
	yNew = y - epsilon * firstPartialDerivative_fy(x,y)
# Adding the values of visited x and y into an array so that we may be able to plot an array in future.	
	Values_of_x_visted.append(x)
	Values_of_y_visted.append(y)
# Updating the values of x and y	
	x=xNew
	y=yNew
	
print("\nFor SGD, the given function converges to minimum at x = ",x," and y= ",y," using Gradient Descent")	

#For AdaGrad algorithm
#Initialising two arrays to store the values of x_AGD and y_AGD which are visited to find the minimum value during the following iterations
Values_of_x_AGD_visted=[]
Values_of_y_AGD_visted=[]

# Number of iterations given = 300
for i in range(0, 300):
    print (x_AGD,y_AGD,f(x_AGD,y_AGD))
	
	#Updating the value of the accumulated squared gradient as mentioned on the slide of the lecture.
    r_x_AGD = r_x_AGD + ((firstPartialDerivative_fx(x_AGD,y_AGD)) * (firstPartialDerivative_fx(x_AGD,y_AGD)))
    r_y_AGD = r_y_AGD + ((firstPartialDerivative_fy(x_AGD,y_AGD)) * (firstPartialDerivative_fy(x_AGD,y_AGD)))
    xNew_AGD = x_AGD - ((epsilon * firstPartialDerivative_fx(x_AGD,y_AGD))/(delta + pow(r_x_AGD,0.5)))
    yNew_AGD = y_AGD - ((epsilon * firstPartialDerivative_fy(x_AGD,y_AGD))/(delta + pow(r_y_AGD,0.5)))
    #Adding the values of visited x_AGD and y_AGD into an array so that we may be able to plot an array in future.
    Values_of_x_AGD_visted.append(x_AGD)
    Values_of_y_AGD_visted.append(y_AGD)
    #Updating the values of x_AGD and y_AGD
    x_AGD=xNew_AGD
    y_AGD=yNew_AGD
	
print("\nFor AGD, the given function converges to minimum at x_AGD = ",x_AGD," and y_AGD= ",y_AGD)	

##Contour plot for Gradient Descent.

X, Y = np.meshgrid(Values_of_x_visted,Values_of_y_visted)

zs = np.array([f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

plt_contour = plt.contour(X,Y,Z,30)  
  
plt.clabel(plt_contour, inline=1, fontsize=10) 
plt.plot(Values_of_x_visted, Values_of_y_visted, color='red', linestyle='solid', marker='o', markersize=4)  
plt.title('Contour Plot for Gradient Descent')    
plt.show() 

###Contour plot for AdaGrade.
X_AGD, Y_AGD = np.meshgrid(Values_of_x_AGD_visted,Values_of_y_AGD_visted)

zs_AGD = np.array([f(x,y) for x,y in zip(np.ravel(X_AGD), np.ravel(Y_AGD))])
Z_AGD = zs_AGD.reshape(X_AGD.shape)

plt_contour_AGD = plt.contour(X_AGD,Y_AGD,Z_AGD,30)
plt.clabel(plt_contour_AGD, inline=1, fontsize=10) 
plt.plot(Values_of_x_AGD_visted, Values_of_y_AGD_visted, color='blue', linestyle='solid', marker='o', markersize=4)      
plt.title('Contour Plot for AdaGrad')    
plt.show()    
