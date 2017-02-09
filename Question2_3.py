##Question 2.3 Gradient Descent Source Code
##
from math import pow
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

## Defining the given function
def f( x,y ):
	return (20*pow(x,2)+ 0.25*pow(y,2))

## Defining the first order partial derivative with respect to x
def firstPartialDerivative_fx(x,y):
	return (40*x)

## Defining the first order partial derivative with respect to y
def firstPartialDerivative_fy(x,y):
	return (0.5*y)
	
# Starting point as given
x=-2
y=4

#Learning rate as given 
epsilon=0.04

#Initialising two arrays to store the values of x and y which are visited to find the minimum value during the following iterations
Values_of_x_visted=[]
Values_of_y_visted=[]

# We have assumed here that we will run 1000 iterations to find the point of minima.
for i in range(0, 1000):
	print (x,y,f(x,y))
	xNew = x - epsilon * firstPartialDerivative_fx(x,y)
	yNew = y - epsilon * firstPartialDerivative_fy(x,y)
# Adding the values of visited x and y into an array so that we may be able to plot an array in future.	
	Values_of_x_visted.append(x)
	Values_of_y_visted.append(y)
# Updating the values of x and y	
	x=xNew
	y=yNew
	
print("\nThe given function converges to minimum at x = ",x," and y= ",y)	

##Solution to plot the contour plot for the given function.
#####################################################################################
##Plotting the contour plot for epsilon = 0.04
X, Y = np.meshgrid(Values_of_x_visted,Values_of_y_visted)

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# difference of Gaussians
Z = 10.0 * (Z2 - Z1)
plt.figure()
CS = plt.contour(X, Y, Z, 6,
                 linewidths=np.arange(.5, 4, .5),
                 colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5')
                 )
plt.clabel(CS, fontsize=9, inline=1)
plt.title('Contour Plot for Epsilon = 0.04')
plt.show()