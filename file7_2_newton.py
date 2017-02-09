##Question 7.2 Part e
##
from math import pow
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

## Defining the given function
def f( x,y ):
	return (3*pow(x,2) + (-1 * pow(y,2)))

## Defining the first order partial derivative with respect to x
def firstPartialDerivative_fx(x,y):
	return (6*x)

## Defining the first order partial derivative with respect to y
def firstPartialDerivative_fy(x,y):
	return (-2*y)

## Defining the second order partial derivative with respect to x
def secondPartialDerivative_fx(x,y):
	return (6)

## Defining the second order partial derivative with respect to y
def secondPartialDerivative_fy(x,y):
	return (-2)
	
# Starting point as given
x=5
y=-1


#Initialising two arrays to store the values of x and y which are visited to find the minimum value during the following iterations
Values_of_x_visted=[]
Values_of_y_visted=[]

# Number of iterations given = 5
for i in range(0, 5):
	print (x,y,f(x,y))
	xNew = x - (firstPartialDerivative_fx(x,y)/secondPartialDerivative_fx(x,y))
	yNew = y - (firstPartialDerivative_fy(x,y)/secondPartialDerivative_fy(x,y))
# Adding the values of visited x and y into an array so that we may be able to plot an array in future.	
	Values_of_x_visted.append(x)
	Values_of_y_visted.append(y)
# Updating the values of x and y	
	x=xNew
	y=yNew
	
print("\nBy Newton's method, the given function converges to minimum at x = ",x," and y= ",y)	

