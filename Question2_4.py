##Exercise 2.4 Source Code
##Newton.py on the function given in Question2.3
from math import pow

## Defining the given function
def f( x,y ):
	return (20*pow(x,2)+ 0.25*pow(y,2))

## Defining the first order partial derivative with respect to x
def firstOrderPartialDerivative_fx(x,y):
	return (40*x)

## Defining the first order partial derivative with respect to y
def firstOrderPartialDerivative_fy(x,y):
	return (0.5*y)

## Defining the second order partial derivative with respect to x
def secondOrderPartialDerivative_fx(x,y):
	return (40)

## Defining the second order partial derivative with respect to y
def secondOrderPartialDerivative_fy(x,y):
	return (0.5)

# Initial point
x=-2
y=4

#Learning rate
epsilon=0.04

## Note the division here by second order Partial derivative in the expression below is basically the inversion of the Hessian Matrix

for i in range(0, 1000):
	print (x,y,f(x,y))
	xNew = x - epsilon * firstOrderPartialDerivative_fx(x,y)/secondOrderPartialDerivative_fx(x,y)
	yNew = y - epsilon * firstOrderPartialDerivative_fy(x,y)/secondOrderPartialDerivative_fy(x,y)
	# Updating the values of x and y	
	x=xNew
	y=yNew
	
print("\n By Newtonian Method, the given function converges to minimum at x = ",x," and y= ",y)	
	
	
	
	