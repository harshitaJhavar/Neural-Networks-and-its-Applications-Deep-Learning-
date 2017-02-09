##Solution for Question 2.5
##Source code for stopping criterion
## Let us solve it for exercise 2.3- Gradient Descent

##Question 2.3 Gradient Descent Source Code

from math import pow

## Initiating number of iterations to 0.

Number_of_Iterations = 0
## Defining the value of precision
precision = 1/1000000
## Limiting number of iterations to be at max 1000.
maxIterations = 1000

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

#Gradient Descent
while True:
	print (x,y,f(x,y))
	xNew = x - epsilon * firstPartialDerivative_fx(x,y)
	yNew = y - epsilon * firstPartialDerivative_fy(x,y)
	if abs(xNew-x) < precision and abs(yNew-y) < precision:
		break
	x=xNew
	y=yNew
	Number_of_Iterations += 1
	##Putting a check if the number of iterations cross the maximum iterations value
	if Number_of_Iterations > maxIterations:
		print("There are many iterations. You might want to change the value of the epsilon.")
		break
if Number_of_Iterations < maxIterations:		
	print("The number of iterations required by this function to minimize were ",Number_of_Iterations)
	print("\nThe given function converges to minimum at x = ",x," and y= ",y)	

