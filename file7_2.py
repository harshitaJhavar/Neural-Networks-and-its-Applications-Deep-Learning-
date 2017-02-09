##Question 7.2 Part a to d
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
	
# Starting point as given
x=5
y=-1
x_mom = 5
y_mom = -1

#Learning rate as given 
epsilon=0.01

#Given for Gradient Descent with momentum, value of momentu parameter alpha
alpha = 0.7

#Initialising for Gradient Descent with momentum, velocity v_x_mom and v_y_mom
v_x_mom = 0
v_y_mom = 0

#For GD algorithm
#Initialising two arrays to store the values of x and y which are visited to find the minimum value during the following iterations
Values_of_x_visted=[]
Values_of_y_visted=[]

# Number of iterations given = 30
for i in range(0, 30):
	print (x,y,f(x,y))
	xNew = x - epsilon * firstPartialDerivative_fx(x,y)
	yNew = y - epsilon * firstPartialDerivative_fy(x,y)
# Adding the values of visited x and y into an array so that we may be able to plot an array in future.	
	Values_of_x_visted.append(x)
	Values_of_y_visted.append(y)
# Updating the values of x and y	
	x=xNew
	y=yNew
	
print("\nFor GD, the given function converges to minimum at x = ",x," and y= ",y," using Gradient Descent")	

#Evaluating the termination function value
Termination_function_value = f(x,y)
print("\n Termination function value for gradient descent is ",Termination_function_value)

#For Gradient Descent with momentum algorithm
#Initialising two arrays to store the values of x_mom and y_mom which are visited to find the minimum value during the following iterations
Values_of_x_mom_visted=[]
Values_of_y_mom_visted=[]

# Number of iterations given = 30
for i in range(0, 30):
    print (x_mom,y_mom,f(x_mom,y_mom))
	
	#Updating the parameters  
    v_x_mom = (v_x_mom * alpha) - (epsilon * (firstPartialDerivative_fx(x_mom,y_mom)))
    v_y_mom = (v_y_mom * alpha) - (epsilon * (firstPartialDerivative_fy(x_mom,y_mom)))
    xNew_mom = x_mom + v_x_mom
    yNew_mom = y_mom + v_y_mom 
    
    #Adding the values of visited x_mom and y_mom into an array so that we may be able to plot an array in future.
    Values_of_x_mom_visted.append(x_mom)
    Values_of_y_mom_visted.append(y_mom)
    #Updating the values of x_mom and y_mom
    x_mom=xNew_mom
    y_mom=yNew_mom
	
print("\nFor GD with momentum, the given function converges to minimum at x_mom = ",x_mom," and y_mom= ",y_mom)	

#Evaluating the termination function value
Termination_function_value_mom = f(x_mom,y_mom)
print("\n Termination function value for gradient descent with momentum is ",Termination_function_value_mom)

##Plotting the contour plot 

#For gradient Descent
X, Y = np.meshgrid(Values_of_x_visted,Values_of_y_visted)

zs = np.array([f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

plt_contour = plt.contour(X,Y,Z,30)  
  
plt.clabel(plt_contour, inline=1, fontsize=10) 
plt.plot(Values_of_x_visted, Values_of_y_visted, color='red', linestyle='solid', marker='o', markersize=4)  
plt.title('Contour Plot for Gradient Descent')    
plt.show() 

#For gradient Descent with momentum

X_mom, Y_mom = np.meshgrid(Values_of_x_mom_visted,Values_of_y_mom_visted)

zs_mom = np.array([f(x_mom,y_mom) for x_mom,y_mom in zip(np.ravel(X_mom), np.ravel(Y_mom))])
Z_mom = zs_mom.reshape(X_mom.shape)

plt_contour_mom = plt.contour(X_mom,Y_mom,Z_mom,30)  
  
plt.clabel(plt_contour_mom, inline=1, fontsize=10) 
plt.plot(Values_of_x_mom_visted, Values_of_y_mom_visted, color='green', linestyle='solid', marker='o', markersize=4)  
plt.title('Contour Plot for Gradient Descent with momentum')    
plt.show() 

