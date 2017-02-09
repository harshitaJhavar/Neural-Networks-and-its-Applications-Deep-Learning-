##Solution for Question 3.1

# Code for Part a of Question 3.1
# Reading the csv file
f = open( 'auto-mpg.data', 'rU' )
# Declaring four empty lists
# X represents mpg (Miles per Galon)
X = []
# Y represents displacement (Engine Displacement)
Y = []
# training set and testing set are currently empty list
trainingSet = []
testSet = []
#Reading the input data file
for line in f:
	cells = line.split( "   " )
#Storing the values of mpg and displacement in X and Y respectively	
	X.append(cells[ 0 ])
	Y.append(cells[ 2 ])
# Initialising a counter with variable i	
i=0
#Storing first 50 values in training Set
for i in range(0,50):
	trainingSet.append([X[i],Y[i]])
	i+=1
#Storing next 50 values in testing set	
for i in range(50,100):
	testSet.append([X[i],Y[i]])
	i+=1
f.close()
print(X)
###########################################################################
# Code for Part b of Question 3.1	
#Plotting the training data points
import matplotlib.pyplot as plt
# We can use zip to unpack our data from pairs into lists
plt.scatter(*zip(*trainingSet))
plt.xlabel('Miles Per Gallon')
plt.ylabel('Displacement')
	  
###########################################################################
# Code for Part d of Question 3.1
# Fitting a first order model to the given training set
# Assuming the value of learning rate alpha = 0.1 and point of initialisation as [9,304].

import numpy as np

X = trainingSet[0]
Y = trainingSet[1]
arrayX = np.asarray(X)
arrayY = np.asarray(Y)

#number of training samples
m = 50

#Add a column of ones to X (interception data)
it = np.ones(shape=(m, 2))
it[1] = arrayX

#Initialize theta parameters
theta = np.zeros(shape=(2, 1))

#Some gradient descent settings
iterations = 1500
alpha = 0.01
#Evaluate the linear regression
def compute_cost(arrayX, arrayY, theta):
    #Number of training samples
	m = 50
	predictions = arrayX.dot(theta).flatten()
	sqErrors = (predictions - arrayY) ** 2
	J = (1.0 / (2 * m)) * sqErrors.sum()
	return J


def gradient_descent(arrayX, arrayY, theta, alpha, num_iters):
	m = 50
	J_history = np.zeros(shape=(num_iters, 1))
	for i in range(num_iters):
		predictions = arrayX.dot(theta).flatten()
		errors_x1 = (predictions - arrayY) * arrayX[:, 0]
		errors_x2 = (predictions - arrayY) * arrayX[:, 1]
		theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
		theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()
		J_history[i, 0] = compute_cost(arrayX, arrayY, theta)
		return theta, J_history
compute_cost(arrayX, arrayY, theta)
gradient_descent(arrayX, arrayy, theta, alpha, num_iters)
print(theta)
print(J_history)
#### Command to show the plot for Part b
plt.show()
