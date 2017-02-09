import pylab
import matplotlib.pyplot as plt
import numpy as np
# y = mx + b
# m is slope, b is y-intercept

####################################################################3
def compute_error_for_line_given_points(b, m, cells,X,Y):
    totalError = 0
  
    for i in range(0, len(cells)):
        x = float(X[i])
        y = float(Y[i])
        totalError += (y - (m * x + b)) ** 2
    
    return (totalError / float(len(cells)))

def step_gradient(b_current, m_current, cells, learningRate,X,Y):
    b_gradient = 0
    m_gradient = 0
    N = float(len(cells))
    for i in range(0, len(cells)):
        x = float(X[i])
        y = float(Y[i])
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
        
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(cells, starting_b, starting_m, learning_rate, num_iterations,X,Y):
    b = starting_b
    m = starting_m
    q = []
    
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(cells), learning_rate,X,Y)
        #for j in range(0,50):
        #    x = float(X[i])
        #    y = float(Y[i])
        #    q[i] = (m*x)+b
           
    #print(q)
    #plt.plot(q)
    #plt.show()

    
    return [b, m]

def run():
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
	
         ###########################################################################
# Code for Part b of Question 3.1	
#Plotting the training data points

# We can use zip to unpack our data from pairs into lists
#    plt.scatter(*zip(*trainingSet))
#    plt.xlabel('Miles Per Gallon')
#    plt.ylabel('Displacement')
#   plt.show()  
########################################################################### 
    #### Command to show the plot for Part b
    #
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    learning_rate = 0.0001     
    num_iterations = 1000
    x=0
    y=0
   
    print( "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, cells,X,Y)))
    [b, m] = gradient_descent_runner(cells, initial_b, initial_m, learning_rate, num_iterations,X,Y)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, cells,X,Y)))
    
  #  pylab.plot(x,y); 
  #  pylab.grid(); 
  #  pylab.show()
  #  for i in range(1,1000):
     #   print(Z[i])   
#	    predictedYset.append([1,Z[i]])
#	    i+=1
#   plt.plot(*zip(*predictedYset))
        
     
    #############

#    print (m,b)
#    for i in range(0,50):
#	    y = m*float(X[i])+b
#	    i=i+1
#    plt.plot(X,y) 

   
   	
if __name__ == '__main__':
    run()



