#Exercise 4.1 solution
import random as rand
from clustering import clustering
from point import Point
import csv
import matplotlib.pyplot as plt
import numpy as np

coordinates = []

f = open('data_for_k_means.txt', 'r')

reader = csv.reader(f, delimiter=" ")
for line in reader:
    loc_ = Point(float(line[0]), float(line[1]))  #tuples for location
    coordinates.append(loc_)

#k-means
cluster = clustering(coordinates, 2 )
flag = cluster.k_means(False)
if flag == -1:
    print ("Error in arguments!")
else:
    #the clustering results is a list of lists where each list represents one cluster
    print ("clustering results:")
    cluster.print_clusters(cluster.clusters)
    
plt.show()
   
