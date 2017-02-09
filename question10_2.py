#Solution for question 2
import random
import numpy as np
import tensorflow as tf
#Part a
data_file = open("data.txt", "r")

for line in data_file:
    result = ''.join(line.strip() for line in data_file)  
    result = result.replace(" ", "")
#print(result)	
data_file.close()

#Part b
#We have to randomize here both, random index position and random k which is length of chunk.

k = random.randint(0,len(result)-1)
random_index = random.randint(0,len(result)-1)

#Checking if the random_index + k is within range of given sequence
while((random_index + k - 1 ) > len(result)):
	random_index = random.randint(0,len(result)-1)
random_chunk = ''
for i in range(k): 
    random_chunk += result[random_index]
    random_index = random_index +1
#So, now random_chunk has random chunk of length k

#Part c

#Declaring an array with all values as zero	
array_Random_Chunk = np.zeros((k+1,4))
#For Adding a zero SoS vector in the beginning, we left the 0th row with all zeros
for i in range(1,k+1):
	if(random_chunk[i-1]=='a'):
		array_Random_Chunk[i][0] = 1
	elif(random_chunk[i-1]=='c'):
		array_Random_Chunk[i][1] = 1
	elif(random_chunk[i-1]=='g'):
		array_Random_Chunk[i][2] = 1
	else: #This is case for t
		array_Random_Chunk[i][3] = 1
#print(array_Random_Chunk)	
