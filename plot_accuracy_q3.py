#File to generate accuracy plot from output of question8_3.py
import matplotlib.pyplot as plt
import itertools
#Creating two empty lists to store x and y axis
x = []
y = []

with open('a.out') as f:
    for line in itertools.islice(f, 4, 104):
    	word = line.split()
    	x.append(float(word[1])+1)
    	y.append(float(word[4]))
    		
plt.plot(x,y,label="Accuracy obtained at each step")
plt.show()    
