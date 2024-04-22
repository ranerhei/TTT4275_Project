#!/usr/bin/python3

import csv
import matplotlib.pyplot as plt
import numpy as np

#data importer
from functions import load_csv

#activator function, equation 20
from functions import sigmoid

#function to calculate Mean Square Error, equation 19
from functions import MSE_function

#function to calculate gradient of MSE, equation 22
from functions import grad_MSE_function

from functions import get_class_from_vector

from functions import round_calculated_vector

from plot import plot_W_changes

#load data
#data has [sepal_length, sepal_width. petal_length, petal_width]
class_1 = load_csv('Iris_TTT4275/class_1')
class_2 = load_csv('Iris_TTT4275/class_2')
class_3 = load_csv('Iris_TTT4275/class_3')

#split each class into vectors containing sepal and petal heights
#and widths
def sepal_petal_seperator(class_x):
    sepal = []
    petal = []
    for vec in class_x:
        sepal.append(vec[:2])
        petal.append(vec[2:4])
    #print("sepal", sepal)
    #print("petal", petal)
    return sepal, petal

def plot_scatter(vector, label, color):
    x =  [point[0] for point in vector]

    y = [point[1] for point in vector]
    
    plt.scatter(x, y, label = label, color = color)


#make training and testing set
class_1_training = class_1[:30]
class_1_testing = class_1[30:]

class_2_training = class_2[:30]
class_2_testing = class_2[30:]

class_3_training = class_3[:30]
class_3_testing = class_3[30:]

#make a total training data set
training_data = [class_1_training, class_2_training, class_3_training]
#make a total test data set
testing_data = [class_1_testing, class_2_testing, class_3_testing]

# D is the amount of dimensions
D = len(class_1[0])-1
# C is the amount of classes
C = len(training_data)

#make the weights equal to 1
W = np.zeros((C, D+1))
#stored W matrixes
W_history = [W]

# store Mean Square Error
MSE_history = []
gradMSE_history = []

# set of target vectors
class_to_vector = {
     0: [1,0,0],
     1: [0,1,0],
     2: [0,0,1]
}


# klarte 1 feil med alpha = 0.01
alpha = 0.0062
steps = 2000

print('W before training:')
print(W)

# start training
for s in range(steps):
    gradMSE = 0
    total_MSE = 0
    #iterate over every class
    for i in range(len(training_data)):
        #define the target vector, aka correct class
        t = class_to_vector.get(i, [0,0,0])
        #iterate over every data point
        for j in range(len(training_data[i])):
            #perform matrix multiplication
            x = training_data[i][j]
            g = np.dot(W,x)
            g = sigmoid(g)
            total_MSE += MSE_function(g, t)
            gradMSE += grad_MSE_function(g,t,x,C)
    # update the W matrix
    W = W-alpha*gradMSE
    W_history.append(W)

print('W after training:')
print(W)

#Creates points of lengths and widths of sepal and petal of
#each class:
sepal_1, petal_1 = sepal_petal_seperator(class_1)
sepal_2, petal_2 = sepal_petal_seperator(class_2)
sepal_3, petal_3 = sepal_petal_seperator(class_3)
#Sepal plotting procedure:
"""
plot_scatter(sepal_1, "Class 1", "red")
plot_scatter(sepal_2, "Class 2", "blue")
plot_scatter(sepal_3, "Class 3", "green")
    
plt.xlabel("lengths")
plt.ylabel("widths")
plt.legend()
plt.title("Sepal lengths and width")
plt.grid(True)
plt.show()
"""
"""
#Petal plotting procedure:
plot_scatter(petal_1, "Class 1", "red")
plot_scatter(petal_2, "Class 2", "blue")
plot_scatter(petal_3, "Class 3", "green")
    
plt.xlabel("lengths")
plt.ylabel("widths")
plt.legend()
plt.title("Petal lengths and width")
plt.grid(True)
plt.show()
"""
errors = 0
confusion_matrix = [[0,0,0],
                    [0,0,0],
                    [0,0,0]]
# start testing:
# i, iterate over every class
for i in range(len(testing_data)):
    t = class_to_vector.get(i, [0,0,0])
    # j, iterate over every datapoint
    for j in range(len(training_data[i])):
        #perform matrix multiplication
        x = training_data[i][j]
        g = np.dot(W,x)
        g = sigmoid(g)
        rounded_g = round_calculated_vector(g)
        print(rounded_g)

        if rounded_g == t:
            confusion_matrix[i][i] += 1
        elif rounded_g != t:
            output_class = rounded_g.index(1)
            confusion_matrix[i][output_class] += 1

print(errors)

print(confusion_matrix[0])
print(confusion_matrix[1])
print(confusion_matrix[2])

        
plot_W_changes(W_history)
