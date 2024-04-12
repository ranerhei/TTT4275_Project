"""
we have
classes,        C=3
inputs,         D=4
training sets,  N=30
test sets,      M=20

"""

import csv
import matplotlib as plt
import numpy as np

#data importer
from functions import load_csv

#activator function, equation 20
from functions import sigmoid

#function to calculate Mean Square Error, equation 19
from functions import MSE_function

#function to calculate gradient of MSE, equation 22
from functions import grad_MSE_function

#load data
#data has [sepal_length, sepal_width. petal_length, petal_width]
class_1 = load_csv('Iris_TTT4275/class_1')
class_2 = load_csv('Iris_TTT4275/class_2')
class_3 = load_csv('Iris_TTT4275/class_3')

#make training and testing set
class_1_training = class_1[:30]
class_1_testing = class_1[30:]

class_2_training = class_2[:30]
class_2_testing = class_2[30:]

class_3_training = class_3[:30]
class_3_testing = class_3[30:]

#make a total training data set
training_data = [class_1_training, class_2_training, class_3_training]

# D is the amount of dimensions
D = len(class_1[0])-1
# C is the amount of classes
C = len(training_data)

#make the weights equal to 1
W = np.ones((C, D+1))
# store total Mean Square Error
MSE = 0
gradMSE = 0

# set of target vectors
target_vectors = {
     0: [1,0,0],
     1: [0,1,0],
     2: [0,0,1]
}

#step size, alpha
alpha = 1
steps = 100

print(W)

# start training
for s in range(steps):
    #iterate over every class
    for i in range(len(training_data)):
        #define the target vector, aka correct class
        t = target_vectors.get(i, [0,0,0])
        #iterate over every data point
        for j in range(len(training_data[i])):
            #perform matrix multiplication
            x = training_data[i][j]
            g = np.dot(W,x)
            g = sigmoid(g)
            MSE += MSE_function(g, t)
            gradMSE += grad_MSE_function(g,t,x,C)
    alpha = alpha*0.9
    W = W-alpha*gradMSE

print("AAAA")
print(W)
#print(gradMSE)