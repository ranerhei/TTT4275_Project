"""
we have
classes,        C=3
inputs,         D=4
training sets,  N=30
test sets,      M=20

"""

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
from plot import plot_spider_web
from plot import paralell_plot

#load data
#data has [sepal_length, sepal_width. petal_length, petal_width]
class_1 = load_csv('Iris_TTT4275/class_1')
class_2 = load_csv('Iris_TTT4275/class_2')
class_3 = load_csv('Iris_TTT4275/class_3')

complete_data = [class_1, class_2, class_3]

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

#step size, alpha
alpha = 0.005
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

confusion_matrix = np.zeros((3,3))
# start testing:
# i, iterate over every class
for i in range(len(testing_data)):
    t = class_to_vector.get(i, [0,0,0])
    # j, iterate over every datapoint
    for j in range(len(testing_data[i])):
        #perform matrix multiplication
        x = testing_data[i][j]
        g = np.dot(W,x)
        g = sigmoid(g)
        rounded_g = round_calculated_vector(g)
        if (rounded_g != t): 
            output_class = rounded_g.index(i)
            confusion_matrix[i][output_class] +=1
        else:
            confusion_matrix[i][i] +=1
        
print(confusion_matrix)
plot_spider_web(W)
#plot_W_changes(W_history)
#paralell_plot(complete_data)