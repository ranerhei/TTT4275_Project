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
from functions1 import load_csv

#activator function, equation 20
from functions1 import sigmoid

#function to calculate Mean Square Error, equation 19
from functions1 import MSE_function

#function to calculate gradient of MSE, equation 22
from functions1 import grad_MSE_function

from functions1 import get_class_from_vector

from functions1 import round_calculated_vector

from functions1 import train_W
from functions1 import test_W

from plot import plot_W_changes
from plot import plot_spider_web
from plot import paralell_plot
from plot import plot_MSE_history

#load data
#data has [sepal_length, sepal_width. petal_length, petal_width]
class_1 = load_csv('Iris_TTT4275/class_1')
class_2 = load_csv('Iris_TTT4275/class_2')
class_3 = load_csv('Iris_TTT4275/class_3')

complete_data = [class_1, class_2, class_3]

training_length = 30
#make training and testing set
class_1_training = class_1[:training_length]
class_1_testing = class_1[training_length:]

class_2_training = class_2[:training_length]
class_2_testing = class_2[training_length:]

class_3_training = class_3[:training_length]
class_3_testing = class_3[training_length:]

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
alpha = 0.01
steps = 2000

print('W before training:')
print(W)

# start training
for s in range(steps):
    W, MSE_history = train_W(W, training_data, alpha, MSE_history)
    W_history.append(W)

print('W after training:')


confusion_matrix, errors = test_W(W,training_data)
        
print(MSE_history[-1])
#plot_MSE_history(MSE_history, alpha)
#lot_spider_web(W)
#plot_W_changes(W_history)


