#!/usr/bin/python3

import csv
import matplotlib.pyplot as plt
import numpy as np

#data importer
def load_csv(filename):
    #make data array
    data = []
    #open file
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        #iterate through data and append
        for row in csv_reader:
            #add a weight at the end of each sample
            data.append([float(val) for val in row]+ [1.0])
    return data

#activator function, equation 20
def sigmoid(input_vector):
    #This vvvv is unstable :(
    #return 1 / (1 + np.exp(-input_vector))
    return np.exp(input_vector) / (1 + np.exp(input_vector))

#function to calculate Mean Square Error, equation 19
def MSE_function(g, t):
    return 0.5 * np.dot(np.transpose(g - t),g - t)

#function to calculate gradient of MSE, equation 22
def grad_MSE_function(g,t,x):
    calculated_vector = np.multiply( np.multiply(g - t, g), 1-g)
    calculated_vector = np.reshape(calculated_vector, (C,1))
    return np.outer( calculated_vector, x)

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


D = len(class_1[0])-1
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
alpha = 0.1
steps = 100

#print(W)

g = training_data[0][0] * W
#print(g)
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
            gradMSE += grad_MSE_function(g,t,x)
            print(gradMSE)
    alpha = alpha*0.9
    W = W-alpha*gradMSE

print("AAAA")
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
