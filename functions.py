import csv
import matplotlib as plt
import numpy as np

#data importer
def load_csv(filename, weight=True):
    #make data array
    data = []
    #open file
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        #iterate through data and append
        for row in csv_reader:
            #add a weight at the end of each sample
            if weight==False:
                data.append([float(val) for val in row])
            else:
                data.append([float(val) for val in row] + [1.0])
    return data

#activator function, equation 20
def sigmoid(input_vector):
    return np.exp(input_vector) / (1 + np.exp(input_vector))

#function to calculate Mean Square Error, equation 19
def MSE_function(calculated_vector, target_vector):
    return 0.5 * np.dot(np.transpose(calculated_vector - target_vector),calculated_vector - target_vector)

#function to calculate gradient of MSE, equation 22
def grad_MSE_function(g,t,x):
    calculated_vector = np.multiply( np.multiply(g - t, g), 1-g)
    calculated_vector = np.reshape(calculated_vector, (C,1))
    return np.outer( calculated_vector, x)
