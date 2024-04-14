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
def grad_MSE_function(g,t,x,C):
    calculated_vector = np.multiply( np.multiply(g - t, g), 1-g)
    calculated_vector = np.reshape(calculated_vector, (C,1))
    return np.outer( calculated_vector, x)
<<<<<<< HEAD
=======

# function to round a calculated vector, to describe a certain class
def round_calculated_vector(vector):
    vector_list = vector.tolist()  # Convert NumPy array to list
    max_index = vector_list.index(max(vector))
    rounded_vector = [0] * len(vector)
    rounded_vector[max_index] = 1
    return rounded_vector

# function to get a class from a vector
def get_class_from_vector(vector, target_vectors):
    for class_num, target_vector in target_vectors.items():
        if vector == target_vector:
            return class_num
    return None  # Return None if the vector doesn't match any target vector
>>>>>>> 8f102c8244829f148aea0029d599af65fa0a30b7
