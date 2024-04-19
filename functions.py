import csv
import matplotlib as plt
import numpy as np

#data importer and preprocessing
def load_csv(filename, weight=True):
    #make data array
    data = []
    #open file
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        #iterate through data and append
        for row in csv_reader:
            #add a weight at the end of each sample if specified
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


def confusion_matrix(vector, target_vectors, target_vector):
    output_matrix = np.zeros(3,3)
    #if correct
    if vector==target_vector:
        placement = get_class_from_vector(vector, target_vectors)
        output_matrix[placement][placement] = 1
        return output_matrix
    #if false


def train_W(old_W, 
            training_data,
            alpha,
            class_to_vector = {
                0: [1,0,0],
                1: [0,1,0],
                2: [0,0,1]
            }
            ):
    C = len(training_data)
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
            g = np.dot(old_W,x)
            g = sigmoid(g)
            total_MSE += MSE_function(g, t)
            gradMSE += grad_MSE_function(g,t,x,C)
    # update the W matrix
    new_W = old_W-alpha*gradMSE
    return new_W

def test_W(W,
           testing_data,
           class_to_vector = {
                0: [1,0,0],
                1: [0,1,0],
                2: [0,0,1]
            }
           ):
    # make confusion matrix
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
    return confusion_matrix