import csv
import matplotlib as plt
import numpy as np

#data importer and preprocessing
#data has [sepal_length, sepal_width. petal_length, petal_width]
def load_csv(filename, weight=True, remove_sepal_width=False, remove_sepal_length=False, remove_petal_width=False, remove_petal_length=False):
    #make data array
    data = []
    # Open file
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        # Iterate through data and append
        for row in csv_reader:
            # Remove specified features from each row
            if remove_petal_width:
                del row[3]  # Remove petal width
            if remove_petal_length:
                del row[2]  # Remove petal length
            if remove_sepal_width:
                del row[1]  # Remove sepal width
            if remove_sepal_length:
                del row[0]  # Remove sepal length
            # Add a weight at the end of each sample if specified
            if weight:
                row.append(1.0)
            data.append([float(val) for val in row])
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

#Function to test W
def train_W(old_W, 
            training_data,
            alpha,
            MSE_history,
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
    MSE_history.append(total_MSE)
    return new_W, MSE_history

#Funtion to train W
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
    # make list to store the specific errors
    errors = []
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
            #if wrong
            if (rounded_g != t): 
                output_class = rounded_g.index(1)
                confusion_matrix[i][output_class] +=1
                errors.append([i,j])
            #if correct
            else:
                confusion_matrix[i][i] +=1
    return confusion_matrix, errors