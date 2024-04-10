#!/usr/bin/python3

import csv
import matplotlib as plt
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

print(class_1[0])

D = len(class_1[0])-1
C = len(training_data)

W = np.ones((C, D+1))
print(W)

g = training_data[0][0] * W
print(g)
