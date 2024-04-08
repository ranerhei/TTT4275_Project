"""
we have
classes,        C=3
inputs,         D=4
training sets,  N=30
test sets,      M=20

"""

import csv
import matplotlib as plt

#data importer
def load_csv(filename):
    #make data array
    data = []
    #open file
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        #iterate through data and append
        for row in csv_reader:
            data.append([float(val) for val in row])
    return data

class_1 = load_csv('Iris_TTT4275/class_2')
class_2 = load_csv('Iris_TTT4275/class_2')
class_3 = load_csv('Iris_TTT4275/class_2')