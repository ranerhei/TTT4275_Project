import numpy as np

from functions2 import load_mnist_images
from functions2 import load_mnist_labels
from functions2 import plot_confusion_matrix
from functions2 import plot_number
from functions2 import create_sorted_matrix
from functions2 import cluster
from functions2 import classifyKNN

#load training data
train_images = load_mnist_images('task2/MNist_ttt4275/train_images.bin')
#load training data labels
train_labels = load_mnist_labels('task2/MNist_ttt4275/train_labels.bin')
#load testing data
test_images = load_mnist_images('task2/MNist_ttt4275/test_images.bin')
#load testing data labels
test_labels = load_mnist_labels('task2/MNist_ttt4275/test_labels.bin')


#Normalize values
train_images = train_images/255
test_images = test_images/255

# sort numbers
sorted_matrix = create_sorted_matrix(train_labels, train_images)
# create cluster matrix
cluster_matrix = cluster(sorted_matrix)
# reshape the cluster matrix
cluster_vector = [cluster for row in cluster_matrix for cluster in row]
# create labels for the cluster matrix
cluster_labels = [label for i in range(10) for label in [i] * 64]



#split data in a small chunk
test_images_chunk = test_images[1000:1050]
test_labels_chunk = test_labels[1000:1050]

error_rate, confusion_matrix, wrong_images, wrong_labels, reference_images, reference_labels, differences = classifyKNN(test_images, test_labels, cluster_vector, cluster_labels, K=2)

print(error_rate)
#plot_number(wrong_images[0], wrong_labels[0])
#plot_number(reference_images[0], reference_labels[0])
#plot_number(differences[0])
plot_number(cluster_vector[130])
plot_number(cluster_vector[135])
plot_number(cluster_vector[140])