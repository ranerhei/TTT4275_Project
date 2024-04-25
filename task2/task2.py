import numpy as np

from functions2 import load_mnist_images
from functions2 import load_mnist_labels
from functions2 import plot_confusion_matrix
from functions2 import plot_number

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

#split data in a small chunk
test_images_chunk = test_images[1000:1010]
test_labels_chunk = test_labels[1000:1010]

#make confusion matrix
confusion_matrix = np.zeros((9,9))

for i in range(len(test_images_chunk)):
    #Contains all distances regarding image i
    distances = []
    actual_image = test_labels_chunk[i]
    for j in range(len(train_images)):
        #Computing the forumla for NN based classifier
        difference = test_images_chunk[i] - train_images[j]
        dist = np.inner(difference, difference)
        distances.append(dist)
        #print(dist)
    image_element = np.argmin(distances)
    classified_image = train_labels[image_element]
    if actual_image == classified_image:
        confusion_matrix[actual_image - 1][actual_image - 1] += 1
    else:
        confusion_matrix[actual_image - 1][classified_image - 1] += 1
