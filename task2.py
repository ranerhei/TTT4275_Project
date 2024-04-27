#!/usr/bin/python3
import numpy as np
from functions_mnist import load_mnist_images, load_mnist_labels, plot_confusion_matrix, plot_missclassified 
import seaborn as sns
from sklearn.metrics import confusion_matrix

train_images = load_mnist_images('MNist_ttt4275/train_images.bin')
train_labels = load_mnist_labels('MNist_ttt4275/train_labels.bin')
test_images = load_mnist_images('MNist_ttt4275/test_images.bin')
test_labels = load_mnist_labels('MNist_ttt4275/test_labels.bin')

#Normalize
train_images = train_images/255
test_images = test_images/255


test_images_chunk = test_images[3000:3200]
test_labels_chunk = test_labels[3000:3200]
train_images = train_images[0:20000]
train_labels = train_labels[0:20000]
confusion_matrix = np.zeros((10,10))


def NN_based_classifier(test_images_chunk, test_labels_chunk, train_images, train_labels):
    confusion_matrix = np.zeros((10,10))
    errors = 0
    misclassified_images = []
    true_images = []
    for i in range(len(test_images_chunk)):
        #Contains all distances regarding image i
        distances = []
        actual_image = test_labels_chunk[i]
        print("Sample nr: ", i)
        for j in range(len(train_images)):
            #Computing the forumla for NN based classifier
            difference = test_images_chunk[i] - train_images[j]
            dist = np.inner(difference, difference)
            distances.append(dist)
        #Finding the element which has the minimum distance, and corresponding image
        image_element = np.argmin(distances)
        classified_image = train_labels[image_element]
        #print("input image: " + str(actual_image))
        #print("classified: " + str(classified_image))
        #Creating confusion matrix
        if actual_image == classified_image:
            confusion_matrix[actual_image][actual_image] += 1
        else:
            confusion_matrix[actual_image][classified_image] += 1
            errors += 1
        error_rate = errors/len(test_images_chunk)*100
    
    return confusion_matrix, error_rate

confusion_matrix, error_rate = NN_based_classifier(test_images_chunk, test_labels_chunk, train_images, train_labels)
#plot_missclassified(true_images, misclassified_images)
print(confusion_matrix)
print("Error rate [%]: ", error_rate)
plot_confusion_matrix(confusion_matrix)
