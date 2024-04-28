import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from collections import Counter



def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        # Read the magic number and the number of images
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        # Read the dimensions of each image
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        # Read the image data
        raw_data = f.read()
    # Convert the raw data into a numpy array of shape (num_images, num_rows, num_cols)
    images = np.frombuffer(raw_data, dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    images = images.reshape(num_images, -1)
    return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        # Read the magic number and the number of labels
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        # Read the label data
        raw_data = f.read()
    # Convert the raw data into a numpy array of shape (num_labels,)
    labels = np.frombuffer(raw_data, dtype=np.uint8)
    return labels

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)  # Adjust font scale for better readability
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)

    # Customize tick labels
    tick_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.xticks(np.arange(9) + 0.5, tick_labels, rotation=0)
    plt.yticks(np.arange(9) + 0.5, tick_labels, rotation=0)

    plt.xlabel('Classified image')
    plt.ylabel('True image')
    plt.title('Confusion Matrix')
    plt.show()

def plot_number(number_vector, title=None):
    # Reshape the 784-long vector into a 28x28 matrix
    number_image = np.reshape(number_vector, (28, 28))
    # Display the image
    plt.imshow(number_image, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    if (title != None):
        plt.title(title)
    plt.show()

def create_sorted_matrix(train_labels, train_images):
    sorted_image_matrix = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(10):
        for j in range(len(train_labels)):
            if train_labels[j] == i:
                sorted_image_matrix[i].append(train_images[j])

    return sorted_image_matrix

def cluster(sorted_image_matrix, M=64):
    cluster_matrix = []
    for i in range(len(sorted_image_matrix)):
        #clustering image number i
        kmeans = KMeans(n_clusters=M)
        clusters = kmeans.fit(sorted_image_matrix[i]).cluster_centers_
        cluster_matrix.append(clusters)
    return cluster_matrix


def classifyKNN(test_images, test_labels, reference_images, reference_labels, K=1):
    #make confusion matrix
    confusion_matrix = np.zeros((10,10))
    errors = 0
    #images that was classified wrongly
    wrong_images = []
    wrong_labels = []
    #images that was used for a incorrect classification
    wrong_reference_images = []
    wrong_reference_labels = []
    #the differences in the wrong pictures
    differences = []

    for i in range(len(test_images)):
        #Contains all distances regarding image i
        distances = []
        actual_image = test_labels[i]
        for j in range(len(reference_images)):
            #Computing the forumla for NN based classifier
            difference = test_images[i] - reference_images[j]
            dist = np.inner(difference, difference)
            distances.append(dist)
        # start vote, and find nearest neighbour in case of even vote
        nearest_neighbour = None
        votes = [0,0,0,0,0,0,0,0,0,0]
        for j in range(K):
            #the closest value
            image_element = np.argmin(distances)
            #remove this value
            distances = np.delete(distances, image_element)
            #save the first value
            if j==0: nearest_neighbour = reference_labels[image_element]
            #increase votes
            votes[reference_labels[image_element]] += 1
        
        print(f'Votes: {votes}')
        # count occurances of the votes
        class_counts = Counter(votes)
        # find maximum count
        max_votes = max(class_counts.values())
        # try to classify image
        classified_image = None
        #if there are multiple max votes:
        if list(class_counts.values()).count(max_votes) > 1:
            classified_image = nearest_neighbour
        #if there is a winned
        else:
            classified_image = np.argmax(votes)
        print(f'Classified Image: {classified_image}')
        print(f'Actual Image: {actual_image}')

        if actual_image == classified_image:
            confusion_matrix[actual_image][actual_image] += 1
        else:
            confusion_matrix[actual_image][classified_image] += 1
            errors+=1
            #append images / labels
            wrong_images.append(test_images[i])
            wrong_labels.append(test_images[i])
            wrong_reference_images.append(reference_images[image_element])
            wrong_reference_labels.append(reference_labels[image_element])
            
            differences.append(abs(test_images[i] - reference_images[image_element]))

    error_rate = errors/len(test_images)
    return error_rate, confusion_matrix, wrong_images, wrong_labels, wrong_reference_images, wrong_reference_labels, differences