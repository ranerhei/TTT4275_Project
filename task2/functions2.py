import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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