import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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

#train_images = load_mnist_images('MNist_ttt4275/train_images.bin')
#train_labels = load_mnist_labels('MNist_ttt4275/train_labels.bin')
#test_images = load_mnist_images('t10k-images-idx3-ubyte')
#test_labels = load_mnist_labels('t10k-labels-idx1-ubyte')
#print(train_images.shape)

#first_image = train_images[0]
#plt.imshow(first_image, cmap='gray')
#plt.axis('off')
#plt.show()

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

