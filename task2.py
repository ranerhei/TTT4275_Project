#!/usr/bin/python3

from functions_minst import load_mnist_images, load_mnist_labels


train_images = load_mnist_images('MNist_ttt4275/train_images.bin')
train_labels = load_mnist_labels('MNist_ttt4275/train_labels.bin')
test_images = load_mnist_images('t10k-images-idx3-ubyte')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte'


