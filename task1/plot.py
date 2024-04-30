import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from functions1 import load_csv



#vector for dimensions
name_vector = {
     0: 'sepal length',
     1: 'sepal width',
     2: 'petal length',
     3: 'petal width'
}

def plot_histogram(data):
    # Get the number of dimensions
    num_dimensions = len(data[0][0])

    dimension_labels = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    # Create subplots for each dimension
    fig, axs = plt.subplots(num_dimensions, 1, figsize=(10, 8), sharex=True)
    # Plot histograms for each dimension of each class
    for dim in range(num_dimensions):
        axs[dim].hist([data[0][i][dim] for i in range(len(class_1))], alpha=0.5, label='Class 1', color='red')
        axs[dim].hist([data[1][i][dim] for i in range(len(class_2))], alpha=0.5, label='Class 2', color='green')
        axs[dim].hist([data[2][i][dim] for i in range(len(class_3))], alpha=0.5, label='Class 3', color='blue')
        axs[dim].set_title(f'{dimension_labels[dim]}')
        axs[dim].set_ylabel('Amount')
    # Add common x-axis label
    axs[num_dimensions - 1].set_xlabel('Value')
    # Add legend
    axs[0].legend()
    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.show()

def plot_histogram2(data):
    # Get the number of dimensions
    num_dimensions = len(data[0][0])

    dimension_labels = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    # Create subplots for each dimension
    fig, axs = plt.subplots(num_dimensions, 1, figsize=(10, 8))
    # Plot histograms for each dimension of each class
    for dim in range(num_dimensions):
        axs[dim].hist([data[0][i][dim] for i in range(len(class_1))], alpha=0.5, label='Class 1', color='red')
        axs[dim].hist([data[1][i][dim] for i in range(len(class_2))], alpha=0.5, label='Class 2', color='green')
        axs[dim].hist([data[2][i][dim] for i in range(len(class_3))], alpha=0.5, label='Class 3', color='blue')
        axs[dim].set_title(f'{dimension_labels[dim]}')
        axs[dim].set_ylabel('Amount')
        # Add separate x-axis for each subplot
        axs[dim].set_xlabel('Value')
        # Add legend
        axs[dim].legend()
    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.show()


def plot_dimensions_XY(data, name_vector):
    # make figure        
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    # need to iterate over the dimensions
    for i in range(len(data[0][0])):
        for j in range(len(data[0][0])):
            # if the dimensions are different
            if (j != i):
                axs[j, i].scatter([point[i] for point in data[0]], [point[j] for point in data[0]], color='red', label='Class 1')
                axs[j, i].scatter([point[i] for point in data[1]], [point[j] for point in data[1]], color='green', label='Class 2')
                axs[j, i].scatter([point[i] for point in data[2]], [point[j] for point in data[2]], color='blue', label='Class 3')
                axs[j,i].set_xlabel(name_vector.get(i))
                axs[j,i].set_ylabel(name_vector.get(j))
                axs[j,i].legend()
            # if the dimensions are equal, use the box to display text
            else:
                axs[j, i].text(0.5, 0.5, f'<<={name_vector.get(i)} =>>', horizontalalignment='center', verticalalignment='center', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_MSE_history(MSE_history, alpha):
    plt.plot(MSE_history, label=f'Total MSE at each iteration, alpha = {alpha}')
    plt.xlabel("Iteration")
    plt.ylabel("Total MSE")
    plt.legend()
    plt.show()

def plot_MSE_histories(MSE_histories, alphas):
    for i in range(len(alphas)):
        plt.plot(MSE_histories[i], label=f'Alpha = {alphas[i]}')
    plt.xlabel("Iteration")
    plt.ylabel("Total MSE")
    plt.title("Scenario 1")
    plt.legend()
    plt.show()

def plot_spider_web(W):
    num_dimensions = len(W[0])
    num_classes = len(W)

    angles = np.linspace(0, 2 * np.pi, num_dimensions, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i in range(num_classes):
        values = W[i].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, label=f'Class {i+1}')
        ax.fill(angles, values, alpha=0.1)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Constant'])
    ax.set_title('Spider Web Plot of Weights')
    ax.legend()

    plt.show()

def plot_parallel_coordinates(W):
    num_classes = len(W)
    num_dimensions = len(W[0])
    colors = ['red', 'green', 'blue']  # You can extend this list for more classes

    plt.figure(figsize=(10, 5))

    # Plot parallel coordinates for each class
    for i in range(num_classes):
        class_weights = W[i]
        for j in range(num_dimensions):
            feature_values = class_weights[:]
            plt.plot(range(num_dimensions), feature_values, color=colors[i], alpha=0.5)
        plt.plot([], [], color=colors[i], label=f'Class {i+1}')  # Dummy plot for legend

    plt.title('Parallel Coordinates Plot')
    plt.xlabel('Features')
    plt.ylabel('Feature Values')
    plt.xticks(range(num_dimensions), ['Petal Width', 'Constant'])  # Update feature names
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.show()

def paralell_plot(data):
    num_dimensions = len(data[0][0])
    num_classes = len(data)
    num_samples = len(data[0])

    # Create a color map for different classes
    colors = ['red', 'green', 'blue']



    # Plot parallel coordinates
    plt.figure(figsize=(10, 5))
    for i in range(num_classes):
        for j in range(num_samples-1):
            data[i][j].pop()
            plt.plot(range(num_dimensions-1), data[i][j], color=colors[i], alpha=0.5)
        data[i][-1].pop()
        plt.plot(range(num_dimensions-1), data[i][-1], color=colors[i], alpha=0.5, label=f'Class {i+1}')

    plt.title('Parallel Coordinates Plot')
    plt.xlabel('Features')
    plt.ylabel('Feature Values')
    plt.xticks(range(num_dimensions-1), ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.show()


def plot_W_changes(W_vector, option=1):
    class_labels = ['Class 1', 'Class 2', 'Class 3']
    dimension_labels = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Constant']

    if option==1:
        num_subplots = len(class_labels)
    elif option==2:
        num_subplots = len(dimension_labels)
    elif option==3:
        num_subplots = 1
    #create subplots
    #fig, axs = plt.subplots(num_subplots, 1, figsize=(10,8))
    
    # make vectors
    class_1_sepal_length = []
    class_1_sepal_width = []
    class_1_petal_length = []
    class_1_petal_width = []
    class_1_constant = []

    class_2_sepal_length = []
    class_2_sepal_width = []
    class_2_petal_length = []
    class_2_petal_width = []
    class_2_constant = []

    class_3_sepal_length = []
    class_3_sepal_width = []
    class_3_petal_length = []
    class_3_petal_width = []
    class_3_constant = []

    # iterate over all Ws
    for i in range(len(W_vector)):
        class_1_sepal_length.append(W_vector[i][0][0])
        class_1_sepal_width.append(W_vector[i][0][1])
        class_1_petal_length.append(W_vector[i][0][2])
        class_1_petal_width.append(W_vector[i][0][3])
        class_1_constant.append(W_vector[i][0][4])

        class_2_sepal_length.append(W_vector[i][1][0])
        class_2_sepal_width.append(W_vector[i][1][1])
        class_2_petal_length.append(W_vector[i][1][2])
        class_2_petal_width.append(W_vector[i][1][3])
        class_2_constant.append(W_vector[i][1][4])

        class_3_sepal_length.append(W_vector[i][2][0])
        class_3_sepal_width.append(W_vector[i][2][1])
        class_3_petal_length.append(W_vector[i][2][2])
        class_3_petal_width.append(W_vector[i][2][3])
        class_3_constant.append(W_vector[i][2][4])

    plt.plot(class_1_sepal_length, label='class 1 sepal length')
    plt.plot(class_1_sepal_width, label='class 1 sepal width')
    plt.plot(class_1_petal_length, label='class 1 petal length')
    plt.plot(class_1_petal_width, label='class 1 petal width')
    plt.plot(class_1_constant, label='class 1 constant')

    plt.plot(class_2_sepal_length, label='class 2 sepal length')
    plt.plot(class_2_sepal_width, label='class 2 sepal width')
    plt.plot(class_2_petal_length, label='class 2 petal length')
    plt.plot(class_2_petal_width, label='class 2 petal width')
    plt.plot(class_2_constant, label='class 2 constant')

    plt.plot(class_3_sepal_length, label='class 3 sepal length')
    plt.plot(class_3_sepal_width, label='class 3 sepal width')
    plt.plot(class_3_petal_length, label='class 3 petal length')
    plt.plot(class_3_petal_width, label='class 3 petal width')
    plt.plot(class_3_constant, label='class 3 constant')

    plt.xlabel('Iteration')
    plt.ylabel('Weighting')
    plt.legend()
    plt.tight_layout()
    plt.show()

#load data
#data has [sepal_length, sepal_width. petal_length, petal_width]
class_1 = load_csv('Iris_TTT4275/class_1', False)
class_2 = load_csv('Iris_TTT4275/class_2', False)
class_3 = load_csv('Iris_TTT4275/class_3', False)

data = [class_1,class_2,class_3]

#plot_histogram2(data)
#plot_dimensions_XY(data, name_vector)
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.6)  # Adjust font scale for better readability
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)

    # Customize tick labels
    tick_labels = ['Class 1', 'Class 2', 'Class 3']
    plt.xticks(np.arange(3) + 0.5, tick_labels, rotation=0)
    plt.yticks(np.arange(3) + 0.5, tick_labels, rotation=0)

    plt.xlabel('Classified Flower')
    plt.ylabel('True Flower')
    plt.title('Confusion Matrix: Scenario 1')
    plt.show()


def plot_decision_rule_and_data_1D(W, data):
    #remove the constant
    max_value = 0
    min_value = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j].pop()
            if data[i][j][0] > max_value: max_value = data[i][j][0]
            if data[i][j][0] < min_value: min_value = data[i][j][0]

    resolution = 10000
    x = np.linspace(min_value, max_value, resolution)

    class_1_function = []
    class_2_function = []
    class_3_function = []
    for i in range(len(x)):
        class_1_function.append(W[0][0]*x[i] + W[0][1])
        class_2_function.append(W[1][0]*x[i] + W[1][1])
        class_3_function.append(W[2][0]*x[i] + W[2][1])

    # Plot decision rule
    plt.plot(x, class_1_function, color='Red',  linewidth=2, label='$\mathregular{z_0 = W_{00}\cdot[Petal Width] + W_{01}}$')
    plt.plot(x, class_2_function, color='Green', linewidth=2, label='$\mathregular{z_1 = W_{10}\cdot[Petal Width] + W_{11}}$')
    plt.plot(x, class_3_function, color='Blue',  linewidth=2, label='$\mathregular{z_2 = W_{20}\cdot[Petal Width] + W_{21}}$')

    num_samples = len(data[0])
    
    y_coordinates = [0]*num_samples
    plt.scatter(data[0], y_coordinates,marker = 'o', label=f'Class 1', color='Red')
    plt.scatter(data[1], y_coordinates,marker = ',', label=f'Class 2', color='Green')
    plt.scatter(data[2], y_coordinates,marker = 'v', label=f'Class 3', color='Blue')

    plt.xlabel('Petal Width')
    plt.ylabel('Z value')
    plt.title('Decision Rule and Data in 1D')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_decision_rule_and_data_2D(W, data):
    #remove the constant
    max_value_x = data[0][0][0]
    min_value_x = data[0][0][0]
    max_value_y = data[0][0][1]
    min_value_y = data[0][0][1]
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j].pop()
            if data[i][j][0] > max_value_x: max_value_x = data[i][j][0]
            if data[i][j][0] < min_value_x: min_value_x = data[i][j][0]
            if data[i][j][1] > max_value_y: max_value_y = data[i][j][1]
            if data[i][j][1] < min_value_y: min_value_y = data[i][j][1]
    
    resolution = 10000
    x = np.linspace(min_value_x, max_value_x, resolution)
    y = np.linspace(min_value_y, max_value_y, resolution)
    
    EnOgTo = []
    EnOgTre = []
    ToOgTre = []
    for i in range(len(x)):
        EnOgTo.append(x[i]*(W[0][0]-W[1][0])/(W[1][1]-W[0][1]) +(W[0][2]-W[1][2])/(W[1][1]-W[0][1]))
        EnOgTre.append(x[i]*(W[0][0]-W[2][0])/(W[2][1]-W[0][1]) +(W[0][2]-W[2][2])/(W[2][1]-W[0][1]))
        ToOgTre.append(x[i]*(W[1][0]-W[2][0])/(W[2][1]-W[1][1]) +(W[1][2]-W[2][2])/(W[2][1]-W[1][1]))

    y_coordinates = [sample[1] for sample in data[0]]
    x_coordinates = [sample[0] for sample in data[0]]  # Extract x coordinates for class 1
    plt.scatter(x_coordinates, y_coordinates, marker='o', label='Class 1', color='Red')
    y_coordinates = [sample[1] for sample in data[1]]
    x_coordinates = [sample[0] for sample in data[1]]  # Extract x coordinates for class 2
    plt.scatter(x_coordinates, y_coordinates, marker=',', label='Class 2', color='Green')
    y_coordinates = [sample[1] for sample in data[2]]
    x_coordinates = [sample[0] for sample in data[2]]  # Extract x coordinates for class 3
    plt.scatter(x_coordinates, y_coordinates, marker='v', label='Class 3', color='Blue')

    plt.plot(x, EnOgTo, color='Red',  linewidth=2, label='$\mathregular{z_0 = z_1}$')
    plt.plot(x, EnOgTre, color='Green',  linewidth=2, label='$\mathregular{z_0 = z_2}$')
    plt.plot(x, ToOgTre, color='Blue',  linewidth=2, label='$\mathregular{z_1 = z_2}$')

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Decision Rule and Data in 2D')
    plt.legend()
    plt.grid(True)
    plt.show()