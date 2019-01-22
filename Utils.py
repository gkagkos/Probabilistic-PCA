import os
import random
import random as rand
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_relative_error(data_original, data_created, num_points):
    """ Calculate the relative error between a collection of vectors
        data_train is a NxD matrix of original data
        data_reconstructed is the NxD matrix of approximated data
        num_points is the N number of data vectors
    """
    err = []
    for i in range(num_points - 1):
        error_per_point = np.linalg.norm(data_created[i] - data_original[i], ord=2) / (
            np.linalg.norm(data_original[i], ord=2))
        error = error_per_point * 100
        err.append(error)
    reconstruction_error = np.array(err)

    return reconstruction_error


def to_One_Hot(classification):
    """ Make your Labels one hot encoded (mnist labelling)
        Emulates the functionality of tf.keras.utils.to_categorical( y )
    """
    hotEncoding = np.zeros([len(classification),
                            np.max(classification) + 1])
    hotEncoding[np.arange(len(hotEncoding)), classification] = 1
    return hotEncoding


def get_missing_data(data_original):
    """ Return data with missining values replaced by zeros"""

    random.seed(10)

    data_original = data_original.astype(float)
    index = list(range(0, len(data_original[1])))
    for i in range(len(data_original)):
        picked_index = rand.sample(index, round(len(index) * 0.2))
        for j in range(len(picked_index)):
            data_original[i][picked_index[j]] = np.NaN

    return data_original


def get_missing_data_test(data_original):
    """ Return data with missining values replaced by NaN"""

    random.seed(1)

    random_indices = random.sample(range(1, 683), 136)

    data_original = data_original.astype(float)
    data_original = np.reshape(data_original, (-1))
    data_original[random_indices] = np.NaN
    data_original = np.reshape(data_original, (38, -1))

    return data_original


def get_missing_data2(data):
    """ Return data with missining values replaced by NaN"""
    data = data.astype(float)
    for i in range(len(data)):
        for j in range(len(data[0])):
            generator = random.randint(1, 101)
            if generator <= 20:
                data[i][j] = np.NaN
            # print(data[i][j])

    return data


def plot_colored_clusters(data, assignments):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Three component PPCA mixture model', fontsize=20)
    targets = [0, 1, 2]
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = np.where(assignments == target)[0]

        ax.scatter(data[indicesToKeep, 0]
                   , data[indicesToKeep, 1]
                   , c=color
                   , s=50)

    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(i + 10))

    plt.axis((-5, 6, -5, 6))

    ax.legend(['Model_0', 'Model_1', 'Model_2'])
    ax.grid()
    plt.show()


def plot_circles(data, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[y == 0, 0], data[y == 0, 1], color='red', alpha=0.5)
    plt.scatter(data[y == 1, 0], data[y == 1, 1], color='blue', alpha=0.5)

    plt.title(title)
    plt.text(-0.18, 0.18, 'gamma = 15', fontsize=12)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

def plot_clusters(data, indices_of_data):
    # print(data.shape)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], indices_of_data[i] + 10)

    plt.axis((-5, 6, -5, 6))

    plt.show()

if __name__ == "__main__":
    pass
    # data_folder = os.path.join(ROOT_DIR, 'data')
    # print(data_folder)
