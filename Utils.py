import os
import random
import random as rand
import numpy as np

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
    data_original = data_original.astype(float)
    index = list(range(0, len(data_original[1])))
    for i in range(len(data_original)):
        picked_index = rand.sample(index, round(len(index) * 0.2))
        for j in range(len(picked_index)):
            data_original[i][picked_index[j]] = np.NaN

    return data_original


def get_missing_data2(data):
    """ Return data with missining values replaced by NaN"""
    data = data.astype(float)
    for i in range(len(data)):
        for j in range(len(data[0])):
            generator = random.randint(1, 101)
            if generator <= 20:
                data[i][j] = np.NaN
            print(data[i][j])


if __name__ == "__main__":
    data_folder = os.path.join(ROOT_DIR, 'data')
    # print(data_folder)
