import os
import pickle

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


if __name__ == "__main__":
    data_folder = os.path.join(ROOT_DIR, 'data')
    # print(data_folder)
